import os
import tqdm
import torch
from transformers import BertModel
import torch.nn as nn
#from positional_encodings.torch_encodings import PositionalEncoding1D
import models_mae
from utils.utils import data2gpu, Averager, metrics, Recorder
from .layers import *
from timm.models.vision_transformer import Block

class MultiDomainPLEFENDModel(torch.nn.Module):
    def __init__(self,emb_dim,mlp_dims,bert,out_channels,dropout):
        super(MultiDomainPLEFENDModel, self).__init__()
        self.num_expert = 5
        self.domain_num = 8
        self.gate_num = 10
        self.num_share = 1
        self.unified_dim, self.text_dim = emb_dim, 768
        self.bert = BertModel.from_pretrained(bert).requires_grad_(False)
        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}


        image_expert_list, text_expert_list, mm_expert_list = [], [], []
        for i in range(self.domain_num):
            image_expert = []
            text_expert = []
            mm_expert = []
            for j in range(self.num_expert):
                image_expert.append(Block(dim=self.unified_dim, num_heads=8))
                text_expert.append(cnn_extractor(emb_dim,feature_kernel))
                mm_expert.append(Block(dim=self.unified_dim, num_heads=8))

            image_expert = nn.ModuleList(image_expert)
            image_expert_list.append(image_expert)
            text_expert = nn.ModuleList(text_expert)
            text_expert_list.append(text_expert)
            mm_expert = nn.ModuleList(mm_expert)
            mm_expert_list.append(mm_expert)

        self.image_experts = nn.ModuleList(image_expert_list)
        self.text_experts = nn.ModuleList(text_expert_list)
        self.mm_experts = nn.ModuleList(mm_expert_list)

        image_share_expert, text_share_expert = [],[]
        for i in range(self.num_share):
            image_share = []
            text_share = []
            for j in range(self.num_expert):
                image_share.append(Block(dim=self.unified_dim, num_heads=8))
                text_share.append(cnn_extractor(emb_dim,feature_kernel))
            image_share = nn.ModuleList(image_share)
            image_share_expert.append(image_share)
            text_share = nn.ModuleList(text_share)
            text_share_expert.append(text_share)
        self.image_share_expert = nn.ModuleList(image_share_expert)
        self.text_share_expert = nn.ModuleList(text_share_expert)


        image_gate_list, text_gate_list, mm_gate_list = [], [], []
        for i in range(self.domain_num):
            image_gate = nn.Sequential(nn.Linear(self.unified_dim*2,self.unified_dim),
                                            nn.SiLU(),
                                            # SimpleGate(),
                                            # nn.BatchNorm1d(int(self.unified_dim/2)),
                                            nn.Linear(self.unified_dim, self.domain_num+self.num_share),
                                            nn.Dropout(0.1),
                                            nn.Softmax(dim=1)
                                            )
            text_gate = nn.Sequential(nn.Linear(self.unified_dim*2,self.unified_dim),
                                            nn.SiLU(),
                                            # SimpleGate(),
                                            # nn.BatchNorm1d(int(self.unified_dim/2)),
                                            nn.Linear(self.unified_dim, self.domain_num+self.num_share),
                                            nn.Dropout(0.1),
                                            nn.Softmax(dim=1)
                                            )
            mm_gate = nn.Sequential(nn.Linear(self.unified_dim*2,self.unified_dim),
                                            nn.SiLU(),
                                            # SimpleGate(),
                                            # nn.BatchNorm1d(int(self.unified_dim/2)),
                                            nn.Linear(self.unified_dim, self.domain_num+self.num_share),
                                            nn.Dropout(0.1),
                                            nn.Softmax(dim=1)
                                            )
            image_gate_list.append(image_gate)
            text_gate_list.append(text_gate)
            mm_gate_list.append(mm_gate)
        self.image_gate_list = nn.ModuleList(image_gate_list)
        self.text_gate_list = nn.ModuleList(text_gate_list)
        self.mm_gate_list = nn.ModuleList(mm_gate_list)
        """"
        !!!!!!!!!!下面没改
        self.gate1 = torch.nn.Sequential(torch.nn.Linear(emb_dim*2,mlp_dims[-1]),
                                        torch.nn.GELU(),
                                        torch.nn.Linear(mlp_dims[-1],self.gate_num),
                                        torch.nn.Softmax(dim = 1))
        """
        self.text_attention = TokenAttention(self.unified_dim)
        self.image_attention = TokenAttention(self.unified_dim)
        self.mm_attention = TokenAttention(self.unified_dim)
        self.final_attention = TokenAttention(self.unified_dim)

        self.domain_embedder = torch.nn.Embedding(num_embeddings=self.domain_num,embedding_dim = emb_dim)

        text_classifier_list,image_classifier_list = [],[]

        for i in range(self.domain_num):
            text_classifier = MLP(320,mlp_dims,dropout)
            text_classifier_list.append(text_classifier)
            image_classifier = MLP(self.unified_dim,mlp_dims,dropout)
            image_classifier_list.append(image_classifier)
        self.text_classifier_list = nn.ModuleList(text_classifier_list)
        self.image_classifier_list = nn.ModuleList(image_classifier_list)

        self.MLP_fusion = MLP_fusion(640, [348], 0.1)

        #image
        self.model_size = "base"
        self.image_model = models_mae.__dict__["mae_vit_{}_patch16".format(self.model_size)](norm_pix_loss=False)
        checkpoint = torch.load('./mae_pretrain_vit_{}.pth'.format(self.model_size), map_location='cpu')
        self.image_model.load_state_dict(checkpoint['model'], strict=False)


    def forward(self,**kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        category = kwargs['category']
        text_feature = self.bert(inputs,attention_mask = masks)[0]

        image = kwargs['image']
        batch_size = image.shape[0]
        print("image",image.size())
        image_feature = self.image_model.forward_ying(image)
        print("image_feature",image_feature.size())
        text_atn_feature, _ = self.text_attention(text_feature)
        image_atn_feature, _ = self.image_attention(image_feature)
        mm_atn_feature, _ = self.mm_attention(torch.cat((image_feature, text_feature), dim=1))

        idxs =torch.tensor([index for index in category]).view(-1,1).cuda()
        domain_embedding = self.domain_embedder(idxs).squeeze(1)
        print(domain_embedding.size())#size!!!!
        text_gate_input = torch.cat([domain_embedding,text_atn_feature],dim = -1)
        image_gate_input = torch.cat([domain_embedding,image_atn_feature],dim=-1)


        text_gate_out_list = []
        for i in range(self.domain_num):
            gate_out = self.text_gate_list[i](text_gate_input)
            text_gate_out_list.append(gate_out)
        self.text_gate_out_list = nn.ModuleList(text_gate_out_list)

        image_gate_out_list = []
        for i in range(self.domain_num):
            gate_out = self.image_gate_list[i](image_gate_input)
            image_gate_out_list.append(gate_out)
        self.image_gate_out_list = nn.ModuleList(image_gate_out_list)

        text_gate_expert_value = []
        for i in range(self.domain_num):
            gate_expert = 0
            for j in range(self.num_expert):
                tmp_expert = self.text_experts[i][j](text_feature)
                gate_expert += (tmp_expert*text_gate_out_list[i][:,j].unsqueeze(1))##([64, 320]*[64, 1])
            for j in range(self.num_share):
                tmp_expert = self.text_share_expert[0][j](text_feature)
                gate_expert += (tmp_expert*text_gate_out_list[i][:,j].unsqueeze(1))
            text_gate_expert_value.append(gate_expert)
        #image
        image_gate_expert_value = []
        for i in range(self.domain_num):
            gate_expert = 0
            for j in range(self.num_expert):
                tmp_expert = self.image_experts[i][j](image_feature)
                print("tmp_expert",tmp_expert.size())####print
                print("image_gate_out_list",image_gate_out_list[i][:,j].unsqueeze(1).size())####print
                print("image_gate_out_list",image_gate_out_list[i][:, j].size())####print
                aaaaaaaa
                gate_expert += (tmp_expert*image_gate_out_list[i][:,j].unsqueeze(1))##([64, 320]*[64, 1])
            for j in range(self.num_share):
                tmp_expert = self.image_share_expert[0][j](text_feature)
                gate_expert += (tmp_expert*image_gate_out_list[i][:,j].unsqueeze(1))
            image_gate_expert_value.append(gate_expert)

        label_pred = []
        for i in range(self.domain_num):
            label_pred.append(torch.sigmoid(self.text_classifier_list[i](text_gate_expert_value[i]).squeeze(1)))

        label_pred_list = []
        label_pred_avg = 0
        for i in range(self.domain_num):
            label_pred_list.append(label_pred[i][idxs.squeeze()==i])
            label_pred_avg+=label_pred[i]
        label_pred_avg =  label_pred_avg/8

        label_pred_list = torch.cat((label_pred_list[0], label_pred_list[1], label_pred_list[2], label_pred_list[3], label_pred_list[4],label_pred_list[5], label_pred_list[6], label_pred_list[7]))
        return label_pred_list,label_pred_avg

class Trainer():
    def __init__(self,
                 emb_dim,
                 mlp_dims,
                 bert,
                 use_cuda,
                 lr,
                 dropout,
                 train_loader,
                 val_loader,
                 test_loader,
                 category_dict,
                 weight_decay,
                 save_param_dir,
                 loss_weight = [1, 0.006, 0.009, 5e-5],
                 early_stop = 5,
                 epoches = 100
                 ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.early_stop = early_stop
        self.epoches = epoches
        self.category_dict = category_dict
        self.loss_weight = loss_weight
        self.use_cuda = use_cuda

        self.emb_dim = emb_dim
        self.mlp_dims = mlp_dims
        self.bert = bert
        self.dropout = dropout
        if not os.path.exists(save_param_dir):
            self.save_param_dir = os.makedirs(save_param_dir)
        else:
            self.save_param_dir = save_param_dir

    def train(self):
        self.model = MultiDomainPLEFENDModel(self.emb_dim, self.mlp_dims, self.bert,320,self.dropout)
        if self.use_cuda:
            self.model = self.model.cuda()
        loss_fn =torch.nn.BCELoss()
        optimizer =torch.optim.Adam(params=self.model.parameters(),lr = self.lr,weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.98)
        recorder = Recorder(self.early_stop)
        for epoch in range(self.epoches):
            self.model.train()
            train_data_iter = tqdm.tqdm(self.train_loader)
            avg_loss = Averager()
            for step_n,batch in enumerate(train_data_iter):
                batch_data = data2gpu(batch)
                label = batch_data['label']
                category = batch_data['category']
                idxs = torch.tensor([index for index in category]).view(-1, 1)
                batch_label = torch.cat((label[idxs.squeeze() == 0], label[idxs.squeeze() == 1],
                                         label[idxs.squeeze() == 2], label[idxs.squeeze() == 3],
                                         label[idxs.squeeze() == 4], label[idxs.squeeze() == 5],
                                         label[idxs.squeeze() == 6], label[idxs.squeeze() == 7]))

                label_pred_list,label_pred = self.model(**batch_data)

                loss = loss_fn(label_pred_list,batch_label.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if(scheduler is not None):
                    scheduler.step()
                avg_loss.add(loss.item())
            print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))
            results = self.test(self.val_loader)
            mark = recorder.add(results)
            if mark == 'save':
                torch.save(self.model.state_dict(),
                           os.path.join(self.save_param_dir, 'parameter_mdfend.pkl'))
            elif mark == 'esc':
                break
            else:
                continue
        self.model.load_state_dict(torch.load(os.path.join(self.save_param_dir, 'parameter_mdfend.pkl')))
        results = self.test(self.test_loader)
        print(results)
        return results, os.path.join(self.save_param_dir, 'parameter_mdfend.pkl')

    def test(self, dataloader):
        pred = []
        label1 = []
        category = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch)
                label = batch_data['label']
                batch_category = batch_data['category']
                label_pred_list,_ = self.model(**batch_data)

                idxs = torch.tensor([index for index in batch_category]).view(-1, 1)
                batch_label_pred = label_pred_list
                batch_label = torch.cat((label[idxs.squeeze()==0],label[idxs.squeeze()==1],label[idxs.squeeze()==2],label[idxs.squeeze()==3],label[idxs.squeeze()==4],label[idxs.squeeze()==5],label[idxs.squeeze()==6],label[idxs.squeeze()==7]))
                batch_category = torch.sort(batch_category).values
                label1.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_label_pred.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())

        return metrics(label1, pred, category, self.category_dict)