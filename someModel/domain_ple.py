import os
import tqdm
import torch
from transformers import BertModel

from utils.utils import data2gpu, Averager, metrics, Recorder
from .layers import *


class MultiDomainPLEFENDModel(torch.nn.Module):
    def __init__(self,emb_dim,mlp_dims,bert,out_channels,dropout):
        super(MultiDomainPLEFENDModel, self).__init__()
        self.num_expert = 5
        self.domain_num = 8
        self.gate_num = 10
        self.bert = BertModel.from_pretrained(bert).requires_grad_(False)
        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        """
        expert_list = []
        for i in range(8):
            expert_list.append([])
            for i in range(self.num_expert):
                expert_list[i].append(cnn_extractor(emb_dim,feature_kernel))
        self.expert_list = torch.nn.ModuleList(expert_list)
        """

        expert1 = []
        for i in range(self.num_expert):
            expert1.append(cnn_extractor(emb_dim,feature_kernel))
        self.expert1 = torch.nn.ModuleList(expert1)
        expert2 = []
        for i in range(self.num_expert):
            expert2.append(cnn_extractor(emb_dim,feature_kernel))
        self.expert2 = torch.nn.ModuleList(expert2)
        expert3 = []
        for i in range(self.num_expert):
            expert3.append(cnn_extractor(emb_dim,feature_kernel))
        self.expert3 = torch.nn.ModuleList(expert3)
        expert4 = []
        for i in range(self.num_expert):
            expert4.append(cnn_extractor(emb_dim,feature_kernel))
        self.expert4 = torch.nn.ModuleList(expert4)
        expert5 = []
        for i in range(self.num_expert):
            expert5.append(cnn_extractor(emb_dim,feature_kernel))
        self.expert5 = torch.nn.ModuleList(expert5)
        expert6 = []
        for i in range(self.num_expert):
            expert6.append(cnn_extractor(emb_dim, feature_kernel))
        self.expert6 = torch.nn.ModuleList(expert6)
        expert7 = []
        for i in range(self.num_expert):
            expert7.append(cnn_extractor(emb_dim, feature_kernel))
        self.expert7 = torch.nn.ModuleList(expert7)
        expert8 = []
        for i in range(self.num_expert):
            expert8.append(cnn_extractor(emb_dim, feature_kernel))
        self.expert8 = torch.nn.ModuleList(expert8)

        expertShare = []
        for i in range(self.num_expert):
            expertShare.append(cnn_extractor(emb_dim, feature_kernel))
        self.expertShare = torch.nn.ModuleList(expertShare)



        self.gate1 = torch.nn.Sequential(torch.nn.Linear(emb_dim*2,mlp_dims[-1]),
                                        torch.nn.GELU(),
                                        torch.nn.Linear(mlp_dims[-1],self.gate_num),
                                        torch.nn.Softmax(dim = 1))
        self.gate2 = torch.nn.Sequential(torch.nn.Linear(emb_dim*2,mlp_dims[-1]),
                                        torch.nn.GELU(),
                                        torch.nn.Linear(mlp_dims[-1],self.gate_num),
                                        torch.nn.Softmax(dim = 1))
        self.gate3 = torch.nn.Sequential(torch.nn.Linear(emb_dim * 2, mlp_dims[-1]),
                                        torch.nn.GELU(),
                                        torch.nn.Linear(mlp_dims[-1], self.gate_num),
                                        torch.nn.Softmax(dim=1))
        self.gate4 = torch.nn.Sequential(torch.nn.Linear(emb_dim * 2, mlp_dims[-1]),
                                        torch.nn.GELU(),
                                        torch.nn.Linear(mlp_dims[-1], self.gate_num),
                                        torch.nn.Softmax(dim=1))
        self.gate5 = torch.nn.Sequential(torch.nn.Linear(emb_dim*2,mlp_dims[-1]),
                                        torch.nn.GELU(),
                                        torch.nn.Linear(mlp_dims[-1],self.gate_num),
                                        torch.nn.Softmax(dim = 1))
        self.gate6 = torch.nn.Sequential(torch.nn.Linear(emb_dim*2,mlp_dims[-1]),
                                        torch.nn.GELU(),
                                        torch.nn.Linear(mlp_dims[-1],self.gate_num),
                                        torch.nn.Softmax(dim = 1))
        self.gate7 = torch.nn.Sequential(torch.nn.Linear(emb_dim*2,mlp_dims[-1]),
                                        torch.nn.GELU(),
                                        torch.nn.Linear(mlp_dims[-1],self.gate_num),
                                        torch.nn.Softmax(dim = 1))
        self.gate8 = torch.nn.Sequential(torch.nn.Linear(emb_dim*2,mlp_dims[-1]),
                                        torch.nn.GELU(),
                                        torch.nn.Linear(mlp_dims[-1],self.gate_num),
                                        torch.nn.Softmax(dim = 1))


        self.attention = MaskAttention(emb_dim)
        self.domain_embedder = torch.nn.Embedding(num_embeddings=self.domain_num,embedding_dim = emb_dim)
        self.classifier1 = MLP(320,mlp_dims,dropout)
        self.classifier2 = MLP(320, mlp_dims, dropout)
        self.classifier3 = MLP(320, mlp_dims, dropout)
        self.classifier4 = MLP(320, mlp_dims, dropout)
        self.classifier5 = MLP(320, mlp_dims, dropout)
        self.classifier6 = MLP(320, mlp_dims, dropout)
        self.classifier7 = MLP(320, mlp_dims, dropout)
        self.classifier8 = MLP(320, mlp_dims, dropout)

        self.MLP_fusion = MLP_fusion(640, [348], 0.1)

        #image
        self.img_backbone = resnet18(pretrained=True)
        self.img_model = torch.nn.ModuleList([
            self.img_backbone.conv1,
            self.img_backbone.bn1,
            self.img_backbone.relu,
            self.img_backbone.layer1,
            self.img_backbone.layer2,
            self.img_backbone.layer3,
            self.img_backbone.layer4
        ])
        self.img_model = torch.nn.Sequential(*self.img_model)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.img_fc = torch.nn.Linear(self.img_backbone.inplanes, out_channels)




    def forward(self,**kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        category = kwargs['category']
        init_feature = self.bert(inputs,attention_mask = masks)[0]
        """
        img = kwargs['image']
        n_batch = img.size(0)
        img_out = self.img_model(img)
        img_out = self.avg_pool(img_out)
        img_out = img_out.view(n_batch, -1)
        img_out = self.img_fc(img_out)
        img_out = F.normalize(img_out, p=2, dim=-1)
        """
        attention_feature = self.attention(init_feature,masks)
        idxs =torch.tensor([index for index in category]).view(-1,1).cuda()
        domain_embedding = self.domain_embedder(idxs).squeeze(1)

        gate_input = torch.cat([domain_embedding,attention_feature],dim = -1)
        gate_out1 = self.gate1(gate_input)
        gate_out2 = self.gate2(gate_input)
        gate_out3 = self.gate3(gate_input)
        gate_out4 = self.gate4(gate_input)
        gate_out5 = self.gate5(gate_input)
        gate_out6 = self.gate6(gate_input)
        gate_out7 = self.gate7(gate_input)
        gate_out8 = self.gate8(gate_input)




        gate_expert1 = 0
        for i in range(self.num_expert):
            tmp_expert = self.expert1[i](init_feature)##64*320
            gate_expert1 += (tmp_expert*gate_out1[:,i].unsqueeze(1))   ##(64,320)*(64,1)

        for i in range(self.num_expert):
            share_expert = self.expertShare[i](init_feature)
            gate_expert1 += (share_expert*gate_out1[:,i+5].unsqueeze(1))

        gate_expert2 = 0
        for i in range(self.num_expert):
            tmp_expert = self.expert2[i](init_feature)
            gate_expert2 += (tmp_expert*gate_out2[:,i].unsqueeze(1))
        for i in range(self.num_expert):
            share_expert = self.expertShare[i](init_feature)
            gate_expert2 += (share_expert*gate_out2[:,i+5].unsqueeze(1))

        gate_expert3 = 0
        for i in range(self.num_expert):
            tmp_expert = self.expert3[i](init_feature)
            gate_expert3 += (tmp_expert*gate_out3[:,i].unsqueeze(1))
        for i in range(self.num_expert):
            share_expert = self.expertShare[i](init_feature)
            gate_expert3 += (share_expert*gate_out3[:,i+5].unsqueeze(1))

        gate_expert4 = 0
        for i in range(self.num_expert):
            tmp_expert = self.expert4[i](init_feature)
            gate_expert4 += (tmp_expert*gate_out4[:,i].unsqueeze(1))
        for i in range(self.num_expert):
            share_expert = self.expertShare[i](init_feature)
            gate_expert4 += (share_expert*gate_out4[:,i+5].unsqueeze(1))

        gate_expert5 = 0
        for i in range(self.num_expert):
            tmp_expert = self.expert5[i](init_feature)
            gate_expert5 += (tmp_expert*gate_out5[:,i].unsqueeze(1))
        for i in range(self.num_expert):
            share_expert = self.expertShare[i](init_feature)
            gate_expert5 += (share_expert*gate_out5[:,i+5].unsqueeze(1))

        gate_expert6 = 0
        for i in range(self.num_expert):
            tmp_expert = self.expert6[i](init_feature)
            gate_expert6 += (tmp_expert * gate_out6[:,i].unsqueeze(1))
        for i in range(self.num_expert):
            share_expert = self.expertShare[i](init_feature)
            gate_expert6 += (share_expert*gate_out6[:,i+5].unsqueeze(1))

        gate_expert7 = 0
        for i in range(self.num_expert):
            tmp_expert = self.expert7[i](init_feature)
            gate_expert7 += (tmp_expert * gate_out7[:,i].unsqueeze(1))
        for i in range(self.num_expert):
            share_expert = self.expertShare[i](init_feature)
            gate_expert7 += (share_expert*gate_out7[:,i+5].unsqueeze(1))

        gate_expert8 = 0
        for i in range(self.num_expert):
            tmp_expert = self.expert8[i](init_feature)
            gate_expert8 += (tmp_expert * gate_out8[:,i].unsqueeze(1))
        for i in range(self.num_expert):
            share_expert = self.expertShare[i](init_feature)
            gate_expert8 += (share_expert*gate_out8[:,i+5].unsqueeze(1))

        label_pred1 = torch.sigmoid(self.classifier1(gate_expert1).squeeze(1))
        label_pred2 = torch.sigmoid(self.classifier2(gate_expert2).squeeze(1))
        label_pred3 = torch.sigmoid(self.classifier3(gate_expert3).squeeze(1))
        label_pred4 = torch.sigmoid(self.classifier4(gate_expert4).squeeze(1))
        label_pred5 = torch.sigmoid(self.classifier5(gate_expert5).squeeze(1))
        label_pred6 = torch.sigmoid(self.classifier6(gate_expert6).squeeze(1))
        label_pred7 = torch.sigmoid(self.classifier7(gate_expert7).squeeze(1))
        label_pred8 = torch.sigmoid(self.classifier8(gate_expert8).squeeze(1))
        #print("label_pred1",label_pred1)
        #tensor([0.7390, 0.5795, 0.3497, 0.5553, 0.4634, 0.5212, 0.6553, 0.1990, 0.7332,0.3245, 0.5670, 0.7286, 0.4258, 0.5126, 0.4262, 0.6568]
        #print("label_pred1", label_pred1.size()) torch.Size([16])
        label_pred_list = []
        label_pred_list.append(label_pred1[idxs.squeeze()==0])
        label_pred_list.append(label_pred2[idxs.squeeze()==1])
        label_pred_list.append(label_pred3[idxs.squeeze()==2])
        label_pred_list.append(label_pred4[idxs.squeeze()==3])
        label_pred_list.append(label_pred5[idxs.squeeze()==4])
        label_pred_list.append(label_pred6[idxs.squeeze()==5])
        label_pred_list.append(label_pred7[idxs.squeeze()==6])
        label_pred_list.append(label_pred8[idxs.squeeze()==7])
        #print("label_pred_list",label_pred_list[0]) tensor([0.6553, 0.5670]
        #print("label_pred_list", label_pred_list[0].size()) torch.Size([2])
        label_pred = (label_pred1+label_pred2+label_pred3+label_pred4+label_pred5+label_pred6+label_pred7+label_pred8)/8
        label_pred_list = torch.cat((label_pred_list[0], label_pred_list[1], label_pred_list[2], label_pred_list[3], label_pred_list[4],label_pred_list[5], label_pred_list[6], label_pred_list[7]))

        return label_pred_list,label_pred

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