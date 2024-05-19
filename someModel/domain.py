import os
import tqdm
import torch
from transformers import BertModel

from utils.utils import data2gpu, Averager, metrics, Recorder
from .layers import *


class MultiDomainFENDModel(torch.nn.Module):
    def __init__(self,emb_dim,mlp_dims,bert,out_channels,dropout):
        super(MultiDomainFENDModel, self).__init__()
        self.num_expert = 5
        self.domain_num = 9
        self.gate_num = 7
        self.bert = BertModel.from_pretrained(bert).requires_grad_(False)
        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        expert = []
        for i in range(self.num_expert):
            expert.append(cnn_extractor(emb_dim,feature_kernel))
        self.expert = torch.nn.ModuleList(expert)
        self.gate = torch.nn.Sequential(torch.nn.Linear(emb_dim*2,mlp_dims[-1]),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(mlp_dims[-1],self.gate_num),
                                        torch.nn.Softmax(dim = 1))
        self.attention = MaskAttention(emb_dim)
        self.domain_embedder = torch.nn.Embedding(num_embeddings=self.domain_num,embedding_dim = emb_dim)
        self.classifier = MLP(320,mlp_dims,dropout)
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

        img = kwargs['image']
        n_batch = img.size(0)
        img_out = self.img_model(img)
        img_out = self.avg_pool(img_out)
        img_out = img_out.view(n_batch, -1)
        img_out = self.img_fc(img_out)
        img_out = F.normalize(img_out, p=2, dim=-1)

        attention_feature = self.attention(init_feature,masks)
        idxs =torch.tensor([index for index in category]).view(-1,1).cuda()
        domain_embedding = self.domain_embedder(idxs).squeeze(1)

        gate_input = torch.cat([domain_embedding,attention_feature],dim = -1)
        gate_out = self.gate(gate_input)
        gate_expert = 0
        for i in range(self.num_expert-1):
            tmp_expert = self.expert[i](init_feature)
            gate_expert += (tmp_expert*gate_out[:,i].unsqueeze(1))

        text_out = self.expert[self.num_expert-1](init_feature)
        text_image = self.MLP_fusion(torch.cat((text_out, img_out), 1))
        gate_expert += (text_image*gate_out[:,4].unsqueeze(1))
        gate_expert += (img_out*gate_out[:,5].unsqueeze(1))
        gate_expert += (text_out*gate_out[:,6].unsqueeze(1))
        label_pred = self.classifier(gate_expert)
        return torch.sigmoid(label_pred.squeeze(1))

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
        self.model = MultiDomainFENDModel(self.emb_dim, self.mlp_dims, self.bert,320,self.dropout)
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
                label_pred = self.model(**batch_data)
                loss = loss_fn(label_pred,label.float())
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
        label = []
        category = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch)
                batch_label = batch_data['label']
                batch_category = batch_data['category']
                batch_label_pred = self.model(**batch_data)

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_label_pred.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())

        return metrics(label, pred, category, self.category_dict)