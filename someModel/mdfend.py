import os
import tqdm
import torch
from transformers import BertModel

from utils.utils import data2gpu, Averager, metrics, Recorder
from .layers import *

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class MultiDomainFENDModel(torch.nn.Module):
    def __init__(self,emb_dim,mlp_dims,bert,dropout):
        super(MultiDomainFENDModel, self).__init__()
        self.num_expert = 5
        self.domain_num = 9
        self.bert = BertModel.from_pretrained(bert).requires_grad_(False)
        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        expert = []
        for i in range(self.num_expert):
            expert.append(cnn_extractor(emb_dim,feature_kernel))
        self.expert = torch.nn.ModuleList(expert)
        self.gate = torch.nn.Sequential(torch.nn.Linear(emb_dim*2,mlp_dims[-1]),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(mlp_dims[-1],self.num_expert),
                                        torch.nn.Softmax(dim = 1))
        self.attention = MaskAttention(emb_dim)
        self.domain_embedder = torch.nn.Embedding(num_embeddings=self.domain_num,embedding_dim = emb_dim)
        self.classifier = MLP(320,mlp_dims,dropout)

    def forward(self,**kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        category = kwargs['category']
        init_feature = self.bert(inputs,attention_mask = masks)[0]

        attention_feature = self.attention(init_feature,masks)
        idxs =torch.tensor([index for index in category]).view(-1,1).cuda()
        domain_embedding = self.domain_embedder(idxs).squeeze(1)

        gate_input = torch.cat([domain_embedding,attention_feature],dim = -1)
        gate_out = self.gate(gate_input)
        gate_expert = 0
        for i in range(self.num_expert):
            tmp_expert = self.expert[i](init_feature)
            gate_expert += (tmp_expert*gate_out[:,i].unsqueeze(1))

        label_pred = self.classifier(gate_expert)
        return torch.sigmoid(label_pred.squeeze(1)),torch.sigmoid(gate_expert)

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
        self.model = MultiDomainFENDModel(self.emb_dim, self.mlp_dims, self.bert, self.dropout)
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
                label_pred,tsnefeature = self.model(**batch_data)

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
                #plt.savefig('tsne_visualization.png', dpi=300)
                break
            else:
                continue
        self.model.load_state_dict(torch.load(os.path.join(self.save_param_dir, 'parameter_mdfend.pkl')))
        results = self.test(self.test_loader)
        self.tsne(self.val_loader)
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
                batch_label_pred,tsnefeature = self.model(**batch_data)

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_label_pred.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())

        return metrics(label, pred, category, self.category_dict)
    def tsne(self, dataloader):
        collected_data = []
        labels = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch)
                batch_label = batch_data['label']
                batch_category = batch_data['category']
                batch_label_pred,tsnefeature = self.model(**batch_data)
                print(tsnefeature.size())
                collected_data.append(tsnefeature.detach().cpu())
                labels.append(batch_label.detach().cpu())
        all_data = torch.cat(collected_data)
        print(all_data.size())
        all_data_np = all_data.numpy()

        all_labels = torch.cat(labels)
        all_data_labels = all_labels.numpy()
        # tsne = TSNE(n_components=2, perplexity=30, early_exaggeration=12, learning_rate=200, random_state=42)
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=0)
        tsne_results = tsne.fit_transform(all_data_np)

        # 可视化
        plt.figure(figsize=(5, 5))
        colors = ['#FFBCBB', '#B99CD4']
        for i, label0 in enumerate(np.unique(all_data_labels)):
            indices = np.where(all_data_labels == label0)
            plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], color=colors[i],
                        label=f'Class {label0}')
        plt.gca().set_axis_off()

        # 移除坐标轴保留的空间
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        # 保存图像时去除空白边距
        plt.savefig('tsne_visualization.png', dpi=300, bbox_inches='tight', pad_inches=0)

        # 显示图表（注意，显示出来的图表可能会因为 Jupyter Notebook 或其他IDE的显示设置而包含边距）
        plt.show()
        return 1