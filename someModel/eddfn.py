import os
import tqdm
import torch
from transformers import BertModel

from utils.utils import data2gpu, Averager, metrics, Recorder
from .layers import *

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class MLP(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)


class EDDFNModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims,dropout):
        super(EDDFNModel, self).__init__()
        self.bert = BertModel.from_pretrained('./pretrained_model/chinese_roberta_wwm_base_ext_pytorch').requires_grad_(False)


        self.shared_mlp = MLP(emb_dim, mlp_dims, dropout, False)
        self.specific_mlp = torch.nn.ModuleList([MLP(emb_dim, mlp_dims, dropout, False) for i in range(9)])
        self.decoder = MLP(mlp_dims[-1] * 2, (64, emb_dim), dropout, False)
        self.classifier = torch.nn.Linear(2 * mlp_dims[-1], 1)
        self.domain_classifier = nn.Sequential(MLP(mlp_dims[-1], mlp_dims, dropout, False), torch.nn.ReLU(),
                                               torch.nn.Linear(mlp_dims[-1], 9))
        self.attention = MaskAttention(emb_dim)

    def forward(self, alpha=1, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        category = kwargs['category']
        bert_feature = self.bert(inputs, attention_mask=masks).last_hidden_state
        bert_feature, _ = self.attention(bert_feature, masks)
        specific_feature = []
        for i in range(bert_feature.size(0)):
            specific_feature.append(self.specific_mlp[category[i]](bert_feature[i].view(1, -1)))
        specific_feature = torch.cat(specific_feature)
        shared_feature = self.shared_mlp(bert_feature)
        feature = torch.cat([shared_feature, specific_feature], 1)
        rec_feature = self.decoder(feature)
        output = self.classifier(feature)

        reverse = ReverseLayerF.apply
        domain_pred = self.domain_classifier(reverse(shared_feature, alpha))

        return torch.sigmoid(output.squeeze(1)), rec_feature, bert_feature, domain_pred

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
        self.model = EDDFNModel(self.emb_dim, self.mlp_dims, self.dropout)
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
            alpha = max(2. / (1. + np.exp(-10 * epoch / self.epoches)) - 1, 1e-1)
            loss_mse = torch.nn.MSELoss(reduce=True, size_average=True)
            loss_fn = torch.nn.BCELoss()
            for step_n, batch in enumerate(train_data_iter):
                batch_data = data2gpu(batch)
                label = batch_data['label']
                domain_label = batch_data['category']

                optimizer.zero_grad()
                pred, rec_feature, bert_feature, domain_pred = self.model(**batch_data, alpha=alpha)
                loss = loss_fn(pred, label.float()) + loss_mse(rec_feature, bert_feature) + 0.1 * F.nll_loss(F.log_softmax(domain_pred, dim=1), domain_label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
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
                batch_pred, _, __, ___ = self.model(**batch_data)

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_pred.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())

        return metrics(label, pred, category, self.category_dict)