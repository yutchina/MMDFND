import datetime
import math
import numpy as np
import torch
from torch import nn, backends
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch.sparse
from scipy.sparse import coo
import time
from pivot import *
class model(torch.nn.Module):
    def __init__(self):
        super(model, self).__init__()
        feature_emb_size = 320
        img_emb_size = 320
        feature_num = 4
        self.feature_num = 4
        text_emb_size = 320
        self.n_node = 64
        self.feature_emb_size = 320
        self.emb_size = 320
        self.layers = 6
        self.transformers = torch.nn.ModuleList([TransformerLayer(feature_emb_size, head_num=4, dropout=0.6,
                                                                  attention_dropout=0,
                                                                  initializer_range=0.02) for _ in
                                                 range(6)])
        self.mlp_img = torch.nn.ModuleList([MLP_trans(img_emb_size, feature_emb_size, dropout=0.6) for _ in
                                            range(feature_num)])

        self.mlp_text = torch.nn.ModuleList([MLP_trans(text_emb_size, feature_emb_size, dropout=0.6) for _ in
                                             range(feature_num)])
        self.mlp_fusion = torch.nn.ModuleList([MLP_trans(text_emb_size, feature_emb_size, dropout=0.6) for _ in
                                             range(feature_num)])
        self.star_emb1 = nn.Embedding(self.n_node, self.feature_emb_size)
        self.star_emb2 = nn.Embedding(self.n_node, self.feature_emb_size)
        self.star_emb3 = nn.Embedding(self.n_node, self.feature_emb_size)
        self.star_emb4 = nn.Embedding(self.n_node, self.feature_emb_size)
        torch.nn.init.normal_(self.star_emb1.weight, 0, 0.02)
        torch.nn.init.normal_(self.star_emb2.weight, 0, 0.02)
        torch.nn.init.normal_(self.star_emb3.weight, 0, 0.02)
        torch.nn.init.normal_(self.star_emb4.weight, 0, 0.02)
        self.active = nn.SiLU()
        self.dropout2 = nn.Dropout(0.2)
        self.mlp_star_f1 = nn.Linear(self.feature_emb_size * 4, self.emb_size)
        self.mlp_star_f2 = nn.Linear(self.emb_size, self.emb_size)
    def fusion_img_text(self, image_emb, text_emb,fusion_emb):
        for img_feature_num in range(0, self.feature_num):
            if img_feature_num == 0:
                img_feature_seq = self.mlp_img[img_feature_num](image_emb)
                img_feature_seq = img_feature_seq.unsqueeze(1)
            else:
                img_feature_seq = torch.cat((img_feature_seq, self.mlp_img[img_feature_num](image_emb).unsqueeze(1)), 1)

        for text_feature_num in range(0, self.feature_num):
            if text_feature_num == 0:
                text_feature_seq = self.mlp_text[text_feature_num](text_emb)
                text_feature_seq = text_feature_seq.unsqueeze(1)
            else:
                text_feature_seq = torch.cat((text_feature_seq, self.mlp_text[text_feature_num](text_emb).unsqueeze(1)), 1)

        for text_feature_num in range(0, self.feature_num):
            if text_feature_num == 0:
                fusion_feature_seq = self.mlp_fusion[text_feature_num](fusion_emb)
                fusion_feature_seq = fusion_feature_seq.unsqueeze(1)
            else:
                fusion_feature_seq = torch.cat((fusion_feature_seq, self.mlp_fusion[text_feature_num](fusion_emb).unsqueeze(1)), 1)
        #batch = image_emb.size()[0]
        #star_emb1 = nn.Parameter(torch.ones((batch, 320)), requires_grad=True)
        #star_emb2 = nn.Parameter(torch.ones((batch, 320)), requires_grad=True)
        #star_emb3 = nn.Parameter(torch.ones((batch, 320)), requires_grad=True)
        #star_emb4 = nn.Parameter(torch.ones((batch, 320)), requires_grad=True)

        for sa_i in range(0, int(self.layers), 3):
            trans_text_item = torch.cat(
                [star_emb1.unsqueeze(1), star_emb2.unsqueeze(1), star_emb3.unsqueeze(1), star_emb4.unsqueeze(1),  text_feature_seq], 1)
            text_output = self.transformers[sa_i + 2](trans_text_item)

            star_emb1 = (text_output[:, 0, :] + star_emb1)/2
            star_emb2 = (text_output[:, 1, :] + star_emb2)/2
            star_emb3 = (text_output[:, 2, :] + star_emb3)/2
            star_emb4 = (text_output[:, 3, :] + star_emb4)/2
            text_feature_seq = text_output[:, 4:self.feature_num+4, :] + text_feature_seq

            trans_img_item = torch.cat(
                [star_emb1.unsqueeze(1), star_emb2.unsqueeze(1), star_emb3.unsqueeze(1),star_emb4.unsqueeze(1),
                 img_feature_seq], 1)
            img_output = self.transformers[sa_i+1](trans_img_item)
            star_emb1 = (img_output[:, 0, :] + star_emb1) / 2
            star_emb2 = (img_output[:, 1, :] + star_emb2) / 2
            star_emb3 = (img_output[:, 2, :] + star_emb3) / 2
            star_emb4 = (img_output[:, 3, :] + star_emb4) / 2
            img_feature_seq = img_output[:, 4:self.feature_num + 4, :] + img_feature_seq

            trans_fusion_item = torch.cat(
                [star_emb1.unsqueeze(1), star_emb2.unsqueeze(1), star_emb3.unsqueeze(1),star_emb4.unsqueeze(1),
                 fusion_feature_seq], 1)
            fusion_output = self.transformers[sa_i](trans_fusion_item)
            star_emb1 = (fusion_output[:, 0, :] + star_emb1) / 2
            star_emb2 = (fusion_output[:, 1, :] + star_emb2) / 2
            star_emb3 = (fusion_output[:, 2, :] + star_emb3) / 2
            star_emb4 = (fusion_output[:, 3, :] + star_emb4) / 2
            fusion_feature_seq = fusion_output[:, 4:self.feature_num + 4, :] + fusion_feature_seq

        item_emb_trans = self.dropout2(torch.cat([star_emb1, star_emb2, star_emb3,star_emb4], 1))
        item_emb_trans = self.dropout2(self.active(self.mlp_star_f1(item_emb_trans)))
        item_emb_trans = self.dropout2(self.active(self.mlp_star_f2(item_emb_trans)))
        return item_emb_trans
    def forward(self,image_emb,text_emb,fusion_emb):
        batch = image_emb.size()[0]
        item_emb_final = self.fusion_img_text(image_emb, text_emb,fusion_emb)
        print(item_emb_final.size())
        return item_emb_final
image_emb = torch.rand(64,320)
text_emb = torch.rand(64,320)
fusion_emb = torch.rand(64,320)
model = model()
model(image_emb,text_emb,fusion_emb)
