import os
import tqdm
import torch
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncodingPermute3D
from transformers import BertModel
import torch.nn as nn
# from positional_encodings.torch_encodings import PositionalEncoding1D
import models_mae
from utils.utils import data2gpu, Averager, metrics, Recorder
from .layers import *
from timm.models.vision_transformer import Block
class SimpleGate(nn.Module):
    def __init__(self, dim=1):
        super(SimpleGate, self).__init__()
        self.dim = dim

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=self.dim)
        return x1 * x2

class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def mu(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the average across
        it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""
        return torch.sum(x,(1))/(x.shape[1])

    def sigma(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the standard deviation
        across it's spatial dimensions as (h,w) tensor [See eq. 6 of paper] Note
        the permutations are required for broadcasting"""
        return torch.sqrt((torch.sum((x.permute([1,0])-self.mu(x)).permute([1,0])**2,(1))+0.000000023)/(x.shape[1]))

    def forward(self, x, mu, sigma):
        """ Takes a content embeding x and a style embeding y and changes
        transforms the mean and standard deviation of the content embedding to
        that of the style. [See eq. 8 of paper] Note the permutations are
        required for broadcasting"""
        # print(mu.shape) # 12
        x_mean = self.mu(x)
        x_std = self.sigma(x)
        x_reduce_mean = x.permute([1, 0]) - x_mean
        x_norm = x_reduce_mean/x_std
        # print(x_mean.shape) # 768, 12
        return (sigma.squeeze(1)*(x_norm + mu.squeeze(1))).permute([1,0])


class MultiDomainPLEFENDModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, bert, out_channels, dropout):
        super(MultiDomainPLEFENDModel, self).__init__()
        self.num_expert = 6
        self.domain_num = 8
        self.gate_num = 10
        self.num_share = 1
        self.unified_dim, self.text_dim = emb_dim, 768
        self.image_dim = 768
        self.bert = BertModel.from_pretrained(bert).requires_grad_(False)
        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        self.text_token_len = 197
        self.image_token_len = 197

        text_expert_list = []
        for i in range(self.domain_num):
            text_expert = []
            for j in range(self.num_expert):
                text_expert.append(cnn_extractor(emb_dim, feature_kernel))

            text_expert = nn.ModuleList(text_expert)
            text_expert_list.append(text_expert)
        self.text_experts = nn.ModuleList(text_expert_list)

        image_expert_list = []
        for i in range(self.domain_num):
            image_expert = []
            for j in range(self.num_expert):
                image_expert.append(cnn_extractor(self.image_dim, feature_kernel))
                #image_expert.append(image_cnn_extractor())
            image_expert = nn.ModuleList(image_expert)
            image_expert_list.append(image_expert)
        self.image_experts = nn.ModuleList(image_expert_list)

        fusion_expert_list = []
        for i in range(self.domain_num):
            fusion_expert = []
            for j in range(self.num_expert):
                expert = nn.Sequential(nn.Linear(320, 320),
                                       nn.SiLU(),
                                       #SimpleGate(),
                                       #nn.BatchNorm1d(160),
                                       nn.Linear(320, 320),
                                       )
                fusion_expert.append(expert)
            fusion_expert = nn.ModuleList(fusion_expert)
            fusion_expert_list.append(fusion_expert)
        self.fusion_experts = nn.ModuleList(fusion_expert_list)

        final_expert_list = []
        for i in range(self.domain_num):
            final_expert = []
            for j in range(self.num_expert):
                final_expert.append(Block(dim=320, num_heads=8))
            final_expert = nn.ModuleList(final_expert)
            final_expert_list.append(final_expert)
        self.final_experts = nn.ModuleList(final_expert_list)

        text_share_expert, image_share_expert, fusion_share_expert,final_share_expert = [], [], [],[]
        for i in range(self.num_share):
            text_share = []
            image_share = []
            fusion_share = []
            final_share = []
            for j in range(self.num_expert*2):
                text_share.append(cnn_extractor(emb_dim, feature_kernel))
                image_share.append(cnn_extractor(self.image_dim, feature_kernel))
                #image_share.append(image_cnn_extractor())
                expert = nn.Sequential(nn.Linear(320, 320),
                                       nn.SiLU(),
                                       #SimpleGate(),
                                       #nn.BatchNorm1d(160),
                                       nn.Linear(320, 320),
                                       )
                fusion_share.append(expert)
                final_share.append(Block(dim=320, num_heads=8))
            text_share = nn.ModuleList(text_share)
            text_share_expert.append(text_share)
            image_share = nn.ModuleList(image_share)
            image_share_expert.append(image_share)
            fusion_share = nn.ModuleList(fusion_share)
            fusion_share_expert.append(fusion_share)
            final_share = nn.ModuleList(final_share)
            final_share_expert.append(final_share)
        self.text_share_expert = nn.ModuleList(text_share_expert)
        self.image_share_expert = nn.ModuleList(image_share_expert)
        self.fusion_share_expert = nn.ModuleList(fusion_share_expert)
        self.final_share_expert = nn.ModuleList(final_share_expert)

        image_gate_list, text_gate_list, fusion_gate_list, fusion_gate_list0,final_gate_list = [], [], [], [],[]
        for i in range(self.domain_num):
            image_gate = nn.Sequential(nn.Linear(self.unified_dim * 2, self.unified_dim),
                                       nn.SiLU(),
                                       #SimpleGate(),
                                       #nn.BatchNorm1d(int(self.unified_dim / 2)),
                                       nn.Linear(self.unified_dim, self.num_expert * 3),
                                       nn.Dropout(0.1),
                                       nn.Softmax(dim=1)
                                       )
            text_gate = nn.Sequential(nn.Linear(self.unified_dim * 2, self.unified_dim),
                                      nn.SiLU(),
                                      #SimpleGate(),
                                      #nn.BatchNorm1d(int(self.unified_dim / 2)),
                                      nn.Linear(self.unified_dim, self.num_expert * 3),
                                      nn.Dropout(0.1),
                                      nn.Softmax(dim=1)
                                      )
            fusion_gate = nn.Sequential(nn.Linear(self.unified_dim * 2, self.unified_dim),
                                        nn.SiLU(),
                                        #SimpleGate(),
                                        #nn.BatchNorm1d(int(self.unified_dim / 2)),
                                        nn.Linear(self.unified_dim, self.num_expert * 4),
                                        nn.Dropout(0.1),
                                        nn.Softmax(dim=1)
                                        )
            fusion_gate0 = nn.Sequential(nn.Linear(320, 160),
                                         nn.SiLU(),
                                         #SimpleGate(),
                                         #nn.BatchNorm1d(80),
                                         nn.Linear(160, self.num_expert * 3),
                                         nn.Dropout(0.1),
                                         nn.Softmax(dim=1)
                                         )
            final_gate = nn.Sequential(nn.Linear(1088, 720),
                                        nn.SiLU(),
                                        #SimpleGate(),
                                        #nn.BatchNorm1d(int(self.unified_dim / 2)),
                                        nn.Linear(720, 160),
                                        nn.SiLU(),
                                        nn.Linear(160, self.num_expert * 3),
                                        nn.Dropout(0.1),
                                        nn.Softmax(dim=1)
                                         )
            image_gate_list.append(image_gate)
            text_gate_list.append(text_gate)
            fusion_gate_list.append(fusion_gate)
            fusion_gate_list0.append(fusion_gate0)
            final_gate_list.append(final_gate)
        self.image_gate_list = nn.ModuleList(image_gate_list)
        self.text_gate_list = nn.ModuleList(text_gate_list)
        self.fusion_gate_list = nn.ModuleList(fusion_gate_list)
        self.fusion_gate_list0 = nn.ModuleList(fusion_gate_list0)
        self.final_gate_list = nn.ModuleList(final_gate_list)

        #self.text_attention = TokenAttention(self.unified_dim)
        self.text_attention = MaskAttention(self.unified_dim)
        self.image_attention = TokenAttention(self.unified_dim)
        self.fusion_attention = TokenAttention(self.unified_dim * 2)
        self.final_attention = TokenAttention(320)

        self.domain_embedder = torch.nn.Embedding(num_embeddings=self.domain_num, embedding_dim=emb_dim)

        text_classifier_list = []

        for i in range(self.domain_num):
            text_classifier = MLP(320, mlp_dims, dropout)
            text_classifier_list.append(text_classifier)
        self.text_classifier_list = nn.ModuleList(text_classifier_list)

        image_classifier_list = []

        for i in range(self.domain_num):
            image_classifier = MLP(320, mlp_dims, dropout)
            image_classifier_list.append(image_classifier)
        self.image_classifier_list = nn.ModuleList(image_classifier_list)

        fusion_classifier_list = []

        for i in range(self.domain_num):
            fusion_classifier = MLP(320, mlp_dims, dropout)
            fusion_classifier_list.append(fusion_classifier)
        self.fusion_classifier_list = nn.ModuleList(fusion_classifier_list)

        final_classifier_list = []

        for i in range(self.domain_num):
            final_classifier = MLP(320, mlp_dims, dropout)
            final_classifier_list.append(final_classifier)
        self.final_classifier_list = nn.ModuleList(final_classifier_list)

        self.MLP_fusion = MLP_fusion(640, 320, [348], 0.1)
        self.domain_fusion = MLP_fusion(1088, 320, [348], 0.1)
        self.MLP_fusion0 = MLP_fusion(768 * 2, 768, [348], 0.1)

        self.model_size = "base"
        self.image_model = models_mae.__dict__["mae_vit_{}_patch16".format(self.model_size)](norm_pix_loss=False)
        self.image_model.cuda()
        checkpoint = torch.load('./mae_pretrain_vit_{}.pth'.format(self.model_size), map_location='cpu')
        self.image_model.load_state_dict(checkpoint['model'], strict=False)
        for param in self.image_model.parameters():
            param.requires_grad = False

        #### mapping MLPs
        self.mapping_IS_MLP_mu = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.mapping_IS_MLP_sigma = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim,1),
        )
        self.mapping_T_MLP_mu = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.mapping_T_MLP_sigma = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.mapping_IP_MLP_mu = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.mapping_IP_MLP_sigma = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.mapping_CC_MLP_mu = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.mapping_CC_MLP_sigma = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.adaIN = AdaIN()
        self.irrelevant_tensor = []
        for i in range(self.domain_num):
            self.irrelevant_tensor.append(nn.Parameter(torch.ones((1, 320)), requires_grad=True))

    def forward(self, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        category = kwargs['category']
        text_feature = self.bert(inputs, attention_mask=masks)[0]  # ([64, 197, 768])
        image = kwargs['image']
        image_feature = self.image_model.forward_ying(image)  # ([64, 197, 768])
        #image_feature = self.bert(inputs, attention_mask=masks)[0]
        #text_atn_feature, _ = self.text_attention(text_feature)  # ([64, 768])
        text_atn_feature = self.text_attention(text_feature,masks)
        image_atn_feature, _ = self.image_attention(image_feature)
        fusion_feature = torch.cat((image_feature, text_feature), dim=-1)
        fusion_atn_feature, _ = self.fusion_attention(fusion_feature)  # ([64, 1536])
        fusion_atn_feature = self.MLP_fusion0(fusion_atn_feature)
        # print("image_atn_feature", image_atn_feature.size())

        idxs = torch.tensor([index for index in category]).view(-1, 1).cuda()
        domain_embedding = self.domain_embedder(idxs).squeeze(1)  ##([32, 768])
        text_gate_input = torch.cat([domain_embedding, text_atn_feature], dim=-1)  # ([64, 1536])
        image_gate_input = torch.cat([domain_embedding, image_atn_feature], dim=-1)
        fusion_gate_input = torch.cat([domain_embedding, fusion_atn_feature], dim=-1)

        text_gate_out_list = []
        for i in range(self.domain_num):
            gate_out = self.text_gate_list[i](text_gate_input)
            text_gate_out_list.append(gate_out)
        self.text_gate_out_list = text_gate_out_list
        # self.text_gate_out_list = nn.ModuleList(text_gate_out_list)

        image_gate_out_list = []
        for i in range(self.domain_num):
            gate_out = self.image_gate_list[i](image_gate_input)
            image_gate_out_list.append(gate_out)
        self.image_gate_out_list = image_gate_out_list

        fusion_gate_out_list = []
        for i in range(self.domain_num):
            gate_out = self.fusion_gate_list[i](fusion_gate_input)
            fusion_gate_out_list.append(gate_out)
        self.fusion_gate_out_list = fusion_gate_out_list


        # text
        text_gate_expert_value = []
        text_gate_spacial_expert_value = []
        text_gate_share_expert_value = []
        for i in range(self.domain_num):
            gate_expert = 0
            gate_spacial_expert = 0
            gate_share_expert = 0
            for j in range(self.num_expert):
                tmp_expert = self.text_experts[i][j](text_feature)  # ([64, 320])
                gate_expert += (tmp_expert * text_gate_out_list[i][:, j].unsqueeze(1))  ##([64, 320]*[64, 1])
                gate_spacial_expert += (tmp_expert * text_gate_out_list[i][:, j].unsqueeze(1))
            for j in range(self.num_expert*2):
                tmp_expert = self.text_share_expert[0][j](text_feature)
                gate_expert += (tmp_expert * text_gate_out_list[i][:, (self.num_expert+j)].unsqueeze(1))
                gate_share_expert += (tmp_expert * text_gate_out_list[i][:, (self.num_expert+j)].unsqueeze(1))
            #print("gate_expert",gate_expert.size()) ([64, 320])
            text_gate_expert_value.append(gate_expert)
            text_gate_spacial_expert_value.append(gate_spacial_expert)
            text_gate_share_expert_value.append(gate_share_expert)

        image_gate_expert_value = []
        image_gate_spacial_expert_value = []
        image_gate_share_expert_value = []
        for i in range(self.domain_num):
            gate_expert = 0
            gate_spacial_expert = 0
            gate_share_expert = 0
            for j in range(self.num_expert):
                tmp_expert = self.image_experts[i][j](image_feature)  # ([64, 320])
                gate_expert += (tmp_expert * image_gate_out_list[i][:, j].unsqueeze(1))  ##([64, 320]*[64, 1])
                gate_spacial_expert += (tmp_expert * image_gate_out_list[i][:, j].unsqueeze(1))
            for j in range(self.num_expert*2):
                tmp_expert = self.image_share_expert[0][j](image_feature)
                gate_expert += (tmp_expert * image_gate_out_list[i][:, (self.num_expert+j)].unsqueeze(1))
                gate_share_expert += (tmp_expert * image_gate_out_list[i][:, (self.num_expert+j)].unsqueeze(1))
            # print("gate_expert",gate_expert.size()) ([64, 320])
            image_gate_expert_value.append(gate_expert)
            image_gate_spacial_expert_value.append(gate_spacial_expert)
            image_gate_share_expert_value.append(gate_share_expert)


        #fusion
        text_gate_share_expert_value = text_gate_share_expert_value[0]
        image_gate_share_expert_value = image_gate_share_expert_value[0]
        fusion_share_feature = torch.cat((text_gate_share_expert_value, image_gate_share_expert_value), dim=-1)
        fusion_share_feature = self.MLP_fusion(fusion_share_feature)
        fusion_gate_input0 = self.domain_fusion(torch.cat([domain_embedding, fusion_share_feature], dim=-1))
        fusion_gate_out_list0 = []
        for k in range(self.domain_num):
            gate_out = self.fusion_gate_list0[k](fusion_gate_input0)
            fusion_gate_out_list0.append(gate_out)
        self.fusion_gate_out_list0 = fusion_gate_out_list0

        fusion_gate_expert_value0 = []
        for m in range(self.domain_num):
            share_gate_expert0 = 0
            for n in range(self.num_expert):
                fusion_tmp_expert0 = self.fusion_experts[m][n](fusion_share_feature)
                share_gate_expert0 += (fusion_tmp_expert0 * self.fusion_gate_out_list0[m][:, n].unsqueeze(1))
            for n in range(self.num_expert * 2):
                fusion_tmp_expert0 = self.fusion_share_expert[0][n](fusion_share_feature)
                share_gate_expert0 += (
                        fusion_tmp_expert0 * self.fusion_gate_out_list0[m][:, (self.num_expert + n)].unsqueeze(1))
            fusion_gate_expert_value0.append(share_gate_expert0)

##continue

        #test
        text_only_output = []
        text_label_pred = []
        final_text_feature = []
        for i in range(self.domain_num):
            # label_pred.append(torch.sigmoid(self.text_classifier_list[i](text_gate_expert_value[i]).squeeze(1)))
            final_text_feature.append(text_gate_expert_value[i])
            text_class = self.text_classifier_list[i](text_gate_expert_value[i]).squeeze(1)
            text_only_output.append(text_class)
            pre = torch.sigmoid(text_class)
            text_label_pred.append(pre)
        text_label_pred_list = []
        text_label_pred_avg = 0
        for i in range(self.domain_num):
            text_label_pred_list.append(text_label_pred[i][idxs.squeeze() == i])
            text_label_pred_avg += text_label_pred[i]
        text_label_pred_avg = text_label_pred_avg / 8
        text_label_pred_list = torch.cat((text_label_pred_list[0], text_label_pred_list[1], text_label_pred_list[2], text_label_pred_list[3],
                                     text_label_pred_list[4], text_label_pred_list[5], text_label_pred_list[6], text_label_pred_list[7]))
        #image
        image_only_output = []
        image_label_pred = []
        final_image_feature = []
        for i in range(self.domain_num):
            # label_pred.append(torch.sigmoid(self.text_classifier_list[i](text_gate_expert_value[i]).squeeze(1)))
            final_image_feature.append(image_gate_expert_value[i])
            image_class = self.image_classifier_list[i](image_gate_expert_value[i]).squeeze(1)
            image_only_output.append(image_class)
            pre = torch.sigmoid(image_class)
            image_label_pred.append(pre)
        image_label_pred_list = []
        image_label_pred_avg = 0
        for i in range(self.domain_num):
            image_label_pred_list.append(image_label_pred[i][idxs.squeeze() == i])
            image_label_pred_avg += image_label_pred[i]
        image_label_pred_avg = image_label_pred_avg / 8

        image_label_pred_list = torch.cat((image_label_pred_list[0], image_label_pred_list[1], image_label_pred_list[2], image_label_pred_list[3],
                                     image_label_pred_list[4], image_label_pred_list[5], image_label_pred_list[6], image_label_pred_list[7]))

        # fusion
        fusion_only_output = []
        fusion_label_pred = []
        final_fusion_feature = []
        for i in range(self.domain_num):
            # label_pred.append(torch.sigmoid(self.text_classifier_list[i](text_gate_expert_value[i]).squeeze(1)))
            final_fusion_feature.append(fusion_gate_expert_value0[i])
            fusion_class = self.fusion_classifier_list[i](fusion_gate_expert_value0[i]).squeeze(1)
            fusion_only_output.append(fusion_class)
            pre = torch.sigmoid(fusion_class)
            fusion_label_pred.append(pre)
        fusion_label_pred_list = []
        fusion_label_pred_avg = 0
        for i in range(self.domain_num):
            fusion_label_pred_list.append(fusion_label_pred[i][idxs.squeeze() == i])
            fusion_label_pred_avg += fusion_label_pred[i]
        fusion_label_pred_avg = fusion_label_pred_avg / 8
        fusion_label_pred_list = torch.cat(
            (fusion_label_pred_list[0], fusion_label_pred_list[1], fusion_label_pred_list[2], fusion_label_pred_list[3],
             fusion_label_pred_list[4], fusion_label_pred_list[5], fusion_label_pred_list[6],
             fusion_label_pred_list[7]))

        fusion_atn_score = []
        image_mu = []
        text_mu = []
        fusion_mu = []
        image_sigma = []
        text_sigma = []
        fusion_sigma = []
        for i in range(self.domain_num):
            fusion_atn_score.append(1 - torch.sigmoid(fusion_only_output[i]).clone().detach())
            image_mu.append(self.mapping_IS_MLP_mu(torch.sigmoid(image_only_output[i]).clone().detach().view(-1,1)))
            text_mu.append(self.mapping_T_MLP_mu(torch.sigmoid(text_only_output[i]).clone().detach().view(-1,1)))
            fusion_mu.append(self.mapping_CC_MLP_mu(fusion_atn_score[i].clone().detach().view(-1,1)))  # 1-aux_atn_score
            image_sigma.append(self.mapping_IS_MLP_sigma(torch.sigmoid(image_only_output[i]).clone().detach().view(-1,1)))
            text_sigma.append(self.mapping_T_MLP_sigma(torch.sigmoid(text_only_output[i]).clone().detach().view(-1,1)))
            fusion_sigma.append(self.mapping_CC_MLP_sigma(fusion_atn_score[i].clone().detach().view(-1,1)))  # 1-aux_atn_score

        concat_feature_list = []
        for i in range(self.domain_num):
            final_image_feature0 = self.adaIN(final_image_feature[i],image_mu[i],image_sigma[i])  # shared_image_feature * (image_atn_score)
            final_text_feature0 = self.adaIN(final_text_feature[i], text_mu[i],
                                             text_sigma[i])  # shared_text_feature * (text_atn_score)
            final_fusion_feature0 = final_fusion_feature[i]  # shared_mm_feature #* (aux_atn_score)
            irr_score = torch.ones_like(final_fusion_feature0) * self.irrelevant_tensor[i].cuda()  # torch.ones_like(shared_mm_feature).cuda()
            irrelevant_token = self.adaIN(irr_score, fusion_mu[i], fusion_sigma[i])
            concat_feature_main_biased = torch.stack((final_image_feature0,
                                                      final_text_feature0,
                                                      final_fusion_feature0,
                                                      irrelevant_token
                                                      ), dim=1)#([64, 4, 320])
            concat_feature_list.append(concat_feature_main_biased)
        final_gate_out_list = []
        for i in range(self.domain_num):
            fusion_tempfeat_main_task, _ = self.final_attention(concat_feature_list[i])
            final_gate_input = torch.cat([domain_embedding, fusion_tempfeat_main_task], dim=-1)
            final_gate_out = self.final_gate_list[i](final_gate_input)
            final_gate_out_list.append(final_gate_out)

        final_gate_expert_value = []
        for i in range(self.domain_num):
            gate_expert = 0
            for j in range(self.num_expert):
                tmp_expert = self.final_experts[i][j](concat_feature_list[i])  # [64, 4, 320]
                tmp_expert = tmp_expert[:,0]
                gate_expert += (tmp_expert * final_gate_out_list[i][:, j].unsqueeze(1))  ##([64, 320]*[64, 1])
            for j in range(self.num_expert*2):
                tmp_expert = self.final_share_expert[0][j](concat_feature_list[i])
                tmp_expert = tmp_expert[:, 0]
                gate_expert += (tmp_expert * final_gate_out_list[i][:, (self.num_expert+j)].unsqueeze(1))
            # print("gate_expert",gate_expert.size()) ([64, 320])
            final_gate_expert_value.append(gate_expert)

        #final
        final_label_pred = []
        for i in range(self.domain_num):
            # label_pred.append(torch.sigmoid(self.text_classifier_list[i](text_gate_expert_value[i]).squeeze(1)))
            pre = torch.sigmoid(self.final_classifier_list[i](final_gate_expert_value[i]).squeeze(1))
            final_label_pred.append(pre)
        final_label_pred_list = []
        final_label_pred_avg = 0
        for i in range(self.domain_num):
            final_label_pred_list.append(final_label_pred[i][idxs.squeeze() == i])
            final_label_pred_avg += final_label_pred[i]
        final_label_pred_avg = final_label_pred_avg / 8
        final_label_pred_list = torch.cat((final_label_pred_list[0], final_label_pred_list[1], final_label_pred_list[2], final_label_pred_list[3],
                                     final_label_pred_list[4], final_label_pred_list[5], final_label_pred_list[6], final_label_pred_list[7]))



        #return final_label_pred_list, final_label_pred_avg,fusion_label_pred_list, fusion_label_pred_avg,image_label_pred_list, image_label_pred_avg,text_label_pred_list, text_label_pred_avg
        return final_label_pred_list,fusion_label_pred_list,image_label_pred_list,text_label_pred_list


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
                 loss_weight=[1, 0.006, 0.009, 5e-5],
                 early_stop=5,
                 epoches=100
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
        self.model = MultiDomainPLEFENDModel(self.emb_dim, self.mlp_dims, self.bert, 320, self.dropout)
        if self.use_cuda:
            self.model = self.model.cuda()
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.98)
        recorder = Recorder(self.early_stop)
        for epoch in range(self.epoches):
            self.model.train()
            train_data_iter = tqdm.tqdm(self.train_loader)
            avg_loss = Averager()
            for step_n, batch in enumerate(train_data_iter):
                batch_data = data2gpu(batch)
                label = batch_data['label']
                category = batch_data['category']
                idxs = torch.tensor([index for index in category]).view(-1, 1)
                batch_label = torch.cat((label[idxs.squeeze() == 0], label[idxs.squeeze() == 1],
                                         label[idxs.squeeze() == 2], label[idxs.squeeze() == 3],
                                         label[idxs.squeeze() == 4], label[idxs.squeeze() == 5],
                                         label[idxs.squeeze() == 6], label[idxs.squeeze() == 7]))

                final_label_pred_list,fusion_label_pred_list,image_label_pred_list,text_label_pred_list = self.model(**batch_data)
                loss0 = loss_fn(final_label_pred_list, batch_label.float())
                loss1 = loss_fn(fusion_label_pred_list, batch_label.float())
                loss2 = loss_fn(image_label_pred_list, batch_label.float())
                loss3 = loss_fn(text_label_pred_list, batch_label.float())
                loss = 0.7*loss0+0.1*loss1+0.1*loss2+0.1*loss3
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (scheduler is not None):
                    scheduler.step()
                avg_loss.add(loss.item())
            print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))
            results0,results1,results2,results3 = self.test(self.val_loader)
            mark = recorder.add(results0)
            if mark == 'save':
                torch.save(self.model.state_dict(),
                           os.path.join(self.save_param_dir, 'parameter_ple6.pkl'))
            elif mark == 'esc':
                break
            else:
                continue
        self.model.load_state_dict(torch.load(os.path.join(self.save_param_dir, 'parameter_ple6.pkl')))
        results0,results1,results2,results3 = self.test(self.test_loader)

        return results0, os.path.join(self.save_param_dir, 'parameter_ple6.pkl')

    def test(self, dataloader):
        pred0 = []
        pred1 = []
        pred2 = []
        pred3 = []
        label1 = []
        category = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch)
                label = batch_data['label']
                batch_category = batch_data['category']
                final_label_pred_list,fusion_label_pred_list,image_label_pred_list,text_label_pred_list= self.model(**batch_data)

                idxs = torch.tensor([index for index in batch_category]).view(-1, 1)
                batch_label_pred0 = final_label_pred_list
                batch_label_pred1 = fusion_label_pred_list
                batch_label_pred2 = image_label_pred_list
                batch_label_pred3 = text_label_pred_list


                batch_label = torch.cat((label[idxs.squeeze() == 0], label[idxs.squeeze() == 1],
                                         label[idxs.squeeze() == 2], label[idxs.squeeze() == 3],
                                         label[idxs.squeeze() == 4], label[idxs.squeeze() == 5],
                                         label[idxs.squeeze() == 6], label[idxs.squeeze() == 7]))
                batch_category = torch.sort(batch_category).values
                label1.extend(batch_label.detach().cpu().numpy().tolist())
                pred0.extend(batch_label_pred0.detach().cpu().numpy().tolist())
                pred1.extend(batch_label_pred1.detach().cpu().numpy().tolist())
                pred2.extend(batch_label_pred2.detach().cpu().numpy().tolist())
                pred3.extend(batch_label_pred3.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())

        return metrics(label1, pred0, category, self.category_dict),metrics(label1, pred1, category, self.category_dict),metrics(label1, pred2, category, self.category_dict),metrics(label1, pred3, category, self.category_dict)