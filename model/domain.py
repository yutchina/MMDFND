import os
import tqdm
import torch
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncodingPermute3D
from transformers import BertModel
import torch.nn as nn
# from positional_encodings.torch_encodings import PositionalEncoding1D
import models_mae
from utils.utils import data2gpu, Averager, metrics, Recorder, clipdata2gpu
from utils.utils import metricsTrueFalse
from .layers import *
from .pivot import *
from timm.models.vision_transformer import Block
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
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
        self.task_num = 2
        #self.domain_num = 9
        self.domain_num = self.task_num
        self.gate_num = 3
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
            image_gate = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                       nn.SiLU(),
                                       #SimpleGate(),
                                       #nn.BatchNorm1d(int(self.unified_dim / 2)),
                                       nn.Linear(self.unified_dim, self.num_expert * 3),
                                       nn.Dropout(0.1),
                                       nn.Softmax(dim=1)
                                       )
            text_gate = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                      nn.SiLU(),
                                      #SimpleGate(),
                                      #nn.BatchNorm1d(int(self.unified_dim / 2)),
                                      nn.Linear(self.unified_dim, self.num_expert * 3),
                                      nn.Dropout(0.1),
                                      nn.Softmax(dim=1)
                                      )
            fusion_gate = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
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
            final_gate = nn.Sequential(nn.Linear(320, 320),
                                        nn.SiLU(),
                                        #SimpleGate(),
                                        #nn.BatchNorm1d(int(self.unified_dim / 2)),
                                        nn.Linear(320, 160),
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

        self.text_classifier = MLP(320, mlp_dims, dropout)
        self.text_classifier_Mu = MLP_Mu(320, mlp_dims, dropout)
        self.image_classifier = MLP(320, mlp_dims, dropout)
        self.image_classifier_Mu = MLP_Mu(320, mlp_dims, dropout)
        self.fusion_classifier = MLP(320, mlp_dims, dropout)
        self.fusion_classifier_Mu = MLP_Mu(320, mlp_dims, dropout)

        self.max_classifier = MLP(640, mlp_dims, dropout)

        share_classifier_list = []

        for i in range(self.domain_num):
            share_classifier = MLP(320, mlp_dims, dropout)
            share_classifier_list.append(share_classifier)
        self.share_classifier_list = nn.ModuleList(share_classifier_list)

        dom_classifier_list = []

        for i in range(self.domain_num):
            dom_classifier = MLP(320, mlp_dims, dropout)
            dom_classifier_list.append(dom_classifier)
        self.dom_classifier_list = nn.ModuleList(dom_classifier_list)



        final_classifier_list = []

        for i in range(self.domain_num):
            final_classifier = MLP(320, mlp_dims, dropout)
            final_classifier_list.append(final_classifier)
        self.final_classifier_list = nn.ModuleList(final_classifier_list)

        self.MLP_fusion = MLP_fusion(960, 320, [348], 0.1)
        self.domain_fusion = MLP_fusion(320, 320, [348], 0.1)
        self.MLP_fusion0 = MLP_fusion(768 * 2, 768, [348], 0.1)
        self.clip_fusion = clip_fuion(1024, 320, [348], 0.1)


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

        self.ClipModel,_ = load_from_name("ViT-B-16", device="cuda", download_root='./')


        #pivot:
        feature_emb_size = 320
        img_emb_size =320
        feature_num = 4
        self.feature_num = 4
        text_emb_size = 320
        #self.n_node = 64
        self.feature_emb_size = 320
        self.emb_size = 320
        self.layers = 12
        self.transformers = torch.nn.ModuleList([TransformerLayer(feature_emb_size, head_num=4, dropout=0.6,
                                                                  attention_dropout=0,
                                                                  initializer_range=0.02) for _ in
                                                 range(self.layers)])
        self.mlp_img = torch.nn.ModuleList([MLP_trans(img_emb_size, feature_emb_size, dropout=0.6) for _ in
                                                 range(feature_num)])

        self.mlp_text = torch.nn.ModuleList([MLP_trans(text_emb_size, feature_emb_size, dropout=0.6) for _ in
                                            range(feature_num)])
        self.pivot_mlp_fusion = torch.nn.ModuleList([MLP_trans(text_emb_size, feature_emb_size, dropout=0.6) for _ in
                                             range(feature_num)])
        self.transformers_list = torch.nn.ModuleList()
        self.mlp_img_list = torch.nn.ModuleList()
        self.mlp_text_list = torch.nn.ModuleList()
        self.pivot_mlp_fusion_list = torch.nn.ModuleList()
        for i in range(self.domain_num):
            self.transformers_list.append(torch.nn.ModuleList([TransformerLayer(feature_emb_size, head_num=4, dropout=0.6,
                                                                  attention_dropout=0,
                                                                  initializer_range=0.02) for _ in
                                                 range(self.layers)]))
            self.mlp_img_list.append(torch.nn.ModuleList([MLP_trans(img_emb_size, feature_emb_size, dropout=0.6) for _ in
                                                 range(feature_num)]))
            self.mlp_text_list.append(torch.nn.ModuleList([MLP_trans(text_emb_size, feature_emb_size, dropout=0.6) for _ in
                                            range(feature_num)]))
            self.pivot_mlp_fusion_list.append(torch.nn.ModuleList([MLP_trans(text_emb_size, feature_emb_size, dropout=0.6) for _ in
                                             range(feature_num)]))


        self.active = nn.SiLU()
        self.dropout2 = nn.Dropout(0.2)
        self.mlp_star_f1 = nn.Linear(self.feature_emb_size * 4, self.emb_size)
        self.mlp_star_f2 = nn.Linear(self.emb_size, self.emb_size)
        self.mlp_star_f1_list = torch.nn.ModuleList()
        self.mlp_star_f2_list = torch.nn.ModuleList()
        for i in range(self.domain_num):
            self.mlp_star_f1_list.append(nn.Linear(self.feature_emb_size * 4, self.emb_size))
            self.mlp_star_f2_list.append(nn.Linear(self.emb_size, self.emb_size))

    def forward(self, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        text_feature = self.bert(inputs, attention_mask=masks)[0]  # ([64, 197, 768])
        image = kwargs['image']
        image_feature = self.image_model.forward_ying(image)  # ([64, 197, 768])
        #image_feature = self.bert(inputs, attention_mask=masks)[0]
        clip_image = kwargs['clip_image']
        clip_text = kwargs['clip_text']
        with torch.no_grad():
            clip_image_feature = self.ClipModel.encode_image(clip_image)# ([64, 512])
            clip_text_feature = self.ClipModel.encode_text(clip_text)  # ([64, 512])
            clip_image_feature /= clip_image_feature.norm(dim=-1, keepdim=True)
            clip_text_feature /= clip_text_feature.norm(dim=-1, keepdim=True)
            #print(clip_image_feature.size())
            #print(clip_text_feature.size())
        clip_fusion_feature = torch.cat((clip_image_feature, clip_text_feature),dim=-1)#torch.Size([64, 1024])
        clip_fusion_feature = self.clip_fusion(clip_fusion_feature.float())#torch.Size([64, 320])

        #text_atn_feature, _ = self.text_attention(text_feature)  # ([64, 768])
        text_atn_feature = self.text_attention(text_feature,masks)
        image_atn_feature, _ = self.image_attention(image_feature)
        fusion_feature = torch.cat((image_feature, text_feature), dim=-1)
        fusion_atn_feature, _ = self.fusion_attention(fusion_feature)  # ([64, 1536])
        fusion_atn_feature = self.MLP_fusion0(fusion_atn_feature)
        # print("image_atn_feature", image_atn_feature.size())

        text_gate_input = text_atn_feature  # ([64, 1536])
        image_gate_input = image_atn_feature
        fusion_gate_input = fusion_atn_feature

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

        #clip_fusion_feature
        #fusion

        text = text_gate_share_expert_value[0]
        image = image_gate_share_expert_value[0]
        fusion_share_feature = torch.cat((clip_fusion_feature,text, image), dim=-1)

        fusion_share_feature = self.MLP_fusion(fusion_share_feature)
        #fusion_share_feature = self.MLP_fusion(fusion_share_feature)
        #fusion_share_feature = clip_fusion_feature
        fusion_gate_input0 = self.domain_fusion(fusion_share_feature)
        fusion_gate_out_list0 = []
        for k in range(self.domain_num):
            gate_out = self.fusion_gate_list0[k](fusion_gate_input0)
            fusion_gate_out_list0.append(gate_out)
        self.fusion_gate_out_list0 = fusion_gate_out_list0


        fusion_gate_expert_value0 = []
        fusion_gate_spacial_expert_value0 = []
        fusion_gate_share_expert_value0 = []
        for m in range(self.domain_num):
            share_gate_expert0 = 0
            gate_spacial_expert = 0
            gate_share_expert = 0
            for n in range(self.num_expert):
                fusion_tmp_expert0 = self.fusion_experts[m][n](fusion_share_feature)
                share_gate_expert0 += (fusion_tmp_expert0 * self.fusion_gate_out_list0[m][:, n].unsqueeze(1))
                gate_spacial_expert += (fusion_tmp_expert0 * self.fusion_gate_out_list0[m][:, n].unsqueeze(1))
            for n in range(self.num_expert * 2):
                fusion_tmp_expert0 = self.fusion_share_expert[0][n](fusion_share_feature)
                share_gate_expert0 += (fusion_tmp_expert0 * self.fusion_gate_out_list0[m][:, (self.num_expert + n)].unsqueeze(1))
                gate_share_expert += (fusion_tmp_expert0 * self.fusion_gate_out_list0[m][:, (self.num_expert + n)].unsqueeze(1))
            fusion_gate_expert_value0.append(share_gate_expert0)
            fusion_gate_spacial_expert_value0.append(gate_spacial_expert)
            fusion_gate_share_expert_value0.append(gate_share_expert)

##continue

        #text
        text_two_task = []
        image_two_task = []
        fusion_two_task = []
        text_two_task.append(self.text_classifier(text_gate_expert_value[0]).squeeze(1))
        text_two_task.append(self.text_classifier_Mu(text_gate_expert_value[1]).squeeze(1))
        image_two_task.append(self.image_classifier(image_gate_expert_value[0]).squeeze(1))
        image_two_task.append(self.image_classifier_Mu(image_gate_expert_value[1]).squeeze(1))
        fusion_two_task.append(self.fusion_classifier(fusion_gate_expert_value0[0]).squeeze(1))
        fusion_two_task.append(self.fusion_classifier_Mu(fusion_gate_expert_value0[1]).squeeze(1))


        text_fake_news = torch.sigmoid(text_two_task[0])
        image_fake_news = torch.sigmoid(image_two_task[0])
        fusion_fake_news = torch.sigmoid(fusion_two_task[0])

        #print(text_gate_expert_value[0].size())
        text_multi_domain = torch.softmax(text_two_task[1],-1)
        image_multi_domain = torch.softmax(image_two_task[1],-1)
        fusion_multi_domain = torch.softmax(fusion_two_task[1],-1)

        multi_label_feature = text_gate_expert_value[0]+image_gate_expert_value[0]+fusion_gate_expert_value0[0]
        fake_news_feature = text_gate_expert_value[1]+image_gate_expert_value[1]+fusion_gate_expert_value0[1]

        a = torch.cat([multi_label_feature,fake_news_feature],dim = -1)
        label = torch.sigmoid(self.max_classifier(a).squeeze(1))

        return label,text_fake_news,text_multi_domain,image_fake_news,image_multi_domain,fusion_fake_news,fusion_multi_domain

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
                batch_data = clipdata2gpu(batch)
                label = batch_data['label']
                category = batch_data['category']
                idxs = torch.tensor([index for index in category]).view(-1, 1)
                labels_domain = torch.nn.functional.one_hot(idxs.squeeze(), 9).cuda()

                label0,text_fake_news,text_multi_domain,image_fake_news,image_multi_domain,fusion_fake_news,fusion_multi_domain = self.model(**batch_data)
                loss0 = loss_fn(label0,label.float())
                #print(text_multi_domain.size())
                loss11 = torch.nn.functional.binary_cross_entropy_with_logits(text_multi_domain, labels_domain.float())
                loss21 = torch.nn.functional.binary_cross_entropy_with_logits(image_multi_domain, labels_domain.float())
                loss31 = torch.nn.functional.binary_cross_entropy_with_logits(fusion_multi_domain,
                                                                              labels_domain.float())
                loss12 = loss_fn(text_fake_news,label.float())
                loss22 = loss_fn(image_fake_news, label.float())
                loss32 = loss_fn(fusion_fake_news, label.float())
                loss = loss0+(loss11+loss12+loss21+loss22+loss31+loss32)/6

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (scheduler is not None):
                    scheduler.step()
                avg_loss.add(loss.item())
            print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))
            #results0,results1,results2,results3 = self.test(self.val_loader)
            results0 = self.test(self.test_loader)
            mark = recorder.add(results0)
            if mark == 'save':
                torch.save(self.model.state_dict(),
                           os.path.join(self.save_param_dir, 'parameter_clip111.pkl'))
            elif mark == 'esc':
                break
            else:
                continue
        self.model.load_state_dict(torch.load(os.path.join(self.save_param_dir, 'parameter_clip111.pkl')))
        results0 = self.test(self.test_loader)

        return results0, os.path.join(self.save_param_dir, 'parameter_clip111.pkl')

    def test(self, dataloader):
        pred = []
        label = []
        category = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = clipdata2gpu(batch)
                batch_label = batch_data['label']
                batch_category = batch_data['category']
                batch_label_pred,_,_,_,_,_,_ = self.model(**batch_data)

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_label_pred.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())

        return metricsTrueFalse(label, pred, category, self.category_dict)