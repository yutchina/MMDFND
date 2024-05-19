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


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


class HyperConv(Module):
    def __init__(self, layers, dataset, emb_size, n_node, n_price, img_emb_size, text_emb_size):
        super(HyperConv, self).__init__()
        self.emb_size = emb_size
        self.layers = layers
        self.dataset = dataset
        self.n_node = n_node
        self.n_price = n_price
        self.img_emb_size = img_emb_size
        self.text_emb_size = text_emb_size

        # self.img_mlp = nn.Linear(self.img_emb_size, self.emb_size)
        # self.text_mlp = nn.Linear(self.text_emb_size, self.emb_size)
        # self.pri_mlp = nn.Linear(self.emb_size, self.emb_size)
        # self.id_mlp = nn.Linear(self.emb_size, self.emb_size)
        self.dif2one_mlp = nn.Linear(3 * self.emb_size, self.emb_size)

        self.w_pv = nn.Linear(self.emb_size, self.emb_size)

        self.w_vp = nn.Linear(self.emb_size, self.emb_size)

        self.tran_pv = nn.Linear(self.emb_size, self.emb_size)
        self.tran_pc = nn.Linear(self.emb_size, self.emb_size)

        self.mat_pv = nn.Parameter(torch.Tensor(self.n_price, 1))

        self.a_i_g = nn.Linear(self.emb_size, self.emb_size)
        self.b_i_g = nn.Linear(self.emb_size, self.emb_size)

        self.a_o_g_i = nn.Linear(self.emb_size * 3, self.emb_size)
        self.b_o_gi1 = nn.Linear(self.emb_size, self.emb_size)
        self.b_o_gi2 = nn.Linear(self.emb_size, self.emb_size)
        self.b_o_gi3 = nn.Linear(self.emb_size, self.emb_size)

        self.a_o_g_p = nn.Linear(self.emb_size * 3, self.emb_size)
        self.b_o_gp1 = nn.Linear(self.emb_size, self.emb_size)
        self.b_o_gp2 = nn.Linear(self.emb_size, self.emb_size)
        self.b_o_gp3 = nn.Linear(self.emb_size, self.emb_size)

        self.a_o_g_c = nn.Linear(self.emb_size * 3, self.emb_size)
        self.b_o_gc1 = nn.Linear(self.emb_size, self.emb_size)
        self.b_o_gc2 = nn.Linear(self.emb_size, self.emb_size)
        self.b_o_gc3 = nn.Linear(self.emb_size, self.emb_size)

        self.a_o_g_b = nn.Linear(self.emb_size * 3, self.emb_size)
        self.b_o_gb1 = nn.Linear(self.emb_size, self.emb_size)
        self.b_o_gb2 = nn.Linear(self.emb_size, self.emb_size)
        self.b_o_gb3 = nn.Linear(self.emb_size, self.emb_size)

        self.dropout10 = nn.Dropout(0.1)
        self.dropout20 = nn.Dropout(0.2)
        self.dropout30 = nn.Dropout(0.3)
        self.dropout40 = nn.Dropout(0.4)
        self.dropout50 = nn.Dropout(0.5)
        self.dropout60 = nn.Dropout(0.6)
        self.dropout70 = nn.Dropout(0.7)

    def forward(self, adjacency, adjacency_pv, adjacency_vp, embedding, pri_emb, img_emb, text_emb):
        # updating embeddings with different types
        # convert image_emb and text_emb to the embeddings dimension as item_emb
        image_embeddings = self.img_mlp(img_emb)
        text_embeddings = self.text_mlp(text_emb)
        price_embeddings = self.pri_mlp(pri_emb)
        id_embeddings = self.id_mlp(embedding)


        return id_embeddings, image_embeddings, text_embeddings, price_embeddings

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, hidden_size, activate="relu", head_num=2, dropout=0, initializer_range=0.02):
        super(MultiHeadSelfAttention, self).__init__()
        self.config = list()

        self.hidden_size = hidden_size

        self.head_num = head_num
        if (self.hidden_size) % head_num != 0:
            raise ValueError(self.head_num, "error")
        self.head_dim = self.hidden_size // self.head_num

        self.query = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.key = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.value = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.concat_weight = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        torch.nn.init.normal_(self.query.weight, 0, initializer_range)
        torch.nn.init.normal_(self.key.weight, 0, initializer_range)
        torch.nn.init.normal_(self.value.weight, 0, initializer_range)
        torch.nn.init.normal_(self.concat_weight.weight, 0, initializer_range)
        self.dropout = torch.nn.Dropout(dropout)

    def dot_score(self, encoder_output):
        query = self.dropout(self.query(encoder_output))
        key = self.dropout(self.key(encoder_output))
        # head_num * batch_size * session_length * head_dim
        querys = torch.stack(query.chunk(self.head_num, -1), 0)
        keys = torch.stack(key.chunk(self.head_num, -1), 0)
        # head_num * batch_size * session_length * session_length
        dots = querys.matmul(keys.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))
        #         print(len(dots),dots[0].shape)
        return dots

    def forward(self, encoder_outputs, mask=None):
        attention_energies = self.dot_score(encoder_outputs)
        value = self.dropout(self.value(encoder_outputs))

        values = torch.stack(value.chunk(self.head_num, -1))

        if mask is not None:
            eye = torch.eye(mask.shape[-1]).to('cuda')
            new_mask = torch.clamp_max((1 - (1 - mask.float()).unsqueeze(1).permute(0, 2, 1).bmm(
                (1 - mask.float()).unsqueeze(1))) + eye, 1)
            attention_energies = attention_energies - new_mask * 1e12
            weights = F.softmax(attention_energies, dim=-1)
            weights = weights * (1 - new_mask)
        else:
            weights = F.softmax(attention_energies, dim=2)

        # head_num * batch_size * session_length * head_dim
        outputs = weights.matmul(values)
        # batch_size * session_length * hidden_size
        outputs = torch.cat([outputs[i] for i in range(outputs.shape[0])], dim=-1)
        outputs = self.dropout(self.concat_weight(outputs))

        return outputs


class PositionWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_size, initializer_range=0.02):
        super(PositionWiseFeedForward, self).__init__()
        self.final1 = torch.nn.Linear(hidden_size, hidden_size * 4, bias=True)
        self.final2 = torch.nn.Linear(hidden_size * 4, hidden_size, bias=True)
        torch.nn.init.normal_(self.final1.weight, 0, initializer_range)
        torch.nn.init.normal_(self.final2.weight, 0, initializer_range)

    def forward(self, x):
        x = F.relu(self.final1(x))
        x = self.final2(x)
        return x


class TransformerLayer(torch.nn.Module):
    def __init__(self, hidden_size, activate="relu", head_num=4, dropout=0, attention_dropout=0,
                 initializer_range=0.02):
        super(TransformerLayer, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.mh = MultiHeadSelfAttention(hidden_size=hidden_size, activate=activate, head_num=head_num,
                                         dropout=attention_dropout, initializer_range=initializer_range)
        self.pffn = PositionWiseFeedForward(hidden_size, initializer_range=initializer_range)
        self.layer_norm = torch.nn.LayerNorm(hidden_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, encoder_outputs, mask=None):
        encoder_outputs = self.layer_norm(encoder_outputs + self.dropout(self.mh(encoder_outputs, mask)))
        encoder_outputs = self.layer_norm(encoder_outputs + self.dropout(self.pffn(encoder_outputs)))
        return encoder_outputs

class MLP_trans(torch.nn.Module):
    def __init__(self, input_size, out_size, dropout=0.2):
        super(MLP_trans, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.activate = torch.nn.SiLU()
        self.mlp_1 = nn.Linear(input_size, out_size)
        self.mlp_2 = nn.Linear(out_size, out_size)
        # self.mlp_3 = nn.Linear(input_size, input_size)
        # self.mlp_4 = nn.Linear(input_size, out_size)

    def forward(self, emb_trans):
        emb_trans = self.dropout(self.activate(self.mlp_1(emb_trans)))
        emb_trans = self.dropout(self.activate(self.mlp_2(emb_trans)))
        # emb_trans = self.dropout(self.activate(self.mlp_3(emb_trans)))
        # emb_trans = self.dropout(self.activate(self.mlp_4(emb_trans)))
        return emb_trans

class MLP_merge_star(torch.nn.Module):
    def __init__(self, imput_size, out_size, dropout=0.2):
        super(MLP_merge_star, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.activate = torch.nn.Tanh()
        self.mlp_s = nn.Linear(imput_size, int(imput_size / 2))

    def forward(self, emb_trans):
        results = self.dropout(self.activate(self.mlp_s(emb_trans)))
        return results

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Beyond(Module):
    def __init__(self, price_list, category_list, n_node, n_price, n_category, lr, layers, feature_num, l2, lam, dataset, num_heads=4, emb_size=100, img_emb_size=100, text_emb_size=100, feature_emb_size=100, batch_size=100, num_negatives=100):
        super(Beyond, self).__init__()
        self.price_list = price_list
        self.category_list = category_list
        self.emb_size = emb_size
        self.img_emb_size = img_emb_size
        self.text_emb_size = text_emb_size
        self.feature_emb_size = feature_emb_size
        self.batch_size = batch_size
        self.n_node = n_node
        self.n_price = n_price
        self.n_category = n_category
        self.L2 = l2
        self.lr = lr
        self.layers = layers
        self.lam = lam
        self.num_negatives = num_negatives
        self.num_heads = num_heads
        self.feature_num = feature_num
        self.transformers = torch.nn.ModuleList([TransformerLayer(feature_emb_size, head_num=num_heads, dropout=0.6,
                                                                  attention_dropout=0,
                                                                  initializer_range=0.02) for _ in
                                                 range(layers)])
        self.mlp_img = torch.nn.ModuleList([MLP_trans(img_emb_size, feature_emb_size, dropout=0.6) for _ in
                                                 range(feature_num)])

        self.mlp_text = torch.nn.ModuleList([MLP_trans(text_emb_size, feature_emb_size, dropout=0.6) for _ in
                                            range(feature_num)])

        self.mlp_merge_star1 = torch.nn.ModuleList([MLP_merge_star(emb_size*2, emb_size, dropout=0.5) for _ in
                                             range(int(layers/4))])
        self.mlp_merge_star2 = torch.nn.ModuleList([MLP_merge_star(emb_size*2, emb_size, dropout=0.5) for _ in
                                                    range(int(layers/4))])
        self.mlp_merge_star3 = torch.nn.ModuleList([MLP_merge_star(emb_size*2, emb_size, dropout=0.5) for _ in
                                                    range(int(layers/4))])
        self.mlp_merge_star4 = torch.nn.ModuleList([MLP_merge_star(emb_size*2, emb_size, dropout=0.5) for _ in
                                                    range(int(layers/4))])

        self.mlp_star_f1 = nn.Linear(self.feature_emb_size*4, self.emb_size)
        self.mlp_star_f2 = nn.Linear(self.emb_size, self.emb_size )

        self.embedding = nn.Embedding(self.n_node, self.emb_size)
        self.cate_embedding = nn.Embedding(self.n_category, self.emb_size)

        self.LayerNorm = LayerNorm(self.emb_size, eps=1e-12)

        self.price_mean_embedding = nn.Embedding(self.n_price, self.emb_size)
        self.price_cov_embedding = nn.Embedding(self.n_price, self.emb_size)

        # init price embedding with different distribution
        # torch.nn.init.uniform_(self.price_mean_embedding.weight, a=0.0, b=1.0)
        # torch.nn.init.uniform_(self.price_cov_embedding.weight, a=0.0, b=1.0)

        self.star_emb1 = nn.Embedding(self.n_node, self.feature_emb_size)
        self.star_emb2 = nn.Embedding(self.n_node, self.feature_emb_size)
        self.star_emb3 = nn.Embedding(self.n_node, self.feature_emb_size)
        self.star_emb4 = nn.Embedding(self.n_node, self.feature_emb_size)
        torch.nn.init.normal_(self.star_emb1.weight, 0, 0.02)
        torch.nn.init.normal_(self.star_emb2.weight, 0, 0.02)
        torch.nn.init.normal_(self.star_emb3.weight, 0, 0.02)
        torch.nn.init.normal_(self.star_emb4.weight, 0, 0.02)

        self.mean_dense = nn.Linear(self.emb_size, self.emb_size)
        self.cov_dense = nn.Linear(self.emb_size, self.emb_size)

        # introducing text&image embeddings
        img_path = './datasets/'+ dataset + '/imgMatrixpca.npy'
        imgWeights = np.load(img_path)
        self.image_embedding = nn.Embedding(self.n_node, img_emb_size)
        img_pre_weight = np.array(imgWeights)
        self.image_embedding.weight.data.copy_(torch.from_numpy(img_pre_weight))

        text_path = './datasets/' + dataset + '/textMatrixpca.npy'
        textWeights = np.load(text_path)
        self.text_embedding = nn.Embedding(self.n_node, text_emb_size)
        text_pre_weight = np.array(textWeights)
        self.text_embedding.weight.data.copy_(torch.from_numpy(text_pre_weight))

        # introducing generated emb img&text
        imgText_path = './datasets/' + dataset + '/imgTextMatrixpca.npy'
        imgTextWeights = np.load(imgText_path)
        self.text_img_embedding = nn.Embedding(self.n_node, text_emb_size)
        img_text_pre_weight = np.array(imgTextWeights)
        self.text_img_embedding.weight.data.copy_(torch.from_numpy(img_text_pre_weight))

        textImg_path = './datasets/' + dataset + '/textImgMatrixpca.npy'
        textImgWeights = np.load(textImg_path)
        self.img_text_embedding = nn.Embedding(self.n_node, img_emb_size)
        text_img_pre_weight = np.array(textImgWeights)
        self.img_text_embedding.weight.data.copy_(torch.from_numpy(text_img_pre_weight))


        self.pos_embedding = nn.Embedding(2000, self.emb_size)
        self.pos_pri_embedding = nn.Embedding(2000, self.emb_size)

        self.img_mlp = nn.Linear(self.img_emb_size, self.emb_size)
        self.text_mlp = nn.Linear(self.text_emb_size, self.emb_size)
        self.pri_mlp = nn.Linear(self.emb_size, self.emb_size)
        self.id_mlp = nn.Linear(self.emb_size, self.emb_size)
        # self.img_text_cat = nn.Linear(self.emb_size*2, self.emb_size)

        # feature interaction


        # feature gate
        # self.star1_gate_w1 = nn.Linear(self.feature_emb_size, self.feature_emb_size)
        # self.star1_gate_w2 = nn.Linear(self.feature_emb_size, self.feature_emb_size)
        # self.star2_gate_w1 = nn.Linear(self.feature_emb_size, self.feature_emb_size)
        # self.star2_gate_w2 = nn.Linear(self.feature_emb_size, self.feature_emb_size)
        # self.star3_gate_w1 = nn.Linear(self.feature_emb_size, self.feature_emb_size)
        # self.star3_gate_w2 = nn.Linear(self.feature_emb_size, self.feature_emb_size)
        # self.star4_gate_w1 = nn.Linear(self.feature_emb_size, self.feature_emb_size)
        # self.star4_gate_w2 = nn.Linear(self.feature_emb_size, self.feature_emb_size)

        # self.pri_int_merge = nn.Linear(self.emb_size, self.emb_size, bias=False)

        self.active = nn.ReLU()
        self.w_1 = nn.Linear(self.emb_size * 2, self.emb_size)
        self.w_2 = nn.Linear(self.emb_size, 1)
        self.glu1 = nn.Linear(self.emb_size, self.emb_size)
        self.glu2 = nn.Linear(self.emb_size, self.emb_size, bias=False)

        # self.intre_mlp1 = nn.Linear(self.emb_size, self.emb_size, bias=True)
        # self.intre_mlp2 = nn.Linear(self.emb_size, self.emb_size, bias=True)

        # self.w_pm1 = nn.Linear(self.emb_size * 2, self.emb_size)
        # self.w_pm2 = nn.Linear(self.emb_size, 1)
        # self.glu_pm1 = nn.Linear(self.emb_size, self.emb_size)
        # self.glu_pm2 = nn.Linear(self.emb_size, self.emb_size, bias=False)
        # self.glu_pm3 = nn.Linear(self.emb_size, self.emb_size, bias=False)
        #
        # self.w_pc1 = nn.Linear(self.emb_size * 2, self.emb_size)
        # self.w_pc2 = nn.Linear(self.emb_size, 1)
        # self.glu_pc1 = nn.Linear(self.emb_size, self.emb_size)
        # self.glu_pc2 = nn.Linear(self.emb_size, self.emb_size, bias=False)
        # self.glu_pc3 = nn.Linear(self.emb_size, self.emb_size, bias=False)

        # self_attention
        if self.emb_size % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (emb_size, num_heads))

        # self.num_heads = num_heads  # 4
        self.attention_head_size = int(emb_size / num_heads)  # 16
        self.all_head_size = int(self.num_heads * self.attention_head_size)
        # query, key, value
        self.mean_query = nn.Linear(self.emb_size, self.emb_size)  # 128, 128
        self.mean_key = nn.Linear(self.emb_size, self.emb_size)
        self.mean_value = nn.Linear(self.emb_size, self.emb_size)
        self.cov_query = nn.Linear(self.emb_size, self.emb_size)  # 128, 128
        self.cov_key = nn.Linear(self.emb_size, self.emb_size)
        self.cov_value = nn.Linear(self.emb_size, self.emb_size)
        self.elu_activation = nn.ELU()

        self.dropout = nn.Dropout(0.2)
        self.emb_dropout = nn.Dropout(0.25)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout5 = nn.Dropout(0.5)
        self.dropout7 = nn.Dropout(0.7)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def generate_sess_emb(self, item_embedding, price_mean_emb, price_cov_emb, category_emb, session_item, sess_price, sess_category, session_len, reversed_sess_item, mask):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        # zeros = torch.zeros(1, self.emb_size)
        mask = mask.float().unsqueeze(-1)

        item_embedding = torch.cat([zeros, item_embedding], 0)
        price_mean_embedding = torch.cat([zeros, price_mean_emb], 0)
        price_cov_embedding = torch.cat([zeros, price_cov_emb], 0)
        category_embedding = torch.cat([zeros, category_emb], 0)

        # get = lambda i: item_embedding[reversed_sess_item[i]]
        # seq_h = torch.cuda.FloatTensor(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size).fill_(0)

        get = lambda i: item_embedding[session_item[i]]
        seq_h = torch.cuda.FloatTensor(self.batch_size, list(session_item.shape)[1], self.emb_size).fill_(0)
        for i in torch.arange(session_item.shape[0]):
            seq_h[i] = get(i)

        get_pri_mean = lambda i: price_mean_embedding[sess_price[i]]
        seq_pri_mean = torch.cuda.FloatTensor(self.batch_size, list(sess_price.shape)[1], self.emb_size).fill_(0)
        for i in torch.arange(sess_price.shape[0]):
            seq_pri_mean[i] = get_pri_mean(i)

        get_pri_cov = lambda i: price_cov_embedding[sess_price[i]]
        seq_pri_cov = torch.cuda.FloatTensor(self.batch_size, list(sess_price.shape)[1], self.emb_size).fill_(0)
        for i in torch.arange(sess_price.shape[0]):
            seq_pri_cov[i] = get_pri_cov(i)

        get_cate = lambda i: category_embedding[sess_category[i]]
        seq_cate = torch.cuda.FloatTensor(self.batch_size, list(sess_category.shape)[1], self.emb_size).fill_(0)
        for i in torch.arange(sess_category.shape[0]):
            seq_cate[i] = get_cate(i)

        # stamp to get session emb
        hs = torch.div(torch.sum(seq_h, 1), session_len.type(torch.cuda.FloatTensor))
        #
        len = seq_h.shape[1]
        # #  position embedding reverse
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(self.batch_size, 1, 1)
        # price_seq position emb
        pos_pri_emb = self.pos_pri_embedding.weight[:len]
        pos_pri_emb = pos_pri_emb.unsqueeze(0).repeat(self.batch_size, 1, 1)

        hs = hs.unsqueeze(-2).repeat(1, len, 1)

        nh = seq_h
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = self.w_2(nh)
        beta = beta * mask
        interest_pre = torch.sum(beta * seq_h, 1)

        # seq_pri_mean & seq_pri_cov & pos_pri_emb

        seq_pri_mean_emb = self.LayerNorm(seq_pri_mean)  + self.LayerNorm(seq_cate)
        seq_pri_mean_emb = self.dropout2(seq_pri_mean_emb)

        seq_pri_cov_emb = self.LayerNorm(seq_pri_cov)  + self.LayerNorm(seq_cate)
        seq_pri_cov_emb = self.dropout2(seq_pri_cov_emb)
        seq_pri_cov_emb = self.elu_activation(seq_pri_cov_emb) + 1

        attention_mask = mask.permute(0,2,1).unsqueeze(1)  # [bs, 1, 1, seqlen]
        attention_mask = (1.0 - attention_mask) * -10000.0

        mixed_mean_query_layer = self.mean_query(seq_pri_mean_emb)  # [bs, seqlen, hid_size]
        mixed_mean_key_layer = self.mean_key(seq_pri_mean_emb)  # [bs, seqlen, hid_size]
        mixed_mean_value_layer = self.mean_value(seq_pri_mean_emb)  # [bs, seqlen, hid_size]

        mixed_cov_query_layer = self.elu_activation(self.cov_query(seq_pri_cov_emb)) + 1  # [bs, seqlen, hid_size]
        mixed_cov_key_layer = self.elu_activation(self.cov_key(seq_pri_cov_emb)) + 1  # [bs, seqlen, hid_size]
        mixed_cov_value_layer = self.elu_activation(self.cov_value(seq_pri_cov_emb)) + 1

        attention_head_size = int(self.emb_size / self.num_heads)
        mean_query_layer = self.transpose_for_scores(mixed_mean_query_layer, attention_head_size)  # [bs, 8, seqlen, 16]
        mean_key_layer = self.transpose_for_scores(mixed_mean_key_layer, attention_head_size)
        mean_value_layer = self.transpose_for_scores(mixed_mean_value_layer, attention_head_size)  # [bs, 8, seqlen, 16]

        cov_query_layer = self.transpose_for_scores(mixed_cov_query_layer, attention_head_size)
        cov_key_layer = self.transpose_for_scores(mixed_cov_key_layer, attention_head_size)
        cov_value_layer = self.transpose_for_scores(mixed_cov_value_layer, attention_head_size)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # [bs, 8, seqlen, 16]*[bs, 8, 16, seqlen]  ==> [bs, 8, seqlen, seqlen]
        attention_scores = -wasserstein_distance_matmul(mean_query_layer, cov_query_layer, mean_key_layer,
                                                        cov_key_layer)

        attention_scores = attention_scores / math.sqrt(attention_head_size)  # [bs, 8, seqlen, seqlen]
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [bs, 8, seqlen, seqlen]
        attention_probs = self.dropout(attention_probs)


        mean_context_layer = torch.matmul(attention_probs, mean_value_layer)
        cov_context_layer = torch.matmul(attention_probs ** 2, cov_value_layer)
        mean_context_layer = mean_context_layer.permute(0, 2, 1, 3).contiguous()
        cov_context_layer = cov_context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = mean_context_layer.size()[:-2] + (self.emb_size,)

        mean_context_layer = mean_context_layer.view(*new_context_layer_shape)
        cov_context_layer = cov_context_layer.view(*new_context_layer_shape)
        # seq_pri_mean & seq_pri_cov
        mean_hidden_states = self.mean_dense(mean_context_layer)
        mean_hidden_states = self.dropout2(mean_hidden_states)
        mean_hidden_states = self.LayerNorm(mean_hidden_states + seq_pri_mean)

        cov_hidden_states = self.cov_dense(cov_context_layer)
        cov_hidden_states = self.dropout2(cov_hidden_states)
        cov_hidden_states = self.LayerNorm(cov_hidden_states + seq_pri_cov)

        item_pos = torch.tensor(range(1, seq_h.size()[1] + 1), device='cuda')
        item_pos = item_pos.unsqueeze(0).expand_as(session_item)

        item_pos = item_pos.type(torch.cuda.FloatTensor) * mask.squeeze(2)
        item_last_num = torch.max(item_pos, 1)[0].unsqueeze(1).expand_as(item_pos)
        last_pos_t = torch.where(item_pos - item_last_num >= 0, torch.tensor([1.0], device='cuda'),
                                 torch.tensor([0.0], device='cuda'))

        mean_hidden_states = last_pos_t.unsqueeze(2).expand_as(mean_hidden_states) * mean_hidden_states
        sess_price_mean = torch.sum(mean_hidden_states, 1)
        cov_hidden_states = last_pos_t.unsqueeze(2).expand_as(cov_hidden_states) * cov_hidden_states
        sess_price_cov = torch.sum(cov_hidden_states, 1)

        pri_list = trans_to_cuda(torch.Tensor(self.price_list.tolist()).long())
        cate_list = trans_to_cuda(torch.Tensor(self.category_list.tolist()).long())
        price_item_mean = price_mean_embedding[pri_list] + category_embedding[cate_list]
        price_item_cov = price_cov_embedding[pri_list] + category_embedding[cate_list]

        return interest_pre, sess_price_mean, sess_price_cov, price_item_mean, price_item_cov

    def fusion_img_text(self, image_emb, text_emb, star_emb1, star_emb2, star_emb3, star_emb4):
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

        for sa_i in range(0, int(self.layers), 2):
            trans_text_item = torch.cat(
                [star_emb1.unsqueeze(1), star_emb2.unsqueeze(1), star_emb3.unsqueeze(1), star_emb4.unsqueeze(1),  text_feature_seq], 1)
            text_output = self.transformers[sa_i + 1](trans_text_item)

            star_emb1 = (text_output[:, 0, :] + star_emb1)/2
            star_emb2 = (text_output[:, 1, :] + star_emb2)/2
            star_emb3 = (text_output[:, 2, :] + star_emb3)/2
            star_emb4 = (text_output[:, 3, :] + star_emb4)/2
            text_feature_seq = text_output[:, 4:self.feature_num+4, :] + text_feature_seq

            trans_img_item = torch.cat(
                [star_emb1.unsqueeze(1), star_emb2.unsqueeze(1), star_emb3.unsqueeze(1),star_emb4.unsqueeze(1),
                 img_feature_seq], 1)
            img_output = self.transformers[sa_i](trans_img_item)
            star_emb1 = (img_output[:, 0, :] + star_emb1) / 2
            star_emb2 = (img_output[:, 1, :] + star_emb2) / 2
            star_emb3 = (img_output[:, 2, :] + star_emb3) / 2
            star_emb4 = (img_output[:, 3, :] + star_emb4) / 2
            img_feature_seq = img_output[:, 4:self.feature_num + 4, :] + img_feature_seq

        item_emb_trans = self.dropout2(torch.cat([star_emb1, star_emb2, star_emb3,star_emb4], 1))
        item_emb_trans = self.dropout2(self.active(self.mlp_star_f1(item_emb_trans)))
        item_emb_trans = self.dropout2(self.active(self.mlp_star_f2(item_emb_trans)))
        return item_emb_trans

    def contrastive(self, img_emb, text_emb, img_gen_emb, text_gen_emb):
        tau = 1
        num_neg = self.num_negatives
        img_textimg_mat = torch.matmul(img_emb, img_gen_emb.permute(1, 0))
        img_textimg_mat = img_textimg_mat / tau
        img_textimg_mat = torch.exp(img_textimg_mat, out=None)
        topk_img_values, topk_indice = torch.topk(img_textimg_mat, k=num_neg, dim=1)

        img_loss_one = torch.sum(torch.log10(torch.diag(img_textimg_mat)))
        img_loss_two = torch.sum(torch.log10(torch.sum(topk_img_values, 1)))
        img_con_loss = img_loss_two -img_loss_one


        text_imgtext_mat = torch.matmul(text_emb, text_gen_emb.permute(1, 0))
        text_imgtext_mat = text_imgtext_mat / tau
        text_imgtext_mat = torch.exp(text_imgtext_mat, out=None)
        topk_text_values, topk_indice = torch.topk(text_imgtext_mat, k=num_neg, dim=1)

        text_loss_one = torch.sum(torch.log10(torch.diag(text_imgtext_mat)))
        text_loss_two = torch.sum(torch.log10(torch.sum(topk_text_values, 1)))
        text_con_loss = text_loss_two - text_loss_one

        con_loss = text_con_loss + img_con_loss
        return con_loss
    def transpose_for_scores(self, x, attention_head_size):
        # INPUT:  x'shape = [bs, seqlen, hid_size]  假设hid_size=128
        new_x_shape = x.size()[:-1] + (self.num_heads, attention_head_size)  # [bs, seqlen, 8, 16]
        x = x.view(*new_x_shape)  #
        return x.permute(0, 2, 1, 3)

    def forward(self, session_item, sess_price, sess_category, session_len, reversed_sess_item, mask):
        image_emb = self.image_embedding.weight
        text_emb = self.text_embedding.weight
        price_mean_emb = self.price_mean_embedding.weight
        price_cov_emb = self.price_cov_embedding.weight
        category_emb = self.cate_embedding.weight
        item_emb = self.embedding.weight
        star_emb1 = self.star_emb1.weight
        star_emb2 = self.star_emb2.weight
        star_emb3 = self.star_emb3.weight
        star_emb4 = self.star_emb4.weight

        img_gen_emb = self.img_text_embedding.weight
        text_gen_emb = self.text_img_embedding.weight
        # self-contrastive -> refining image&text embedding, two loss
        con_loss = self.contrastive(image_emb, text_emb, img_gen_emb, text_gen_emb)
        # fusion image&text

        item_emb_final = self.fusion_img_text(image_emb, text_emb, star_emb1, star_emb2, star_emb3, star_emb4)
        # obtain session emb
        sess_emb_hgnn, sess_price_mean, sess_price_cov, price_item_mean, price_item_cov = self.generate_sess_emb(item_emb_final, price_mean_emb, price_cov_emb, category_emb, session_item, sess_price, sess_category, session_len, reversed_sess_item, mask) #batch session embeddings

        return item_emb_final, price_item_mean, price_item_cov, sess_emb_hgnn, sess_price_mean, sess_price_cov, self.lam*con_loss


def perform(model, i, data):
    tar, session_len, session_item, reversed_sess_item, mask, price_seqs, category_seqs = data.get_slice(i) # 得到一个batch里的数据
    session_item = trans_to_cuda(torch.Tensor(session_item).long())
    session_len = trans_to_cuda(torch.Tensor(session_len).long())
    sess_price = trans_to_cuda(torch.Tensor(price_seqs).long())
    sess_category = trans_to_cuda(torch.Tensor(category_seqs).long())
    tar = trans_to_cuda(torch.Tensor(tar).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    reversed_sess_item = trans_to_cuda(torch.Tensor(reversed_sess_item).long())
    item_emb_final, price_mean_emb, price_cov_emb, sess_emb_hgnn, sess_price_mean, sess_price_cov, con_loss = model(session_item, sess_price, sess_category, session_len, reversed_sess_item, mask)
    scores_interest = torch.mm(sess_emb_hgnn, torch.transpose(item_emb_final, 1, 0))
    # considering the influence of price
    elu_activation = torch.nn.ELU()
    price_cov_emb = elu_activation(price_cov_emb) + 1
    sess_price_cov = elu_activation(sess_price_cov) + 1
    scores_price = wasserstein_distance_matmul(sess_price_mean, sess_price_cov, price_mean_emb, price_cov_emb)
    scores = scores_interest + scores_interest * scores_price
    scores = trans_to_cuda(scores)
    return tar, scores, con_loss

def wasserstein_distance_matmul(mean1, cov1, mean2, cov2):
    mean1_2 = torch.sum(mean1**2, -1, keepdim=True)
    mean2_2 = torch.sum(mean2**2, -1, keepdim=True)
    ret = -2 * torch.matmul(mean1, mean2.transpose(-1, -2)) + mean1_2 + mean2_2.transpose(-1, -2)
    cov1_2 = torch.sum(cov1, -1, keepdim=True)
    cov2_2 = torch.sum(cov2, -1, keepdim=True)
    cov_ret = -2 * torch.matmul(torch.sqrt(torch.clamp(cov1, min=1e-24)), torch.sqrt(torch.clamp(cov2, min=1e-24)).transpose(-1, -2)) + cov1_2 + cov2_2.transpose(-1, -2)

    return ret + cov_ret

def kl_distance_matmul(mean1, cov1, mean2, cov2):
    cov1_det = 1 / torch.prod(cov1, -1, keepdim=True)
    cov2_det = torch.prod(cov2, -1, keepdim=True)
    log_det = torch.log(torch.matmul(cov1_det, cov2_det.transpose(-1, -2)))

    trace_sum = torch.matmul(1 / cov2, cov1.transpose(-1, -2))
    mean_cov_part = torch.matmul((mean1 - mean2) ** 2, (1/cov2).transpose(-1, -2))

    return (log_det + mean_cov_part + trace_sum - mean1.shape[-1]) / 2

def train_test(model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    torch.autograd.set_detect_anomaly(True)
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size) #将session随机打乱，每x个一组（#session/batch_size)
    for i in slices:
        model.zero_grad()
        targets, scores, con_loss = perform(model, i, train_data)
        loss = model.loss_function(scores + 1e-8, targets)
        loss = loss + con_loss
        loss.backward()
        #        print(loss.item())
        model.optimizer.step()
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)
    top_K = [1, 5, 10, 20]
    metrics = {}
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['mrr%d' % K] = []
        metrics['ndcg%d' % K] = []
    print('start predicting: ', datetime.datetime.now())

    model.eval()
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        tar, scores, con_loss = perform(model, i, test_data)
        scores = trans_to_cpu(scores).detach().numpy()
        index = np.argsort(-scores, 1)
        tar = trans_to_cpu(tar).detach().numpy()
        for K in top_K:
            for prediction, target in zip(index[:, :K], tar):
                metrics['hit%d' % K].append(np.isin(target, prediction))
                if len(np.where(prediction == target)[0]) == 0:
                    metrics['mrr%d' % K].append(0)
                    metrics['ndcg%d' % K].append(0)
                else:
                    metrics['mrr%d' % K].append(1 / (np.where(prediction == target)[0][0] + 1))
                    metrics['ndcg%d' % K].append(1 / (np.log2(np.where(prediction == target)[0][0] + 2)))
    return metrics, total_loss