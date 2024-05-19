import  torch
from torchvision.models import resnet18
import torch.nn.functional as F
import numpy as np
import math
import torch.nn as nn
from torch.autograd import Function
class EmbeddingLayer(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)

class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
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

class MLP(torch.nn.Module):
    def __init__(self,input_dim,embed_dims,dropout):
        super(MLP, self).__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim,embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        layers.append(torch.nn.Linear(input_dim,1))
        self.mlp = torch.nn.Sequential(*layers)
    def forward(self,x):
        return self.mlp(x)

class MLP_Mu(torch.nn.Module):
    def __init__(self,input_dim,embed_dims,dropout):
        super(MLP_Mu, self).__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim,embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        layers.append(torch.nn.Linear(input_dim,9))
        self.mlp = torch.nn.Sequential(*layers)
    def forward(self,x):
        return self.mlp(x)

class MLP_fusion(torch.nn.Module):
    def __init__(self,input_dim,out_dim,embed_dims,dropout):
        super(MLP_fusion, self).__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim,embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        layers.append(torch.nn.Linear(input_dim,out_dim))
        self.mlp = torch.nn.Sequential(*layers)
    def forward(self,x):
        return self.mlp(x)
"""
class MLP_fusion_gate(torch.nn.Module):
    def __init__(self,input_dim,embed_dims,dropout):
        super(MLP_fusion_gate, self).__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim,embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        layers.append(torch.nn.Linear(input_dim,768))
        self.mlp = torch.nn.Sequential(*layers)
    def forward(self,x):
        return self.mlp(x)
"""
class clip_fuion(torch.nn.Module):
    def __init__(self,input_dim,out_dim,embed_dims,dropout):
        super(clip_fuion, self).__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim,embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        layers.append(torch.nn.Linear(input_dim,out_dim))
        self.mlp = torch.nn.Sequential(*layers)
    def forward(self,x):
        return self.mlp(x)



class cnn_extractor(torch.nn.Module):
    def __init__(self,input_size,feature_kernel):
        super(cnn_extractor, self).__init__()
        self.convs =torch.nn.ModuleList(
            [torch.nn.Conv1d(input_size,feature_num,kernel)
            for kernel,feature_num in feature_kernel.items()]
        )
    def forward(self,input_data):
        input_data = input_data.permute(0, 2, 1)
        feature = [conv(input_data)for conv in self.convs]
        feature = [torch.max_pool1d(f,f.shape[-1])for f in feature]
        feature = torch.cat(feature,dim = 1)
        feature = feature.view([-1,feature.shape[1]])
        return feature

class image_cnn_extractor(nn.Module):
    def __init__(self):
        super(image_cnn_extractor, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(197, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 24 * 96, 512)
        self.fc2 = nn.Linear(512, 320)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Convolutional layers with ReLU activation and pooling
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)

        # Flatten the output for fully connected layers
        x = x.view(-1, 256 * 24 * 96)

        # Fully connected layers with ReLU activation
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class image_extractor(torch.nn.Module):
    def __init__(self,out_channels):
        super(image_extractor, self).__init__()
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
    def forward(self,img):
        n_batch = img.size(0)
        img_out = self.img_model(img)
        img_out = self.avg_pool(img_out)#([64, 512, 1, 1])
        img_out = img_out.view(n_batch, -1)#([64, 512])
        img_out = self.img_fc(img_out)#([64, 320])
        img_out = F.normalize(img_out, p=2, dim=-1)
        return img_out

class classifier(torch.nn.Module):
    def __init__(self,out_dim=1):
        super(classifier, self).__init__()
        self.trim = nn.Sequential(
            nn.Linear(self.unified_dim, 64),
            nn.SiLU(),
            # SimpleGate(),
            # nn.BatchNorm1d(64),
            # nn.Dropout(0.2),
        )
        self.classifier1 = nn.Sequential(
            nn.Linear(64, out_dim),
        )
    def forward(self,x):
        x = self.classifier1(self.trim(x))
        return x

class MaskAttention(torch.nn.Module):
    def __init__(self,input_dim):
        super(MaskAttention, self).__init__()
        self.Line = torch.nn.Linear(input_dim,1)

    def forward(self,input,mask):
        score = self.Line(input).view(-1,input.size(1))
        if mask is not None:
            score = score.masked_fill(mask == 0, float("-inf"))
        score = torch.softmax(score, dim=-1).unsqueeze(1)
        output = torch.matmul(score,input).squeeze(1)
        return output

class TokenAttention(torch.nn.Module):
    """
    Compute attention layer
    """

    def __init__(self, input_shape):
        super(TokenAttention, self).__init__()
        self.attention_layer = nn.Sequential(
                            torch.nn.Linear(input_shape, input_shape),
                            nn.SiLU(),
                            #SimpleGate(dim=2),
                            torch.nn.Linear(input_shape, 1),
        )

    def forward(self, inputs):
        scores = self.attention_layer(inputs).view(-1, inputs.size(1))
        #scores = torch.softmax(scores, dim=-1).unsqueeze(1)
        scores = scores.unsqueeze(1)
        outputs = torch.matmul(scores, inputs).squeeze(1)
        # scores = self.attention_layer(inputs)
        # outputs = scores*inputs
        return outputs, scores
class Attention(torch.nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn
class MultiHeadedAttention(torch.nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = torch.nn.ModuleList([torch.nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = torch.nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        if mask is not None:
            mask = mask.repeat(1, self.h, 1, 1)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x), attn
class Resnet(torch.nn.Module):
    def __init__(self,out_channels):
        super(Resnet, self).__init__()
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

    def forward(self, img):
        n_batch = img.size(0)
        img_out = self.img_model(img)
        img_out = self.avg_pool(img_out)
        img_out = img_out.view(n_batch, -1)
        img_out = self.img_fc(img_out)
        img_out = F.normalize(img_out, p=2, dim=-1)
        return img_out

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, input_, alpha):
        ctx.alpha = alpha
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None