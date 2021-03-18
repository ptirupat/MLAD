# Description : In this model two different branches are used, one for time and another for class. Features are combined after every layer. 

import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.nn.init import kaiming_uniform_
import torch
import torchvision.models.video as video_models
from torch import nn
from torch.nn import functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.2, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, dropout = 0.2, activation='relu'):
        super().__init__()
        self.d_model = d_model
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def attention(self, q, k, v, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(self.d_model)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        output = torch.matmul(scores, v)
        return output, scores

    def forward(self, q, k, v):
        bs, n, c, f = q.size(0), q.size(1), q.size(2), q.size(3)
        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, n, c, self.d_model)
        q = self.q_linear(q).view(bs, n, c, self.d_model)
        v = self.v_linear(v).view(bs, n, c, self.d_model)
        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        # calculate attention using function we will define next
        scores, attention = self.attention(q, k, v, None, self.dropout)
        output = self.activation(self.out(scores)).transpose(1,2)
        return output, (attention)


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, dim_feedforward=64, dropout=0.2, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.self_attn = MultiHeadAttention(hidden_dim, dropout=dropout, activation=activation)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src):
        src2, weights = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, weights


class VideoTransformer(nn.Module):
    def __init__(self, num_clips, num_classes, feature_dim, hidden_dim, num_layers):
        super(VideoTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_clips = num_clips
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.positional_encoding = PositionalEncoding(feature_dim, max_len=num_clips)
        self.feature_expansion = nn.ModuleList([nn.Sequential(nn.Linear(feature_dim, self.hidden_dim), nn.ReLU()) for i in range(self.num_classes)])
        self.initial_classifiers = nn.ModuleList([nn.Linear(self.hidden_dim, 1) for i in range(self.num_classes)])
        self.encoder_layers1 = nn.ModuleList([TransformerEncoderLayer(hidden_dim=self.hidden_dim, dim_feedforward=self.hidden_dim) for i in range(self.num_layers)])
        self.encoder_layers2 = nn.ModuleList([TransformerEncoderLayer(hidden_dim=self.hidden_dim, dim_feedforward=self.hidden_dim) for i in range(self.num_layers)])
        self.final_classifiers = nn.ModuleList([nn.Linear(self.hidden_dim, 1) for i in range(self.num_classes)])
        self.alpha = nn.Parameter(torch.tensor([0.0]*self.num_layers)) #torch.tensor([0.0]*self.num_layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        activations = {}
        features = self.positional_encoding(features)
        features = [self.feature_expansion[i](features) for i in range(self.num_classes)]
        expanded_features = torch.stack(features, dim=0).permute(1, 2, 0, 3)
        initial_outputs = [self.initial_classifiers[i](expanded_features[:,:,i,:]) for i in range(self.num_classes)]
        initial_outputs = torch.stack(initial_outputs, dim=2).squeeze(-1)
        activations['init_output'] = initial_outputs
        features = expanded_features
        for i in range(self.num_layers):
            encoder_output1 = self.encoder_layers1[i](features)[0]
            encoder_output2 = self.encoder_layers2[i](features.transpose(1,2))[0].transpose(1,2)
            features = (self.sigmoid(self.alpha[i]) * encoder_output1) + ((1 - self.sigmoid(self.alpha[i])) *  encoder_output2)
        final_outputs = [self.final_classifiers[i](features[:,:,i,:]) for i in range(self.num_classes)]
        final_outputs = torch.stack(final_outputs, dim=2).squeeze(-1)
        activations['final_output'] = final_outputs
        return activations

