# Description : Baseline model with just the initial classification layers.. 

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
    def __init__(self, d_model, dropout=0.1, max_len=5000):
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

    def forward(self, features):
        activations = {}
        features = self.positional_encoding(features)
        features = [self.feature_expansion[i](features) for i in range(self.num_classes)]
        expanded_features = torch.stack(features, dim=0).permute(1, 2, 0, 3)
        initial_outputs = [self.initial_classifiers[i](expanded_features[:,:,i,:]) for i in range(self.num_classes)]
        initial_outputs = torch.stack(initial_outputs, dim=2).squeeze(-1)
        activations['init_output'] = initial_outputs
        activations['final_output'] = initial_outputs
        return activations

