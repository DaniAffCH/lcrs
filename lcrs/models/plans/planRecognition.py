#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from utils.distribution import Distribution


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)

        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class PlanRecognition(nn.Module):
    def __init__(
        self,
        visual_features: int,
        hidden_size: int,
        plan_features: int,
        depth: int,
        transformer_heads: int,
        dropout: float,
        dist: Distribution
    ):
        super(PlanRecognition, self).__init__()
        in_size = visual_features
        self.dist = dist

        self.pos_encoder = PositionalEncoding(in_size, dropout)
        encoder_layers = TransformerEncoderLayer(in_size, transformer_heads, hidden_size, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, depth)

        layers = [nn.Linear(in_features=in_size, out_features=hidden_size),
                  nn.ReLU(),]
        for _ in range(depth):
            layers.append(nn.Linear(in_features=hidden_size, out_features=hidden_size))
            layers.append(nn.ReLU())

        self.fc = nn.Sequential(*layers)
        self.stateProj = nn.Linear(in_features=hidden_size, out_features=plan_features)

    def forward(self, visual_video: torch.Tensor) -> torch.Tensor:
        pos_encoded = self.pos_encoder(visual_video.permute(1, 0, 2))
        x = self.transformer_encoder(pos_encoded)
        x = self.fc(x.permute(1, 0, 2))
        x = torch.mean(x, dim=1)  # TODO: there could be a better way to aggregate data
        state = self.stateProj(x)
        return self.dist.get_state(state), x
