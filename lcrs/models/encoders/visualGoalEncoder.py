#!/usr/bin/env python3
import torch
import torch.nn as nn


class VisualGoalEncoder(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        hidden_size: int,
        depth: int,
    ):
        super(VisualGoalEncoder, self).__init__()

        layers = [nn.Linear(in_features=in_size, out_features=hidden_size),
                  nn.ReLU(),]
        for _ in range(depth):
            layers.append(nn.Linear(in_features=hidden_size, out_features=hidden_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(in_features=hidden_size, out_features=out_size))
        self.fc = nn.Sequential(*layers)

    def forward(self, goalFeatures: torch.Tensor) -> torch.Tensor:
        return self.fc(goalFeatures)
