#!/usr/bin/env python3
import torch
import torch.nn as nn
from utils.distribution import Distribution


class PlanProposal(nn.Module):
    def __init__(
        self,
        visual_features: int,
        language_features: int,
        plan_features: int,
        hidden_size: int,
        depth: int,
        dist: Distribution
    ):
        super(PlanProposal, self).__init__()
        self.dist = dist
        in_size = visual_features + language_features

        layers = [nn.Linear(in_features=in_size, out_features=hidden_size),
                  nn.ReLU(),]
        for _ in range(depth):
            layers.append(nn.Linear(in_features=hidden_size, out_features=hidden_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(in_features=hidden_size, out_features=plan_features))

        self.fc = nn.Sequential(*layers)

    def forward(self, visual: torch.Tensor, language: torch.Tensor) -> torch.Tensor:
        x = torch.cat([visual, language], dim=-1)
        dist_x = self.fc(x)
        return self.dist.get_state(dist_x)
