#!/usr/bin/env python3

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from omegaconf import DictConfig
import hydra


class ImageEncoder(nn.Module):
    def __init__(
        self,
        num_c: int,
        visual_features: int,
        train_decoder: DictConfig
    ):
        super(ImageEncoder, self).__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_decoder = hydra.utils.instantiate(train_decoder)

        # TODO: make it adaptive wrt w and h
        self.conv_model = nn.Sequential(
            # input shape: [N, 3, 200, 200]
            nn.Conv2d(in_channels=num_c, out_channels=32, kernel_size=8, stride=4),  # shape: [N, 32, 49, 49]
            # nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),  # shape: [N, 64, 23, 23]
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),  # shape: [N, 64, 21, 21]
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=128, out_features=512),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=512, out_features=visual_features)
        )  # shape: [N, visual_features]
        self.ln = nn.LayerNorm(visual_features)

        # TODO: adjust temperature
        self.temperature = torch.ones(1).to(device)

        grid_x, grid_y = torch.meshgrid(
            torch.linspace(-1.0, 1.0, 21), torch.linspace(-1.0, 1.0, 21), indexing="ij"
        )
        self.x_map = grid_x.reshape(-1).to(device)
        self.y_map = grid_y.reshape(-1).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_model(x)

        b, c, w, h = x.shape
        x = x.view(b * c, w * h)
        softmax_attention = F.softmax(x / self.temperature, dim=1)

        expected_x = torch.sum(self.x_map * softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.y_map * softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat((expected_x, expected_y), 1)
        x = expected_xy.view(b, c * 2)
        x = self.fc(x)

        return self.ln(x)

    def getLoss(self, visual_features: torch.Tensor, obs_state: torch.Tensor) -> torch.Tensor:
        predicted_state = self.train_decoder(visual_features)
        return F.mse_loss(predicted_state, obs_state)
