#!/usr/bin/env python3

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class ImageEncoder(nn.Module):
    def __init__(
        self,
        num_c: int,
        visual_features: int
    ):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super(ImageEncoder, self).__init__()

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
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=128, out_features=512), nn.LeakyReLU(), nn.Dropout(
                0.1), nn.Linear(in_features=512, out_features=visual_features)
        )  # shape: [N, 512]
        self.fc2 = nn.Linear(in_features=512, out_features=visual_features)  # shape: [N, 64]
        self.ln = nn.LayerNorm(visual_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_model(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return self.ln(x)  # shape: [N, 64]
