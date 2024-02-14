from typing import Dict, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
import hydra


class ActionDecoder(nn.Module):
    def __init__(self,
                 decoder: DictConfig,
                 piDecoder: DictConfig,
                 muDecoder: DictConfig,
                 sigmaDecoder: DictConfig,
                 gripperDecoder: DictConfig,
                 visual_features: int,
                 language_features: int,
                 plan_features: int,
                 hidden_size: int,
                 action_space_size: int,
                 mixtures: int):
        super().__init__()

        in_features = visual_features + language_features + plan_features
        self.mixtures = mixtures
        self.action_space_size = action_space_size
        out_features_gmm = action_space_size * mixtures
        self.fcDecoder = hydra.utils.instantiate(decoder, in_size=in_features)
        self.piDecoder = hydra.utils.instantiate(piDecoder, out_size=out_features_gmm)
        self.muDecoder = hydra.utils.instantiate(muDecoder, out_size=out_features_gmm)
        self.sigmaDecoder = hydra.utils.instantiate(sigmaDecoder, out_size=out_features_gmm)
        self.gripperDecoder = hydra.utils.instantiate(gripperDecoder)

    def forward(self, plan: torch.Tensor, visual: torch.Tensor, language: torch.Tensor) -> torch.Tensor:
        batchSize = visual.shape[0]
        sequenceLength = visual.shape[1]
        plan = plan.unsqueeze(1).expand(-1, sequenceLength, -1)
        language = language.unsqueeze(1).expand(-1, sequenceLength, -1)
        x = torch.cat([plan, visual, language], dim=-1)

        x = torch.cat([plan, visual, language], dim=-1)
        x = self.fcDecoder(x)

        # sum( pi * N(mu, sigma) )

        pi = self.piDecoder(x)
        mu = self.muDecoder(x)
        sigma = self.sigmaDecoder(x)

        gripper = self.gripperDecoder(x)

        pi = pi.view(batchSize, sequenceLength, self.action_space_size, self.mixtures)
        mu = mu.view(batchSize, sequenceLength, self.action_space_size, self.mixtures)
        sigma = sigma.view(batchSize, sequenceLength, self.action_space_size, self.mixtures)

        return pi, mu, sigma, gripper
