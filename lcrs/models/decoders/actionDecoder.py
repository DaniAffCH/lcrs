from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
import hydra
import torch.nn.functional as F
from lcrs.utils.gripper_control import world_to_tcp_frame


def log_sum_exp(x):
    """numerically stable log_sum_exp implementation that prevents overflow"""
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        in_features = visual_features + language_features + plan_features
        self.mixtures = mixtures
        self.action_space_size = action_space_size - 1  # remove the gripper
        out_features_gmm = self.action_space_size * mixtures
        self.fcDecoder = hydra.utils.instantiate(decoder, in_size=in_features)
        self.piDecoder = hydra.utils.instantiate(piDecoder, out_size=out_features_gmm)
        self.muDecoder = hydra.utils.instantiate(muDecoder, out_size=out_features_gmm)
        self.sigmaDecoder = hydra.utils.instantiate(sigmaDecoder, out_size=out_features_gmm)
        self.gripperDecoder = hydra.utils.instantiate(gripperDecoder)

        self.log_scale_min = -7.0

        self.action_max_bound = [1., 1., 1., 1., 1., 1., 1.,]
        self.action_min_bound = [-1., -1., -1., -1., -1., -1., -1.,]

        # =============

        self.gripper_bounds = torch.tensor([self.action_min_bound[-1], self.action_max_bound[-1]], device=self.device)
        self.action_max_bound = self.action_max_bound[:-1]
        self.action_min_bound = self.action_min_bound[:-1]

        self.action_max_bound = torch.tensor(self.action_max_bound, device=self.device).float()
        self.action_min_bound = torch.tensor(self.action_min_bound, device=self.device).float()
        assert self.action_max_bound.shape[0] == self.action_space_size
        assert self.action_min_bound.shape[0] == self.action_space_size
        self.action_max_bound = self.action_max_bound.unsqueeze(0).unsqueeze(0)  # [1, 1, action_space]
        self.action_min_bound = self.action_min_bound.unsqueeze(0).unsqueeze(0)  # [1, 1, action_space]
        # broadcast to [1, 1, action_space, N_DIST]
        self.action_max_bound = self.action_max_bound.unsqueeze(-1) * \
            torch.ones(1, 1, self.mixtures, device=self.device)
        # broadcast to [1, 1, action_space, N_DIST]
        self.action_min_bound = self.action_min_bound.unsqueeze(-1) * \
            torch.ones(1, 1, self.mixtures, device=self.device)

        # =============

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

    # FROM HULC
    def getLoss(
        self,
        gtActions: torch.Tensor,
        proprioceptive: torch.Tensor,
        pi: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        gripper: torch.Tensor
    ) -> torch.Tensor:

        actions_tcp = world_to_tcp_frame(gtActions, proprioceptive)
        logistics_loss = self._logistic_loss(pi, sigma, mu, actions_tcp[:, :, :-1])
        gripper_gt = actions_tcp[:, :, -1].clone()
        gripper_gt[gripper_gt == -1] = 0
        gripper_act_loss = F.cross_entropy(gripper.view(-1, 2), gripper_gt.view(-1).long())

        return logistics_loss, gripper_act_loss

    # FROM HULC
    def _logistic_loss(
        self,
        logit_probs: torch.Tensor,
        log_scales: torch.Tensor,
        means: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        # Appropriate scale
        log_scales = torch.clamp(log_scales, min=self.log_scale_min)
        # Broadcast actions (B, A, N_DIST)
        actions = actions.unsqueeze(-1) * torch.ones(1, 1, self.mixtures, device=self.device)
        # Approximation of CDF derivative (PDF)
        centered_actions = actions - means
        inv_stdv = torch.exp(-log_scales)
        act_range = (self.action_max_bound - self.action_min_bound) / 2.0
        plus_in = inv_stdv * (centered_actions + act_range / (self.mixtures - 1))
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_actions - act_range / (self.mixtures - 1))
        cdf_min = torch.sigmoid(min_in)

        # Corner Cases
        log_cdf_plus = plus_in - F.softplus(plus_in)  # log probability for edge case of 0 (before scaling)
        log_one_minus_cdf_min = -F.softplus(min_in)  # log probability for edge case of 255 (before scaling)
        # Log probability in the center of the bin
        mid_in = inv_stdv * centered_actions
        log_pdf_mid = mid_in - log_scales - 2.0 * F.softplus(mid_in)
        # Probability for all other cases
        cdf_delta = cdf_plus - cdf_min

        # Log probability
        log_probs = torch.where(
            actions < self.action_min_bound + 1e-3,
            log_cdf_plus,
            torch.where(
                actions > self.action_max_bound - 1e-3,
                log_one_minus_cdf_min,
                torch.where(
                    cdf_delta > 1e-5,
                    torch.log(torch.clamp(cdf_delta, min=1e-12)),
                    log_pdf_mid - np.log((self.mixtures - 1) / 2),
                ),
            ),
        )
        log_probs = log_probs + F.log_softmax(logit_probs, dim=-1)
        loss = -torch.sum(log_sum_exp(log_probs), dim=-1).mean()
        return loss
