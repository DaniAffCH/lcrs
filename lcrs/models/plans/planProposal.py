#!/usr/bin/env python3
import torch
import torch.nn as nn
from lcrs.utils.distribution import Distribution
from lcrs.utils.distribution import State
import torch.distributions as D


class PlanProposal(nn.Module):
    def __init__(
        self,
        visual_features: int,
        language_features: int,
        hidden_size: int,
        depth: int,
        dist: Distribution
    ):
        super(PlanProposal, self).__init__()
        self.dist = dist
        in_size = visual_features + language_features

        plan_features = dist.class_size * dist.category_size

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

    def getLoss(self, planProposalState: State, planRecognitionState: State) -> torch.Tensor:
        planProposalDist = self.dist.get_dist(planProposalState)
        planRecognitionDist = self.dist.get_dist(planRecognitionState)

        planProposalDistDetached = self.dist.get_dist(self.dist.detach_state(planProposalState))
        planRecognitionDistDetached = self.dist.get_dist(self.dist.detach_state(planRecognitionState))

        kl_lhs = D.kl_divergence(planRecognitionDistDetached, planProposalDist).mean()
        kl_rhs = D.kl_divergence(planRecognitionDist, planProposalDistDetached).mean()

        return 0.8 * kl_lhs + 0.2 * kl_rhs  # The kl divergence is not symmetric!
