#!/usr/bin/env python3


import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
import numpy as np
from lcrs.utils.distribution import Distribution


class LanguageEncoder(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        hidden_size: int,
        depth: int,
        dist: Distribution
    ):
        super(LanguageEncoder, self).__init__()
        plan_features = dist.class_size * dist.category_size

        layers = [nn.Linear(in_features=in_size, out_features=hidden_size),
                  nn.ReLU(),]
        for _ in range(depth):
            layers.append(nn.Linear(in_features=hidden_size, out_features=hidden_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(in_features=hidden_size, out_features=out_size))

        self.fc = nn.Sequential(*layers)

        self.planProjection = nn.Linear(in_features=plan_features, out_features=out_size)
        self.languageProjection = nn.Linear(in_features=out_size, out_features=out_size)

        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, language: torch.Tensor) -> torch.Tensor:
        return self.fc(language)

    def getLoss(self, planFeatures: torch.Tensor, languageFeatures: torch.Tensor, auxLang) -> torch.Tensor:
        """
        CLIP contrastive loss inspired by
        https://arxiv.org/pdf/2103.00020.pdf
        """

        planFeaturesEmb = self.planProjection(planFeatures[auxLang])
        languageFeaturesEmb = self.languageProjection(languageFeatures[auxLang])

        planFeaturesEmb = planFeaturesEmb / planFeaturesEmb.norm(dim=-1, keepdim=True)
        languageFeaturesEmb = languageFeaturesEmb / languageFeaturesEmb.norm(dim=-1, keepdim=True)

        t = self.temperature.exp()
        logits_per_image = t * planFeaturesEmb @ languageFeaturesEmb.t()
        logits_per_text = logits_per_image.t()

        labels = torch.arange(logits_per_image.shape[0], device=languageFeaturesEmb.device)
        loss_i = cross_entropy(logits_per_image, labels)
        loss_t = cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2

        if not any(auxLang):
            loss.zero_()

        # (Not any(auxLang)) => (loss == 0)
        assert any(auxLang) or (loss == 0)

        return loss

    def getLossAlternative(self, planFeatures: torch.Tensor, languageFeatures: torch.Tensor, auxLang) -> torch.Tensor:
        planFeaturesEmb = self.planProjection(planFeatures[auxLang])
        languageFeaturesEmb = self.languageProjection(languageFeatures[auxLang])
        # normalize embeddings?
        planFeaturesEmb = planFeaturesEmb / planFeaturesEmb.norm(dim=-1, keepdim=True)
        languageFeaturesEmb = languageFeaturesEmb / languageFeaturesEmb.norm(dim=-1, keepdim=True) 

        # ?
        t = self.temperature.exp()
        logits_per_image = t * planFeaturesEmb @ languageFeaturesEmb.t()
        logits_per_text = logits_per_image.t()

        # get labels
        labels = torch.arange(logits_per_image.shape[0], device=languageFeaturesEmb.device)

        # Calculate the contrastive loss as defined by Hadsell et al. in "Dimensionality Reduction by Learning an Invariant Mapping"
        margin = "?"
        y = labels #?
        x1 = planFeatures
        x2 = languageFeatures
        
        # language should be as similar as possible to the recognized plan
        # should be as distant as possible as other plans
        
        d_w = torch.nn.functional.pairwise_distance(w(x1), w(x2))
        loss = (1 - y) * 0.5 * d_w**2 + y * 0.5 * torch.nn.functional.relu(margin - d_w)**2
        