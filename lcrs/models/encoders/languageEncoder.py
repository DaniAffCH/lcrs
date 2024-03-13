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

        planFeaturesEmb = planFeaturesEmb / planFeaturesEmb.norm(dim=-1, keepdim=True)
        languageFeaturesEmb = languageFeaturesEmb / languageFeaturesEmb.norm(dim=-1, keepdim=True)

        t = self.temperature.exp()
        logits_per_image = t * torch.nn.functional.cosine_similarity(planFeaturesEmb, languageFeaturesEmb.t(), dim=-1)
        logits_per_text = logits_per_image.t()
        labels = torch.arange(logits_per_image.shape[0], device=languageFeaturesEmb.device)
        # Contrastive loss le cunn
        loss_i = (1 - labels) * 0.5 * torch.pow(logits_per_image, 2) + labels * 0.5 * torch.pow(torch.clamp(0.5 - logits_per_image, min=0.0), 2)
        loss_t = (1 - labels) * 0.5 * torch.pow(logits_per_text, 2) + labels * 0.5 * torch.pow(torch.clamp(0.5 - logits_per_text, min=0.0), 2) 
        loss = (loss_i + loss_t) / 2
        if not any(auxLang):
            loss.zero_()

        # (Not any(auxLang)) => (loss == 0)
        assert any(auxLang) or (loss == 0)

        return loss