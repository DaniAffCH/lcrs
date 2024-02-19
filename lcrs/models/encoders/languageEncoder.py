#!/usr/bin/env python3


import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
import numpy as np
from utils.distribution import Distribution


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

    def getLossAlternative(self, seq_vis_feat, encoded_lang, use_for_aux_loss):
        """
        CLIP style contrastive loss, adapted from 'Learning transferable visual models from natural language
        supervision' by Radford et al.
        We maximize the cosine similarity between the visual features of the sequence i and the corresponding language
        features while, at the same time, minimizing the cosine similarity between the current visual features and other
        language instructions in the same batch.

        Args:
            seq_vis_feat: Visual embedding.
            encoded_lang: Language goal embedding.
            use_for_aux_loss: Mask of which sequences in the batch to consider for auxiliary loss.

        Returns:
            Contrastive loss.
        """

        skip_batch = False
        if use_for_aux_loss is not None:
            if not torch.any(use_for_aux_loss):
                # Hack for avoiding a crash when using ddp. Loss gets multiplied with 0 at the end of method to
                # effectively skip whole batch. We do a dummy forward pass, to prevent ddp from complaining.
                # see https://github.com/pytorch/pytorch/issues/43259
                skip_batch = True
                seq_vis_feat = seq_vis_feat[0:1]
                encoded_lang = encoded_lang[0:1]
            else:
                seq_vis_feat = seq_vis_feat[use_for_aux_loss]
                encoded_lang = encoded_lang[use_for_aux_loss]
        image_features = self.planProjection(seq_vis_feat)
        lang_features = self.languageProjection(encoded_lang)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = lang_features / lang_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.temperature.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        logits_per_text = logits_per_image.t()

        # symmetric loss function
        labels = torch.arange(logits_per_image.shape[0], device=text_features.device)
        loss_i = cross_entropy(logits_per_image, labels)
        loss_t = cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2

        if skip_batch:
            loss *= 0

        return loss
