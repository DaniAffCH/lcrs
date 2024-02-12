from collections import namedtuple
from typing import Union

import torch
from torch.distributions import Independent, Normal, OneHotCategoricalStraightThrough  # type: ignore
import torch.nn as nn
import torch.nn.functional as F

State = namedtuple("DiscState", ["logit"])


class Distribution:
    def __init__(self, category_size, class_size):
        self.category_size = category_size
        self.class_size = class_size

    def get_dist(self, state):
        shape = state.logit.shape
        logits = torch.reshape(state.logit, shape=(*shape[:-1], self.category_size, self.class_size))
        return Independent(OneHotCategoricalStraightThrough(logits=logits), 1)

    def detach_state(self, state):
        return State(state.logit.detach())

    def sample_latent_plan(self, distribution):
        sampled_plan = distribution.sample()
        sampled_plan = torch.flatten(sampled_plan, start_dim=-2, end_dim=-1)
        return sampled_plan

    def get_state(self, x):
        return State(x)
