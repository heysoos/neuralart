import numpy as np
from math import sqrt

from cppn import CPPN, Sampler

import torch
import torch.nn as nn
import torch.nn.functional as F


class Rule(nn.Module):
    def __init__(self, RES, BETA=1, RADIUS=1):
        super().__init__()
        self.beta = BETA
        self.radius = RADIUS
        self.res = RES
        Rk = 2*RADIUS + 1

        self.J_adapt = False
        self.adapt_lr = 1.
        self.alpha = 0.5
        self.h = 0.001
        self.eps = 0.01
        self.max_weight = 4.

        self.numel = RES[0] * RES[1]

        nearest_neighbours = torch.zeros(1, Rk, Rk, self.numel).cuda()

        nearest_neighbours[:, RADIUS, :, :] = 1.
        nearest_neighbours[:, :, RADIUS, :] = 1.
        nearest_neighbours[:, RADIUS, RADIUS, :] = 0

        self.nearest_neighbours = nearest_neighbours.reshape(1, -1, self.numel)

    def forward(self, x):

        s = x[:, [0], ...]
        tr = x[:, [1], ...]
        b = x[:, [-1], ...]
        shape = s.shape
        Rk = self.radius

        s_pad = F.pad(s, (Rk, Rk, Rk, Rk), mode='circular')

        Js = (F.unfold(s_pad, 2*Rk+1) * self.nearest_neighbours).sum(dim=1).view(*shape)
        delta_e = 2 * s * Js

        definite_flip = delta_e <= 0
        p = torch.exp(-delta_e * b)
        p = torch.where(definite_flip, torch.ones_like(s), p)

        rand = torch.rand_like(s)

        dropout_mask = (torch.rand_like(s[0, 0]) > 0.5).unsqueeze(0).unsqueeze(0)
        flip = -2. * torch.logical_and(rand < p, dropout_mask) + 1

        if self.J_adapt:
            s_i = F.unfold(tr, 1)
            s_j = F.unfold(F.pad(tr, (Rk, Rk, Rk, Rk), mode='circular'), 2*Rk+1)
            m = s_j.mean(dim=1)
            dJ_hebb = self.h * (s_i * s_j) / (4 * (Rk + 1)) ## should this be with traces instead? :/
            noise_eps = self.eps / (4 * (Rk + 1)) * self.nearest_neighbours
            dJ = dJ_hebb - noise_eps * m ** 2

            new_J = self.nearest_neighbours + dJ * self.adapt_lr
            new_J[:, Rk * (2 * Rk + 1) + Rk, :] = 0.  # set center pixel to 0.

            # new_J_sum = new_J.abs().sum(dim=1, keepdim=True)
            # new_J_sum *= (1 + 0.1 * torch.randn_like(new_J_sum))  # noisy normalization
            # new_J = new_J * (self.max_weight / new_J_sum)

            self.nearest_neighbours = (1 - self.alpha) * self.nearest_neighbours + self.alpha * new_J

        tr = 0.9 * tr + 0.1 * s
        return torch.cat([(s * flip), tr, b], axis=1)

class isingCA(nn.Module):
    def __init__(self, RES, BETA=1, RADIUS=2):
        super().__init__()
        self.radius = RADIUS
        self.rule = Rule(RES, BETA, RADIUS)

    def initGrid(self, unfold=True):
        shape = self.rule.res
        rand = (torch.rand(1, 3, shape[0], shape[1]) > torch.rand(1)) * 2. - 1.
        rand[:, 1, ...] = torch.zeros_like(rand[:, 1, ...])
        rand[:, -1, ...] = torch.ones_like(rand[:, -1, ...]) * self.rule.beta

        return rand.cuda()

    def forward(self, x):
        return self.rule(x)
