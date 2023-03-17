import ipdb
import numpy as np
from math import sqrt

from cppn import CPPN, Sampler

import torch
import torch.nn as nn
import torch.nn.functional as F


class Rule(nn.Module):
    def __init__(self, BETA=1, CHANNELS=1, RADIUS=1, TEMP_ADAPT=False):
        super().__init__()
        self.channels = CHANNELS
        self.beta = BETA
        self.radius = RADIUS
        Rk = 2*RADIUS + 1

        self.temp_adapt = TEMP_ADAPT
        self.alpha = 0.5  # update rate
        self.h = 1e-1  # magnetization coef (growth coef)
        self.eps = 2.05e-1  # decay coef

        nearest_neighbours = torch.zeros(1, CHANNELS, Rk, Rk).cuda()

        nearest_neighbours[:, :, RADIUS, :] = 1.
        nearest_neighbours[:, :, :, RADIUS] = 1.
        nearest_neighbours[:, :, RADIUS, RADIUS] = 0

        self.nearest_neighbours = nn.Parameter(nearest_neighbours, requires_grad=False)

    def forward(self, x, return_E=False):

        s = x[:, :self.channels, ...]
        Rk = self.radius

        s_pad = F.pad(s, (Rk, Rk, Rk, Rk), mode='circular')

        Js = F.conv2d(s_pad, self.nearest_neighbours, padding=0)
        if return_E:
            E = (-s * Js).sum() / 2.
        delta_e = 2 * s * Js

        definite_flip = delta_e <= 0
        p = torch.exp(-delta_e * self.beta)
        p = torch.where(definite_flip, torch.ones_like(s), p)

        rand = torch.rand_like(s)

        dropout_mask = (torch.rand_like(s[0, 0]) > 0.5).unsqueeze(0).unsqueeze(0)
        flip = -2. * torch.logical_and(rand < p, dropout_mask) + 1

        if self.temp_adapt:
            if torch.rand(1) > 0.9:
                alpha = self.alpha
                eps = self.eps / sqrt(s.numel())

                T = (1. / self.beta)
                eps_noise = torch.abs(torch.randn(1).cuda() * T**2)
                deltaT = self.h * s.mean().abs() - eps * eps_noise
                newT = T + deltaT
                newT = (1 - alpha) * T + alpha * newT

                self.beta = 1. / newT

        if return_E:
            return s * flip, E
        else:
            return s * flip

class isingCA(nn.Module):
    def __init__(self, CHANNELS=1, BETA=1, RADIUS=2, TEMP_ADAPT=False):
        super().__init__()
        self.channels = CHANNELS
        self.radius = RADIUS

        self.rule = Rule(BETA, CHANNELS, RADIUS, TEMP_ADAPT)

    def initGrid(self, shape):
        rand = (torch.rand(1, self.channels + 1, shape[0], shape[1]) > torch.rand(1)) * 2. - 1.
        rand[:, -1, ...] = torch.ones_like(rand[:, -1, ...]) * self.rule.beta
        self.rule.beta = torch.cuda.FloatTensor([self.rule.beta])
        return rand.cuda()

    def forward(self, x, return_E=False):
        return self.rule(x, return_E)
