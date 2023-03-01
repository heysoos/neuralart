import numpy as np
from math import sqrt

from cppn import CPPN, Sampler

import torch
import torch.nn as nn
import torch.nn.functional as F


class Rule(nn.Module):
    def __init__(self, BETA=1, CHANNELS=8, RADIUS=2):
        super().__init__()
        self.channels = CHANNELS
        self.beta = BETA
        self.radius = RADIUS
        Rk = 2*RADIUS + 1

        self.temp_adapt = False
        self.alpha = 1e-1  # update rate
        self.h = 1e-1  # magnetization coef (growth coef)
        self.eps = 1e-2  # decay coef
        self.D = 1  # diffusion coef

        xm, ym = torch.meshgrid(torch.linspace(-1, 1, Rk), torch.linspace(-1, 1, Rk))
        rm = torch.sqrt(xm ** 2 + ym ** 2).cuda()

        # nearest_neighbours = torch.ones(1, 1, Rk, Rk).cuda()
        # nearest_neighbours[:, :, RADIUS, RADIUS] = 0
        # nearest_neighbours[:, :, RADIUS + 1, RADIUS + 1] = 0.
        # nearest_neighbours[:, :, RADIUS + 1, RADIUS - 1] = 0.
        # nearest_neighbours[:, :, RADIUS - 1, RADIUS + 1] = 0.
        # nearest_neighbours[:, :, RADIUS - 1, RADIUS - 1] = 0.

        nearest_neighbours = torch.zeros(1, CHANNELS, Rk, Rk).cuda()
        nearest_neighbours[:, :, RADIUS, RADIUS] = 0
        nearest_neighbours[:, :, RADIUS, :] = 1.
        nearest_neighbours[:, :, :, RADIUS] = 1.
        # nearest_neighbours[:, :, -1, -1] = -1.
        # nearest_neighbours[:, :, 0, 0] = -1.
        # nearest_neighbours[:, :, 0, -1] = -1.
        # nearest_neighbours[:, :, -1, 0] = -1.


        # nearest_neighbours = nearest_neighbours.repeat(1, CHANNELS, 1, 1)

        # nearest_neighbours /= nearest_neighbours.norm()
        # nearest_neighbours[0, 1, :, :] = -nearest_neighbours[0, 1, :, :]
        # nearest_neighbours[0, 2, :, :] = -nearest_neighbours[0, 2, :, :]

        # nearest_neighbours = torch.zeros(CHANNELS, Rk, Rk)
        # nearest_neighbours[0, ]

        # nearest_neighbours = torch.ones(1, CHANNELS, Rk, Rk).cuda()
        # nearest_neighbours = nearest_neighbours.unsqueeze(0)
        # nearest_neighbours /= nearest_neighbours.norm()
        # nearest_neighbours[:, :, RADIUS, RADIUS] = 0.
        # nearest_neighbours[:, :, RADIUS + 1, RADIUS] = 0.
        # nearest_neighbours[:, :, RADIUS - 1, RADIUS] = 0.
        # nearest_neighbours[:, :, RADIUS, RADIUS - 1] = 0.
        # nearest_neighbours[:, :, RADIUS, RADIUS + 1] = 0.


        self.nearest_neighbours = nn.Parameter(nearest_neighbours, requires_grad=False)
        # self.bias = nn.Parameter()

    def forward(self, x):

        s = x[:, :self.channels, ...]
        b = x[:, [-1], ...]
        shape = s.shape
        Rk = self.radius

        s_pad = F.pad(s, (Rk, Rk, Rk, Rk), mode='circular')
        b_pad = F.pad(b, (Rk, Rk, Rk, Rk), mode='circular')

        Js = F.conv2d(s_pad, self.nearest_neighbours, padding=0)
        delta_e = 2 * s * Js

        definite_flip = delta_e <= 0
        p = torch.exp(-delta_e * b)
        p = torch.where(definite_flip, torch.ones_like(s), p)

        rand = torch.rand_like(s)

        dropout_mask = (torch.rand_like(s[0, 0]) > 0.5).unsqueeze(0).unsqueeze(0)
        flip = -2. * torch.logical_and(rand < p, dropout_mask) + 1

        if self.temp_adapt and torch.rand(1) > 0.5:
            s_unfold = F.unfold(s_pad, 2*Rk + 1) # localize measurements
            sm = (s_unfold.mean(dim=1) ** 2).reshape(shape)

            diff_T = 1 / F.avg_pool2d(b_pad, 2*Rk + 1, stride=1)
            T = (1. / b)

            newT = self.h * sm ** 2 - self.eps * T + self.D * diff_T

            newT = (1 - self.alpha) * T + self.alpha * newT

            b = 1 / newT


        return torch.cat([(s * flip), b], axis=1)

class isingCA(nn.Module):
    def __init__(self, CHANNELS=1, BETA=1, RADIUS=2):
        super().__init__()
        self.channels = CHANNELS
        self.radius = RADIUS

        self.rule = Rule(BETA, CHANNELS, RADIUS)

    def initGrid(self, shape):
        rand = (torch.rand(1, self.channels + 1, shape[0], shape[1]) > 0.5) * 2. - 1.
        rand[:, -1, ...] = torch.ones_like(rand[:, -1, ...]) * self.rule.beta
        return rand.cuda()

    def forward(self, x):
        return self.rule(x)
