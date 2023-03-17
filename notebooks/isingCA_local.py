import numpy as np
from math import sqrt

from cppn import CPPN, Sampler

import torch
import torch.nn as nn
import torch.nn.functional as F


class Rule(nn.Module):
    def __init__(self, BETA=1, CHANNELS=8, RADIUS=1):
        super().__init__()
        self.channels = CHANNELS
        self.beta = BETA
        self.radius = RADIUS
        Rk = 2*RADIUS + 1

        self.temp_adapt = False
        self.alpha = 0.5  # update rate
        self.h = 1e-1  # magnetization coef (growth coef)
        self.eps = 2.00e-2  # decay coef
        self.D = 1. #2. * self.eps  # diffusion coef

        self.m_pow = 2.
        self.temp_pow = 1.
        self.temp_kernel_size = 1

        nearest_neighbours = torch.zeros(1, CHANNELS, Rk, Rk).cuda()

        nearest_neighbours[:, :, RADIUS, :] = 1.
        nearest_neighbours[:, :, :, RADIUS] = 1.
        nearest_neighbours[:, :, RADIUS, RADIUS] = 0

        self.nearest_neighbours = nn.Parameter(nearest_neighbours, requires_grad=False)

    def forward(self, x):

        s = x[:, :self.channels, ...]
        b = x[:, [-1], ...]
        shape = s.shape
        Rk = self.radius

        s_pad = F.pad(s, (Rk, Rk, Rk, Rk), mode='circular')
        # b_pad = F.pad(b, (Rk, Rk, Rk, Rk), mode='circular')

        Js = F.conv2d(s_pad, self.nearest_neighbours, padding=0)
        delta_e = 2 * s * Js

        definite_flip = delta_e <= 0
        p = torch.exp(-delta_e * b)
        p = torch.where(definite_flip, torch.ones_like(s), p)

        rand = torch.rand_like(s)

        dropout_mask = (torch.rand_like(s[0, 0]) > 0.5).unsqueeze(0).unsqueeze(0)
        flip = -2. * torch.logical_and(rand < p, dropout_mask) + 1

        if self.temp_adapt and torch.rand(1) > 0.9:
            temp_rad = self.temp_kernel_size*Rk
            temp_kernel_size = 2*temp_rad + 1
            pads = tuple([Rk * self.temp_kernel_size for i in range(4)])

            s_unfold = F.unfold(s_pad, 2*Rk + 1) # localize measurements
            sm = (s_unfold.mean(dim=1)).reshape(shape)

            b_tpad = F.pad(b, pads, mode='circular')
            T_pad = 1. / b_tpad
            T = T_pad[..., temp_rad:-temp_rad, temp_rad:-temp_rad]
            diff_T = (F.avg_pool2d(T_pad, temp_kernel_size, stride=1) - T)

            # newT = self.h * sm ** 2 - self.eps * T + self.D * diff_T
            deltaT = self.h * sm.abs() ** self.m_pow \
                     - self.eps * T ** self.temp_pow +\
                     self.D * diff_T
            newT = T + deltaT
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
        rand = (torch.rand(1, self.channels + 1, shape[0], shape[1]) > torch.rand(1)) * 2. - 1.
        rand[:, -1, ...] = torch.ones_like(rand[:, -1, ...]) * self.rule.beta
        return rand.cuda()

    def forward(self, x):
        return self.rule(x)
