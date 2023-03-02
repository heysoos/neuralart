import numpy as np
from math import sqrt

from cppn import CPPN, Sampler

import torch
import torch.nn as nn
import torch.nn.functional as F

def totalistic(x, dim2=False):
    if dim2:
        y_idx = 0
        x_idx = 1
    else:
        y_idx = 2
        x_idx = 3
    z = 0.125 * (x + x.flip(y_idx) + x.flip(x_idx) + x.flip(y_idx).flip(x_idx))
    z = z + 0.125 * (x.transpose(y_idx, x_idx) +
                     x.transpose(y_idx, x_idx).flip(y_idx) +
                     x.transpose(y_idx, x_idx).flip(x_idx) +
                     x.transpose(y_idx, x_idx).flip(y_idx).flip(x_idx))
    # if dim2:
    #     z = z - z.mean()
    # else:
    #     z = z - z.mean(x_idx).mean(y_idx).unsqueeze(y_idx).unsqueeze(x_idx)

    return z

class Rule(nn.Module):
    def __init__(self, BETA=1, CHANNELS=8, RADIUS=2):
        super().__init__()
        self.channels = CHANNELS
        self.beta = BETA
        self.radius = RADIUS
        Rk = 2*RADIUS + 1

        self.temp_adapt = False
        self.alpha = 2e-1  # update rate
        self.h = 1e-1  # magnetization coef (growth coef)
        self.eps = 1e-2  # decay coef
        self.D = 1  # diffusion coef


        xm, ym = torch.meshgrid(torch.linspace(-1, 1, Rk), torch.linspace(-1, 1, Rk))
        rm = torch.sqrt(xm ** 2 + ym ** 2).cuda()

        nearest_neighbours = torch.zeros(1, CHANNELS, Rk, Rk).cuda()
        nearest_neighbours[:, :, RADIUS, :] = 1.
        nearest_neighbours[:, :, :, RADIUS] = 1.

        nearest_neighbours[:, :, -1, -1] = -.1
        nearest_neighbours[:, :, 0, 0] = -.1
        nearest_neighbours[:, :, 0, -1] = -.1
        nearest_neighbours[:, :, -1, 0] = -.1
        nearest_neighbours[:, :, RADIUS, RADIUS] = 0

        ########## CPPN KERNEL ################
        cppn_net_size = [32, 32, 32]
        # cppn_net_size = [1]
        dim_z = 16
        dim_c = self.channels  # CHANNELS
        self.cppn = CPPN(cppn_net_size, dim_z, dim_c).cuda().eval()
        self.sampler = Sampler()

        # nearest_neighbours = self.generate_cppn_kernel()


        self.nearest_neighbours = nn.Parameter(nearest_neighbours, requires_grad=False)

    def generate_cppn_kernel(self):
        scale = 5
        zscale = 2
        z = zscale * torch.randn(1, self.cppn.dim_z).cuda()
        Rk = self.radius * 2 + 1
        coords = self.cppn._coordinates(scale, Rk, Rk, z)
        coords[0] = coords[2]
        coords[1] = coords[2]
        coords[2] = 10 + coords[0]

        k = self.cppn.forward(coords, Rk, Rk).reshape(1, Rk, Rk, self.cppn.dim_c).permute(0, 3, 1, 2)
        # k = k / k.sum(dim=-1, keepdim=True).sum(dim=-1, keepdim=True).sqrt()
        # k = k / k.mean(dim=1)
        k = k - k.mean() * 0.5

        # radial mask
        # xm, ym = torch.meshgrid(torch.linspace(-1, 1, Rk), torch.linspace(-1, 1, Rk))
        # rm = torch.sqrt(xm ** 2 + ym ** 2).cuda()
        # condition = rm < 1.
        # k = torch.where(condition, k, k*0.)


        # sparsity
        spars_mat = (torch.rand_like(k) > 0.9) * 1.
        k = k * spars_mat

        k = totalistic(k)
        k[:, :, self.radius, self.radius] = 0


        return k

    def forward(self, x, h):
        '''
        x: spins and temperatures
        h: input matgnetic field
        '''

        s = x[:, :self.channels, ...]
        b = x[:, [-1], ...]
        shape = s.shape
        Rk = self.radius

        s_pad = F.pad(s, (Rk, Rk, Rk, Rk), mode='circular')
        b_pad = F.pad(b, (Rk, Rk, Rk, Rk), mode='circular')

        Js = F.conv2d(s_pad, self.nearest_neighbours, padding=0)
        if h == None:
            delta_e = 2 * s * Js
        else:
            delta_e = s * (2 * Js + h)

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

    def forward(self, x, h=None):
        return self.rule(x, h)
