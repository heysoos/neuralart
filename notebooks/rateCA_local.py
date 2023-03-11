import numpy as np
from math import sqrt

from cppn import CPPN, Sampler

import torch
import torch.nn as nn
import torch.nn.functional as F


class Rule(nn.Module):
    def __init__(self, RES, RADIUS=2):
        super().__init__()
        self.radius = RADIUS

        self.eta = 0.05
        self.target_rate = 0.5
        self.excite_prob = .8
        self.constant_current = 0.01

        self.plasticity = False
        self.plastic_lr = 1e-1

        self.afunc = torch.sigmoid


        self.RES = RES
        RESX, RESY = RES[0], RES[1]
        self.NUMEL = RESX * RESY

        ########## CPPN KERNEL ################
        cppn_net_size = [32, 32, 32]
        # cppn_net_size = [1]
        dim_z = 16
        dim_c = self.NUMEL  # CHANNELS
        self.cppn = CPPN(cppn_net_size, dim_z, dim_c).cuda().eval()
        self.sampler = Sampler()

        # radial decay
        # Rk = RADIUS * 2 + 1
        # xm, ym = torch.meshgrid(torch.linspace(-1, 1, Rk), torch.linspace(-1, 1, Rk))
        # rm = (1. / (torch.sqrt(xm ** 2 + ym ** 2).cuda() + 1.))
        # self.rm = rm.reshape(-1).unsqueeze(0).unsqueeze(2)

        self.nearest_neighbours = self.generate_cppn_kernel()


    def generate_cppn_kernel(self):
        scale = 5
        zscale = 2
        z = zscale * torch.randn(1, self.cppn.dim_z).cuda()
        Rk = self.radius * 2 + 1
        coords = self.cppn._coordinates(scale, Rk, Rk, z)
        coords[0] = torch.rand_like(coords[0])
        coords[1] = torch.rand_like(coords[0])
        coords[2] = torch.rand_like(coords[0])
        # coords[0] = coords[2]
        # coords[1] = coords[2]
        # coords[2] = coords[2]

        k = self.cppn.forward(coords, Rk, Rk).reshape(1, Rk, Rk, self.cppn.dim_c).permute(0, 3, 1, 2)
        # k = k / k.sum(dim=-1, keepdim=True).sum(dim=-1, keepdim=True).sqrt()
        # k = k / k.mean(dim=1)

        # radial mask
        # xm, ym = torch.meshgrid(torch.linspace(-1, 1, Rk), torch.linspace(-1, 1, Rk))
        # rm = torch.sqrt(xm ** 2 + ym ** 2).cuda()
        # condition = rm < 1.
        # k = torch.where(condition, k, k*0.)


        # sparsity
        # spars_mat = (torch.rand_like(k) > 0.9) * 1.
        # k = k * spars_mat

        # local kernels
        k = k.permute(0, 2, 3, 1)
        k[:, self.radius, self.radius, :] = 0
        k = k.view(1, -1, self.NUMEL)
        #k = k.reshape(1, -1, 1)
        # k = k.repeat((1, 1, self.NUMEL))

        return k

    def forward(self, x, noise_input):

        Rk = self.radius

        V = x[:, [0], ...] + noise_input
        V_EI = F.pad(V * self.EI, (Rk, Rk, Rk, Rk), mode='circular') #spikes

        I = F.unfold(V_EI, 2 * Rk + 1) * self.nearest_neighbours # * self.rm
        I = I.sum(dim=1).view(1, 1, self.RES[0], self.RES[1])

        V = (1 - self.eta) * V + self.eta * (1 - V) * self.afunc(I)

        if self.plasticity:
            pre = F.unfold(F.pad(V, (Rk, Rk, Rk, Rk), mode='circular'), 2 * Rk + 1)
            post = F.unfold(V, 1)  # pre-trace

            delta_E = (pre * post) * (~self.if_inhib)
            delta_I = pre * (post - self.target_rate) * self.if_inhib
            delta = delta_E + delta_I
            delta[:, Rk * (2 * Rk + 1) + Rk, :] = 0  # set center pixel to 0
            # delta = delta * (self.nearest_neighbours > 1e-6)

            new_k = torch.clip(self.nearest_neighbours + self.plastic_lr * delta, min=0)
            new_k[:, Rk * (2 * Rk + 1) + Rk, :] = 0  # set center pixel to 0
            new_k_I = new_k * self.if_inhib
            new_k_I = new_k_I / (new_k_I.sum(dim=1) + 1e-6) * self.k_sums_I
            new_k_E = new_k * (~self.if_inhib)
            new_k_E = new_k_E / (new_k_E.sum(dim=1) + 1e-6) * self.k_sums_E
            new_k = new_k_I + new_k_E
            self.nearest_neighbours = 0.5 * self.nearest_neighbours + 0.5 * new_k

        z = torch.cat([V, self.EI], axis=1)
        return z

class rateCA_local(nn.Module):
    def __init__(self, RES=(640,640), RADIUS=2):
        super().__init__()
        self.radius = RADIUS
        self.RES = RES
        RESX, RESY = RES[0], RES[1]
        self.rule = Rule((RESX, RESY), RADIUS)

        self.noise = True
        self.first_run = True


    def initGrid(self, reinit=False):
        shape = self.RES
        rand = torch.rand(1, 1, shape[0], shape[1]) * 2.
        rand[0, 0] = (rand[0, 0] > 0.8) * 1.

        xm, ym = torch.meshgrid(torch.linspace(-1, 1, shape[0]), shape[1] / shape[0] * torch.linspace(-1, 1, shape[1]))
        self.rm = torch.sqrt(xm ** 2 + ym ** 2).cuda()

        if reinit:
            self.rule = Rule((self.RES[0], self.RES[1]), self.radius)

        if self.first_run or reinit:
            self.rule.EI = ((torch.rand(1, 1, shape[0], shape[1]) < self.rule.excite_prob) * 2. - 1).cuda()
            self.rule.EI = torch.where(self.rule.EI < 0., self.rule.EI * (1 / (1 - self.rule.excite_prob + 1e-6)),
                                       self.rule.EI)
            # self.rule.EI = torch.where(self.rule.EI < 0.,
            #                            self.rule.EI * (self.rule.excite_prob / (1 - self.rule.excite_prob)),
            #                            self.rule.EI)
            # self.rule.target_rate_mat = self.rule.target_rate * torch.ones_like(self.rule.EI)

            Rk = self.rule.radius
            self.rule.if_inhib = (F.unfold(F.pad(self.rule.EI, (Rk, Rk, Rk, Rk), mode='circular'), 2 * Rk + 1) < 0.)
            self.rule.k_sums_E = (self.rule.nearest_neighbours * (~self.rule.if_inhib)).sum(dim=1)
            self.rule.k_sums_I = (self.rule.nearest_neighbours * self.rule.if_inhib).sum(dim=1)

            self.first_run = False
        return torch.cat([rand.cuda(), self.rule.EI], axis=1)

    def forward(self, x):
        # noise_input = (torch.rand_like(x[:, [1], ...]) > 0.9) * (self.rm > 0.5) * (self.rm - self.rm.mean())
        if self.noise:
            # noise_input = (torch.rand_like(x[:, [1], ...]) < 0.0002) * 1.
            noise_input = torch.rand_like(x[:, [1], ...]) * self.rule.constant_current * (torch.rand_like(x[:, [1], ...]) > 0.9)
        else:
            noise_input = 0.
        return self.rule(x, noise_input)
