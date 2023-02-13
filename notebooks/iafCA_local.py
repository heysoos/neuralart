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

        self.integration_constant = 4.
        self.decay_constant = 0.2
        self.constant_current = 0.00

        self.threshhold = 1.
        self.minimum_threshold = 0.1
        self.refractor_time = 5
        self.target_rate = 10

        self.energy = False
        self.energy_recovery = 0.0058
        self.target_energy = 1
        self.spike_cost = 0.1
        self.min_energy = 0.1

        self.excite_prob = .8

        self.plasticity = True
        self.plastic_lr = 1e-4
        self.adaptation = False

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

        self.nearest_neighbours = self.generate_cppn_kernel()

    def generate_cppn_kernel(self):
        scale = 5
        zscale = 2
        z = zscale * torch.randn(1, self.cppn.dim_z).cuda()
        Rk = self.radius * 2 + 1
        coords = self.cppn._coordinates(scale, Rk, Rk, z)
        coords[0] = coords[2]
        coords[1] = coords[2]
        # coords[2] = 10 + coords[0]

        k = self.cppn.forward(coords, Rk, Rk).reshape(1, Rk, Rk, self.cppn.dim_c).permute(0, 3, 1, 2)
        # k = k / k.sum(dim=-1, keepdim=True).sum(dim=-1, keepdim=True).sqrt()
        # k = k / k.mean(dim=1)

        # radial mask
        xm, ym = torch.meshgrid(torch.linspace(-1, 1, Rk), torch.linspace(-1, 1, Rk))
        rm = torch.sqrt(xm ** 2 + ym ** 2).cuda()
        condition = rm < 1.
        k = torch.where(condition, k, k*0.)


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

        pre = F.pad(x[:, [3], ...], (Rk, Rk, Rk, Rk), mode='circular')  # pre-trace
        pre = F.unfold(pre, 2*Rk + 1)

        pre_spike = F.pad(x[:, [0], ...], (Rk, Rk, Rk, Rk), mode='circular')  # pre-trace
        pre_spike = F.unfold(pre_spike, 2*Rk + 1)

        x[:, [0], ...] = x[:, [0], ...] * self.EI
        S = F.pad(x[:, [0], ...], (Rk, Rk, Rk, Rk), mode='circular') #spikes
        V = x[:, [1], ...] # voltages
        R = x[:, [2], ...] # + ((torch.rand_like(V) > 0.5) * 2. - 1.) # refractory time
        A = x[:, [3], ...] # traces
        T = x[:, [4], ...] # threshold
        E = x[:, [5], ...] # energy

        # calculate new spikes
        # V = V + noise_input #

        I = F.unfold(S, 2*Rk + 1) * self.nearest_neighbours
        I = I.sum(dim=1).view(1, 1, self.RES[0], self.RES[1])
        V = V - self.decay_constant * V + self.integration_constant * I + self.constant_current
        S = (V > T) * (R > self.refractor_time) * (E > self.min_energy) * 1.

        # update cell properties

        R = (R + 1) * (1 - S)
        A = A - A/100 + S

        if self.adaptation:
            T = T - T/10000 + 5/1000 * (A - self.target_rate_mat)
            # T = torch.clamp(T, min=self.minimum_threshold)

        if self.energy:
            E = E + self.energy_recovery * (self.target_energy - E) - (S * self.spike_cost)

        V = V * (1 - S)

        if self.plasticity:
            post = F.unfold(A, 1)  # pre-trace
            post_spike = F.unfold(S, 1)
            # delta = (pre * (post - F.unfold(self.target_rate_mat, 1))) * (self.nearest_neighbours > 1e-6) # * (F.unfold(self.EI, 1) < 0.)))

            delta_E = (-pre_spike * post + post_spike * pre) * (~self.if_inhib)
            delta_I = (pre_spike * (post - F.unfold(self.target_rate_mat, 1)) + post_spike * pre) * self.if_inhib
            delta = delta_E + delta_I
            delta[:, Rk * (2 * Rk + 1) + Rk, :] = 0  # set center pixel to 0
            # delta = delta * (self.nearest_neighbours > 1e-6)

            new_k = torch.clip(self.nearest_neighbours + self.plastic_lr * delta, min=0)
            new_k[:, Rk * (2 * Rk + 1) + Rk, :] = 0  # set center pixel to 0
            new_k_I = new_k * self.if_inhib
            new_k_I = new_k_I / (new_k_I.sum(dim=1) + 1e-6) * self.k_sums_I
            new_k_E = new_k * (~self.if_inhib)
            new_k_E = new_k_E / (new_k_E.sum(dim=1) + 1e-6) * self.k_sums_E
            new_k = new_k_I + 0.5 * new_k_E
            # new_k = new_k * (self.nearest_neighbours > 1e-6)


            self.nearest_neighbours = 0.9 * self.nearest_neighbours  + 0.1 * new_k


            #
            # self.nearest_neighbours = new_k / (new_k.sum(dim=1) + 1e-6) * self.k_sums
            # self.nearest_neighbours[:, Rk * (2 * Rk + 1) + Rk, :] = 0  # set center pixel to 0

        S += noise_input
        z = torch.cat([S, V, R, A, T, E, self.EI], axis=1)

        return z

class iafCA_local(nn.Module):
    def __init__(self, RES=(640,640), RADIUS=2):
        super().__init__()
        self.radius = RADIUS
        self.RES = RES
        RESX, RESY = RES[0], RES[1]
        self.rule = Rule((RESX, RESY), RADIUS)

        self.noise = True


    def initGrid(self):
        shape = self.RES
        rand = torch.rand(1, 6, shape[0], shape[1]) * 2.
        rand[0, 0] = (rand[0, 0] > 0.8) * 1.
        rand[0, 4] = torch.ones_like(rand[0, 4]) * self.rule.threshhold
        rand[0, 5] = torch.ones_like(rand[0, 5])

        xm, ym = torch.meshgrid(torch.linspace(-1, 1, shape[0]), shape[1] / shape[0] * torch.linspace(-1, 1, shape[1]))
        self.rm = torch.sqrt(xm ** 2 + ym ** 2).cuda()

        self.rule.EI = ((torch.rand(1, 1, shape[0], shape[1]) < self.rule.excite_prob) * 2. - 1).cuda()
        # self.rule.EI = torch.where(self.rule.EI < 0., self.rule.EI * (1/(1 - self.rule.excite_prob + 1e-6)), self.rule.EI)
        self.rule.target_rate_mat = self.rule.target_rate * torch.ones_like(self.rule.EI)

        Rk = self.rule.radius
        self.rule.if_inhib = (F.unfold(F.pad(self.rule.EI, (Rk, Rk, Rk, Rk), mode='circular'), 2 * Rk + 1) < 0.)
        self.rule.k_sums_E = (self.rule.nearest_neighbours * (~self.rule.if_inhib)).sum(dim=1)
        self.rule.k_sums_I = (self.rule.nearest_neighbours * self.rule.if_inhib).sum(dim=1) * 4
        return torch.cat([rand.cuda(), self.rule.EI], axis=1)

    def forward(self, x):
        # noise_input = (torch.rand_like(x[:, [1], ...]) > 0.9) * (self.rm > 0.5) * (self.rm - self.rm.mean())
        if self.noise:
            noise_input = (torch.rand_like(x[:, [1], ...]) < 0.002) * 1.
        else:
            noise_input = 0.
        return self.rule(x, noise_input)
