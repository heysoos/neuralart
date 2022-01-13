import numpy as np

from cppn import CPPN, Sampler

import torch
import torch.nn as nn
import torch.nn.functional as F

class Rule(nn.Module):
    def __init__(self, CHANNELS=8, RADIUS=2):
        super().__init__()
        self.channels = CHANNELS
        self.radius = RADIUS

        self.alpha = 4
        self.mu = 0.15
        self.sigma = 0.017

        ###########################################
        # init CPPN to generate kernels
        cppn_net_size = [32, 32, 32]
        dim_z = 16
        dim_c = 1 #CHANNELS
        self.cppn = CPPN(cppn_net_size, dim_z, dim_c).cuda().eval()
        self.sampler = Sampler()
        ###########################################

        k = self.generate_kernel()
        # k = self.generate_random_kernels()
        # k = self.generate_cppn_kernel()
        # k = self.generate_cppn_kernels()

        self.K = nn.Parameter(k)
        self.G = lambda x: 2 * torch.exp(- torch.abs(x - self.mu)**2 / (2*self.sigma**2)) - 1
        # self.G = lambda x: 2 * torch.exp(-(x - self.mu)**2 / (2*self.sigma**2)) - 1

    def generate_cppn_kernel(self):
        scale = 5
        zscale = 2
        z = zscale * torch.randn(1, self.cppn.dim_z).cuda()
        Rk = self.radius * 2 + 1
        coords = self.cppn._coordinates(scale, Rk, Rk, z)
        coords[0] = 10 + coords[2] * 2
        coords[1] = 10 + coords[2] / 2
        coords[2] = 10 + 5 * coords[2]

        k = self.cppn.forward(coords, Rk, Rk).reshape(1, Rk, Rk, self.cppn.dim_c).permute(0, 3, 1, 2)
        k = k / k.sum()
        k = k.repeat(1, self.channels, 1, 1)
        return k

    def generate_cppn_kernels(self):
        scale = 5
        zscale = 2
        ks = []
        for i in range(self.channels):
            z = zscale * torch.randn(1, self.cppn.dim_z).cuda()
            Rk = self.radius * 2 + 1
            coords = self.cppn._coordinates(scale, Rk, Rk, z)
            coords[0] = 10 + coords[2] * 2
            coords[1] = 10 + coords[2] / 2
            coords[2] = 10 + 5 * coords[2]

            k = self.cppn.forward(coords, Rk, Rk).reshape(1, Rk, Rk, self.cppn.dim_c).permute(0, 3, 1, 2)
            k = k / k.sum()
            ks.append(k)
        ks = torch.cat(ks, dim=1)

        return ks

    def generate_kernel(self):
        Rk = 2 * self.radius + 1
        xm, ym = torch.meshgrid(torch.linspace(-1, 1, Rk), torch.linspace(-1, 1, Rk))
        rm = torch.sqrt(xm ** 2 + ym ** 2).cuda()
        rm1 = torch.where(rm <= 1. - 1e-6, rm, torch.zeros_like(rm))

        u = self.alpha * (1 - 1 / (4 * rm1 * (1 - rm1)))
        k = torch.exp(u)
        k = k / k.sum()
        k = k.repeat(1, self.channels, 1, 1)
        return k

    def generate_random_kernels(self):
        Rk = 2 * self.radius + 1
        xm, ym = torch.meshgrid(torch.linspace(-1, 1, Rk), torch.linspace(-1, 1, Rk))
        rm = torch.sqrt(xm ** 2 + ym ** 2).cuda()
        rm1 = torch.where(rm <= 1. - 1e-6, rm, torch.zeros_like(rm))
        ks = []
        for i in range(self.channels):
            u = np.abs(np.random.normal(4)) * (1 - 1 / (4 * rm1 * (1 - rm1)))
            k = torch.exp(u)
            k = k / k.sum()
            ks.append(k)
        ks = torch.stack(ks, dim=0).unsqueeze(0)
        return ks

    def forward(self, x):

        Rk = self.radius
        A = F.pad(x, (Rk, Rk, Rk, Rk), mode='circular')
        dt = 0.1 * self.G(F.conv2d(A, self.K, padding=0))
        #
        # dropout_mask = (torch.rand_like(x) > 0.5)
        # dt = dt * dropout_mask
        return torch.clip(x + dt, 0, 1)

class leniaCA(nn.Module):
    def __init__(self, CHANNELS=1, RADIUS=2):
        super().__init__()
        self.channels = CHANNELS
        self.radius = RADIUS

        self.rule = Rule(CHANNELS, RADIUS)

    def initGrid(self, shape):
        rand = np.random.rand(1, self.channels, shape[0], shape[1])
        return torch.cuda.FloatTensor(rand)

    def forward(self, x):
        return self.rule(x)
