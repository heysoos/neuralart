import numpy as np
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


HNM_STEPS = 3000  # Max number of steps to mine for a hard-negative sample before while-loop terminates

def totalistic(x):
    z = 0.125 * (x + x.flip(2) + x.flip(3) + x.flip(2).flip(3))
    z = z + 0.125 * (x.transpose(2, 3) + x.transpose(2, 3).flip(2) + x.transpose(2, 3).flip(3) + x.transpose(2, 3).flip(
        2).flip(3))
    z = z - z.mean(3).mean(2).unsqueeze(2).unsqueeze(3)

    return z


class Rule(nn.Module):
    def __init__(self, CHANNELS=8, FILTERS=1, HIDDEN=16, RADIUS=2):
        super().__init__()
        self.channels = CHANNELS
        self.filters = FILTERS
        self.hidden = HIDDEN

        Rk = RADIUS * 2 + 1
        self.filter1 = nn.Parameter(torch.randn(FILTERS * CHANNELS, 1, Rk, Rk) / sqrt(FILTERS * CHANNELS))
        self.bias1 = nn.Parameter(0 * torch.randn(FILTERS * CHANNELS))

        self.filter2 = nn.Conv2d(FILTERS * CHANNELS, HIDDEN, 1, padding_mode='circular')
        nn.init.orthogonal_(self.filter2.weight, gain=2)
        nn.init.zeros_(self.filter2.bias)
        self.filter3 = nn.Conv2d(HIDDEN, CHANNELS, 1, padding_mode='circular')
        nn.init.orthogonal_(self.filter3.weight, gain=2)
        nn.init.zeros_(self.filter3.bias)


class CA(nn.Module):
    def __init__(self, CHANNELS=8, FILTERS=1, HIDDEN=16, RADIUS=2):
        super().__init__()
        self.channels = CHANNELS
        self.filters = FILTERS
        self.hidden = HIDDEN
        self.radius = RADIUS

        self.rule = Rule(CHANNELS, FILTERS, HIDDEN, RADIUS)
        self.optim = torch.optim.Adam(self.parameters(), lr=1e-3)

    def initGrid(self, BS, RES):
        self.psi = torch.cuda.FloatTensor(2 * np.random.rand(BS, self.channels, RES, RES) - 1)

    def get_living_mask(self, x, alive_thres=0.1, dead_thres=0.5):
        alpha_channel = x[:, 3:4, :, :]
        R = self.radius
        alpha_channel = F.pad(alpha_channel, (R, R, R, R), mode='circular')

        alive_mask = F.max_pool2d(alpha_channel, kernel_size=2 * R + 1, stride=1, padding=R) > alive_thres
        alive_mask = alive_mask[:, :, R:-R, R:-R]

        # death_mask = F.avg_pool2d(alpha_channel, kernel_size=2 * R + 1, stride=1, padding=R) < dead_thres
        # death_mask = death_mask[:, :, R:-R, R:-R]
        # return alive_mask & death_mask

        return alive_mask

    def forward(self):
        '''
        The first filter applies a depthwise convolution to the CA grid. Each channel in the filter is applied to its corresponding channel in the CA grid.
        The second and third filters are 1x1 convolutions which act to mix the channels.
        If I understand this correctly, this is essentially applying a depthwise seperable convolution operation on the input (but I am a bit uncertain).
        '''

        weights = totalistic(self.rule.filter1)
        # weights = self.rule.filter1
        bias = self.rule.bias1
        R = self.radius
        # z = F.conv2d(self.psi, weight=weights, bias=bias, padding=2, groups=CHANNELS)
        self.psi = F.pad(self.psi, (R, R, R, R), 'circular')
        z = F.conv2d(self.psi, weight=weights, bias=bias, padding=0, groups=self.channels)

        z = F.leaky_relu(z)
        z = F.leaky_relu(self.rule.filter2(z))

        self.psi = torch.tanh(self.psi[:, :, R:-R, R:-R] + self.rule.filter3(z))

    #         self.psi = torch.clamp(self.psi[:, :, RADIUS:-RADIUS, RADIUS:-RADIUS] + self.rule.filter3(z), 0, 1)

    def forward_masked(self, dt=1):
        '''
        The first filter applies a depthwise convolution to the CA grid. Each channel in the filter is applied to its corresponding channel in the CA grid.
        The second and third filters are 1x1 convolutions which act to mix the channels.
        If I understand this correctly, this is essentially applying a depthwise seperable convolution operation on the input (but I am a bit uncertain).
        '''

        pre_life_mask = self.get_living_mask(self.psi)
        weights = totalistic(self.rule.filter1)
        bias = self.rule.bias1
        R = self.radius

        self.psi = F.pad(self.psi, (R, R, R, R), 'circular')
        z = F.conv2d(self.psi, weight=weights, bias=bias, padding=0, groups=self.channels)

        z = F.leaky_relu(z)
        z = F.leaky_relu(self.rule.filter2(z))

        # self.psi = torch.tanh(self.psi[:, :, R:-R, R:-R] + dt*self.rule.filter3(z))
        self.psi = torch.clamp(self.psi[:, :, R:-R, R:-R] + dt*self.rule.filter3(z), -1, 1)

        post_living_mask = self.get_living_mask(self.psi)

        self.psi = self.psi * (pre_life_mask & post_living_mask).type(torch.cuda.FloatTensor)

    def cleanup(self):
        del self.psi


class Embedder(nn.Module):
    def __init__(self, DIM=16):
        super().__init__()

        self.c1 = nn.Conv2d(3, 32, 3, padding=1)
        nn.init.orthogonal_(self.c1.weight, gain=sqrt(2))
        self.p1 = nn.AvgPool2d(2)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        nn.init.orthogonal_(self.c2.weight, gain=sqrt(2))
        self.p2 = nn.AvgPool2d(2)
        self.c3 = nn.Conv2d(64, 128, 3, padding=1)
        nn.init.orthogonal_(self.c3.weight, gain=sqrt(2))
        self.p3 = nn.AvgPool2d(2)
        self.c4 = nn.Conv2d(128, 128, 3, padding=1)
        nn.init.orthogonal_(self.c4.weight, gain=sqrt(2))
        self.p4 = nn.AvgPool2d(2)
        self.c5 = nn.Conv2d(128, DIM, 3, padding=1)

    def forward(self, x):
        z = self.p1(F.leaky_relu(self.c1(x)))
        z = self.p2(F.leaky_relu(self.c2(z)))
        z = self.p3(F.leaky_relu(self.c3(z)))
        z = self.p4(F.leaky_relu(self.c4(z)))
        z = self.c5(z).mean(3).mean(2)

        z = z / torch.sqrt(1 + torch.sum(z ** 2, 1).unsqueeze(1))

        return z


def findHardNegative(zs, margin, HNM_STEPS=1000):
    '''
    For N steps, find a pair of CAs (i, k) such that the distance between their embeddings is larger than some threshold.
    If such a pairing is found, break out of the loop and return the indices of the pair and the number of steps it took to find the pair.
    '''
    step = 0

    while step < HNM_STEPS:
        i = np.random.randint(zs.shape[0])
        j = i
        k = np.random.randint(zs.shape[0] - 1)
        if k >= i:
            k += 1

        i2 = np.random.randint(zs.shape[1])
        j2 = np.random.randint(zs.shape[1] - 1)
        if j2 >= i2:
            j2 += 1
        k2 = np.random.randint(zs.shape[1])

        z1 = zs[i, i2]
        z2 = zs[j, j2]
        z3 = zs[k, k2]

        delta = np.sqrt(np.sum((z1 - z2) ** 2, axis=0)) - np.sqrt(np.sum((z1 - z3) ** 2, axis=0))
        if delta >= -margin:
            break
        step += 1

    return i, k, step