import numpy as np
from math import sqrt

from cppn import CPPN, Sampler

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


class Rule(nn.Module):
    def __init__(self, RADIUS=2, input_size=(480, 640), output_size=(480, 640)):
        super().__init__()
        self.radius = RADIUS
        self.input_size = input_size
        self.output_size = output_size

        self.layer_kernel = 3
        self.layer_stride = 2

        # self.threshhold = 0.68
        self.minimum_threshold = -5
        self.refractor_time = 5
        self.target_rate = 5

        self.energy_recovery = 0.0025
        self.target_energy = 1
        self.spike_cost = 0.1
        self.min_energy = 0.1

        self.excite_prob = 0.8

        Rk = 2*RADIUS + 1

        ########## CPPN KERNEL ################
        cppn_net_size = [32, 32, 32]
        dim_z = 16
        dim_c = 1  # CHANNELS
        self.cppn = CPPN(cppn_net_size, dim_z, dim_c).cuda().eval()
        self.sampler = Sampler()

        k = self.generate_cppn_kernel()

        # radial kernel
        xm, ym = torch.meshgrid(torch.linspace(-1, 1, Rk), torch.linspace(-1, 1, Rk))
        rm = torch.sqrt(xm ** 2 + ym ** 2).cuda()
        condition = rm < 0.8
        null = torch.zeros_like(rm).cuda()
        k = torch.where(condition, k, null)

        self.nearest_neighbours = k
        ###########################################

        ########## NEAREST NEIGHBOUr KERNEL ################
        # nearest_neighbours = torch.ones(1, 1, Rk, Rk).cuda()
        # nearest_neighbours[:, :, RADIUS, RADIUS] = 0
        # nearest_neighbours = nearest_neighbours * 0.1
        # # nearest_neighbours = nearest_neighbours * (torch.rand_like(nearest_neighbours) > 0.9)
        #
        # self.nearest_neighbours = nearest_neighbours
        ###########################################

        downsample = nn.Conv2d(1, 1, self.layer_kernel, stride=self.layer_stride, padding=0, bias=False).cuda()
        upsample = nn.ConvTranspose2d(1, 1, self.layer_kernel, stride=self.layer_stride, padding=0, bias=False).cuda()

        downsample.weight.data = torch.Tensor([[0, 1, 0], [-1, 0, 1], [0, -1, 0]]).unsqueeze(0).unsqueeze(0).cuda()
        # downsample.weight.data = torch.ones_like(downsample.weight.data)
        upsample.weight.data = torch.ones_like(upsample.weight.data)


        self.layers = [downsample, upsample]
        # expected sizes of conv/trans_conv outputs
        self.layer_sizes = [input_size]
        self.layer_sizes.append(conv_output_shape(self.layer_sizes[0], self.layer_kernel, self.layer_stride))
        self.layer_sizes.append([output_size])


        # for i, layer in enumerate(self.layers):
        #     if isinstance(layer, nn.ConvTranspose2d):
        #         self.layer_sizes.append(convtransp_output_shape(self.layer_sizes[i], self.layer_kernel, self.layer_stride))
        #     else:
        #         self.layer_sizes.append(conv_output_shape(self.layer_sizes[i], self.layer_kernel, self.layer_stride))

        # init E/I neurons in space
        self.EI = [((torch.rand(1, 1, input_size[0], input_size[1]) < self.excite_prob) * 2. - 1).cuda()]
        self.EI.append(
            ((torch.rand(1, 1, self.layer_sizes[1][0], self.layer_sizes[1][1]) < self.excite_prob) * 2. - 1).cuda()
        )
        self.EI.append(((torch.rand(1, 1, output_size[0], output_size[1]) < self.excite_prob) * 2. - 1).cuda())

        # for layer_size in self.layer_sizes[1:]:
        #     self.EI.append(
        #         ((torch.rand(1, 1, layer_size[0], layer_size[1]) < self.excite_prob) * 2. - 1).cuda()
        #     )


    def generate_cppn_kernel(self):
        scale = 5
        zscale = 2
        z = zscale * torch.randn(1, self.cppn.dim_z).cuda()
        Rk = self.radius * 2 + 1
        coords = self.cppn._coordinates(scale, Rk, Rk, z)
        coords[0] = 10 + coords[2]
        coords[1] = 10 + coords[2]

        k = self.cppn.forward(coords, Rk, Rk).reshape(1, Rk, Rk, self.cppn.dim_c).permute(0, 3, 1, 2)
        k = k / k.sum().sqrt()
        return k

    def forward(self, x, layer_idx, I=None):

        # print(x.shape, self.EI[layer_idx].shape)
        Rk = self.radius
        x[:, [0], ...] = x[:, [0], ...] * self.EI[layer_idx]
        S = F.pad(x[:, [0], ...], (Rk, Rk, Rk, Rk), mode='circular') #spikes
        V = x[:, [1], ...] # voltages
        R = x[:, [2], ...] + ((torch.rand_like(V) > 0.5) * 2. - 1.) # refractory time
        A = x[:, [3], ...] # traces
        T = x[:, [4], ...] # threshold
        E = x[:, [5], ...] # energy

        # V = V + noise_input #

        if I is None:
            I = F.conv2d(S, self.nearest_neighbours, padding=0)

        V = V + 0.05 * (-V + I)

        S = (V > T) * (R > self.refractor_time) * (E > self.min_energy) * 1.
        E = E + self.energy_recovery * (self.target_energy - E) - (S * self.spike_cost)
        R = (R + 1) * (1 - S)
        A = A - A/200 + S
        T = T + 0.01 * (A - self.target_rate)
        T = torch.maximum(T, self.minimum_threshold * torch.ones_like(T))
        V = V * (1 - S)

        z = torch.cat([S, V, R, A, T, E], axis=1)

        return z

class multilayer_iafCA(nn.Module):
    def __init__(self, RADIUS=2, input_size=(640, 480), output_size=(640, 480)):
        super().__init__()
        self.radius = RADIUS
        self.rule = Rule(RADIUS, input_size=(480, 640), output_size=(480, 640))


    def initGrid(self):
        Xs = [ (torch.rand(1, 6, self.rule.input_size[0], self.rule.input_size[1]) * 2.).cuda() ]

        for layer_size in self.rule.layer_sizes[1:-1]:
            Xs.append(
                (torch.rand(1, 6, layer_size[0], layer_size[1]) * 2.).cuda()
            )

        Xs.append((torch.rand(1, 6, self.rule.output_size[0], self.rule.output_size[1]) * 2.).cuda())

        xm, ym = torch.meshgrid(
            torch.linspace(-1, 1, self.rule.input_size[0]),
            self.rule.input_size[1] / self.rule.input_size[0] * torch.linspace(-1, 1, self.rule.input_size[1]))
        self.rm = torch.sqrt(xm ** 2 + ym ** 2).cuda()

        return Xs

    def forward(self, Xs):
        # print(f"layer 0: {Xs[0].shape}")
        Xs[0] = self.rule(Xs[0], layer_idx=0)

        # multilayer stuff
        i = 1
        for layer in self.rule.layers:
            # calculate input into next layer
            if isinstance(layer, nn.ConvTranspose2d):
                I = layer(Xs[i-1][:, [1], ...], output_size=self.rule.output_size)
            else:
                I = layer(Xs[i-1][:, [1], ...])

            # print(f"layer {i}: {Xs[i].shape}")
            # do iaf rule
            Xs[i] = self.rule(Xs[i], layer_idx=i, I=I)
            i += 1

        return Xs