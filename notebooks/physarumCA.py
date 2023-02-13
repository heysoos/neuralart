import numpy as np
from math import sqrt

from cppn import CPPN, Sampler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softmax


class Rule(nn.Module):
    def __init__(self):
        super().__init__()

        self.moves = make_moves()
        self.directions = make_directions()
        self.kernel = make_kernel()
        self.decay_kernel = self.make_decay_kernel(Rk=5, KCENTER=0., KSMOOTH=0.7, OUTR=1., INR=0.)

    def make_decay_kernel(self, Rk, KCENTER, KSMOOTH, OUTR, INR):
        # make a kernel of size Rk that decays like a gaussian and has a inner/outer cutoff radius
        xm, ym = torch.meshgrid(torch.linspace(-1, 1, Rk), torch.linspace(-1, 1, Rk))
        rm = torch.sqrt(xm ** 2 + ym ** 2).type(torch.double)
        decay = torch.exp(-(rm - KCENTER) ** 2 / (KSMOOTH ** 2 + 1e-6))
        decay = torch.where(rm <= OUTR, decay, 0.)
        decay = torch.where(rm >= INR, decay, 0.)
        decay = decay / decay.sum()
        decay = decay.type(torch.cuda.FloatTensor)
        return decay.unsqueeze(0).unsqueeze(0)


    def forward(self, mass, trail, momentum, force, A=1., dt=.1, temp=10.):

        # trail = trail - 0.1 * trail + A * F.conv2d(F.pad(trail, (2, 2, 2, 2), 'circular'), self.decay_kernel) + mass
        # trail = A * trail
        # trail = F.avg_pool2d(mass + (1 - A) * trail, 1, 1)

        # shape = trail.shape
        # trail[..., shape[-2]//2, shape[-1]//2] = 1e3 # food deposition

        trail = (trail + mass)
        trail = F.avg_pool2d(F.pad(trail, (1, 1, 1, 1), 'circular'), 3, 1)
        trail = (1 - A) * trail

        # new_force = conv_pad(trail, -self.kernel, padding=1).permute(1, 0, 2, 3)
        # force_delta = new_force - force
        # force = force + force_delta * 0.5

        velocity = torch.where(mass < 1e-8, momentum.new_zeros(()), momentum / mass)
        theta = torch.atan2(velocity[[0]], velocity[[1]])

        force = F.pad(trail, (1, 1, 1, 1), mode='circular')
        force = F.unfold(force, 3)



        momentum = torch.where(mass < 1e-8, momentum.new_zeros(()), momentum + force * dt)
        velocity = torch.where(mass < 1e-8, momentum.new_zeros(()), momentum / mass )
        # velocity *= torch.randn_like(velocity)



        # Update mass
        propagation = torch.einsum('dn,nchw->cdhw', self.directions, velocity)
        propagation = torch.where(propagation <= 0., 1e-10 * torch.ones_like(propagation), propagation)
        propagation = softmax(propagation.mul(temp), dim=1)
        propagation = torch.where(propagation <= 0., propagation.new_zeros(()), propagation)

        mass_tot = mass.sum()
        mass_propagation = mass * propagation
        momentum_propagation = momentum * propagation

        mass = conv_pad(mass_propagation, self.moves)
        momentum = conv_pad(momentum_propagation, self.moves)
        mass = mass_tot * mass / (mass.sum() + 1e-10)

        return mass, trail, momentum, force



class physarumCA(nn.Module):
    def __init__(self):
        super().__init__()
        self.rule = Rule()

    def forward(self, mass, trail, momentum, force, A):
        return self.rule(mass, trail, momentum, force, A)

def conv_pad(
    input: torch.Tensor, weight: torch.Tensor, padding: int = 1, groups: int = 1
) -> torch.Tensor:
    input = torch.nn.functional.pad(
        input, [padding, padding, padding, padding], "circular"
    )
    input = torch.nn.functional.conv2d(
        input, weight, bias=None, stride=1, padding=0, dilation=1, groups=groups
    )
    return input

def make_moves(device="cuda"):
    moves = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ]
        ],
        requires_grad=False,
        device=device,
    )
    return moves


def make_directions(device="cuda"):
    sq2 = 2 ** -0.5
    directions = torch.tensor(
        [
            [sq2, sq2],
            [0.0, 1.0],
            [-sq2, sq2],
            [1.0, 0.0],
            [0.0, 0.0],
            [-1.0, 0.0],
            [sq2, -sq2],
            [0.0, -1.0],
            [-sq2, -sq2],
        ],
        requires_grad=False,
        device=device,
    )
    return directions

def make_kernel(x_mul=1.0, y_mul=1.0, pow=1., swap=False, device="cuda"):
    grid = torch.tensor([-1.0, 0.0, 1.0], device=device)
    grid_x, grid_y = torch.meshgrid(grid, grid)
    d = (grid_x ** 2 + grid_y ** 2).sqrt()
    d = torch.where(torch.isclose(d, torch.zeros(())), d.new_zeros(()), 1.0 / d)
    grid_x = grid_x * d
    grid_y = grid_y * d
    grid_x = x_mul * grid_x / grid_x.abs().sum() ** pow
    grid_y = y_mul * grid_y / grid_y.abs().sum() ** pow
    if swap:
        kernel = torch.stack([grid_x, grid_y], dim=0).reshape(2, 1, 3, 3)
    else:
        kernel = torch.stack([grid_y, grid_x], dim=0).reshape(2, 1, 3, 3)

    return kernel.to(device)


def min_max(input: torch.Tensor) -> torch.Tensor:
    return (input - input.min()) / (input.max() - input.min() + 1e-6
                                    )