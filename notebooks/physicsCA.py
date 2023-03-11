import numpy as np
from math import sqrt

from cppn import CPPN, Sampler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softmax


class Rule(nn.Module):
    def __init__(self, CHANNELS=2, RADIUS=1):
        super().__init__()

        # for make_kernel
        self.kpow = 0.2
        self.kernel = make_kernel(pow=self.kpow)

        # for cppn
        self.radius = RADIUS
        self.channels = CHANNELS

        ###########################################
        # init CPPN to generate kernels
        self.generate_cppn()
        ###########################################

        self.moves = make_moves()
        self.directions = make_directions()

    def generate_cppn(self):
        cppn_net_size = [32, 32, 32]
        dim_z = 16
        dim_c = self.channels
        self.cppn = CPPN(cppn_net_size, dim_z, dim_c).cuda().eval()
        self.sampler = Sampler()
        self.generate_cppn_kernel()

    def generate_cppn_kernel(self):
        scale = 5
        zscale = 2
        z = zscale * torch.randn(1, self.cppn.dim_z).cuda()
        Rk = self.radius * 2 + 1
        coords = self.cppn._coordinates(scale, Rk, Rk, z)
        # coords[0] = 10 + coords[2] * 2
        # coords[1] = 10 + coords[2] / 2
        # coords[2] = 10 + 5 * coords[2]

        x_orig = coords[0:1][0]
        y_orig = coords[1:2][0]
        r_orig = coords[2:3][0]
        falloff_coord = r_orig
        alpha = torch.rand(1).cuda()
        # falloff_coord = -alpha / (falloff_coord ** 3 + 1e-6) + (1 - alpha) / (falloff_coord + 1e-6)
        # falloff_coord = alpha / (falloff_coord ** 4 + 1e-6) - (1 - alpha) / (falloff_coord ** 2 + 1e-6)
        # falloff_coord = torch.cos(5 * falloff_coord) * torch.exp(-falloff_coord)
        # falloff_coord =

        coords[0] = x_orig
        coords[1] = x_orig
        coords[2] = x_orig


        with torch.no_grad():
            k = self.cppn.forward(coords, Rk, Rk).reshape(1, Rk, Rk, self.cppn.dim_c).permute(3, 0, 1, 2)

            # k = torch.stack([ck / (coords[2].reshape(Rk, Rk) + 1e-6) for ck in k], dim=0)
            # k[..., self.radius, self.radius] = 0.
            k = (k + k.flip(2, 3)) / 2

            #k = torch.stack([ck / (ck.norm() + 1e-6) for ck in k], dim=0)
            k = torch.stack([ck - ck.mean() for ck in k], dim=0)
            # k = torch.cat([k, -k.flip(2, 3)], dim=0)
            #k[..., self.radius, self.radius] = 0.
            # k = torch.stack([ck / (Rk) for ck in k], dim=0)
            parity_matrix = torch.sign(x_orig.reshape(1, 1, Rk, Rk)) * ((x_orig.reshape(1, 1, Rk, Rk).abs() > 1e-6))
            falloff_matrix = (falloff_coord).reshape(1, 1, Rk, Rk)
            #falloff_matrix *= (falloff_matrix > 1e-5) * 1.
            k = (k * falloff_matrix) * parity_matrix
            # k[..., self.radius, self.radius] = 0.

            k = torch.cat([k, k.permute(0, 1, 3, 2)], dim=0)

            self.kernel = k


    def forward(self, mass, momentum, force, A, dt=1., temp=1e-1, fdelta=0.1):
        # Update force
        #kernel_constant = self.kernel.abs().sum() ** -1.

        # KE = (momentum ** 2 / (mass + 1e-6)).sum()
        # PE = force.sum()
        # E = KE + PE

        force_norm = force.norm(p=2, dim=0, keepdim=True)
        #new_force = conv_pad(A * mass + force_norm, self.kernel, padding=self.radius).permute(1, 0, 2, 3)

        # new_force = conv_pad(A * mass + force_norm, self.kernel, padding=self.radius).permute(1, 0, 2, 3)
        new_force = conv_pad(A * mass, self.kernel, padding=self.radius).permute(1, 0, 2, 3)

        # field = conv_pad(A * mass + field, self.kernel, padding=self.radius).permute(1, 0, 2, 3)

        # new_force = torch.cat([new_force[:2].mean(dim=0, keepdim=True), new_force[2:].mean(dim=0, keepdim=True)])

        # new_force = torch.cat(
        #     [new_force[:self.channels].sum(dim=0, keepdim=True),
        #      new_force[self.channels:].sum(dim=0, keepdim=True)]
        # )
        new_force_sum = (new_force[:self.channels].abs() + new_force[self.channels:].abs())
        idx = torch.argsort(new_force_sum, dim=0)
        z_x = torch.gather(new_force, 0, idx)[[-1]]
        z_y = torch.gather(new_force, 0, idx + self.channels - 1)[[-1]]


        new_force = torch.cat([z_x, z_y])

        # new_force_momentum = conv_pad(A * momentum.permute(1, 0, 2, 3), self.kernel, padding=self.radius,
        #                               groups=2).permute(1, 0, 2, 3)
        # new_force_momentum = torch.cat(
        #     [new_force_momentum[:self.channels].sum(dim=0, keepdim=True),
        #      new_force_momentum[self.channels:].sum(dim=0, keepdim=True)]
        # )

        force_delta = new_force - force
        #force = force + force_delta * kernel_constant
        force = (1 - fdelta) * force + fdelta * force_delta

        # Update momentum
        # PE_new = force.sum()
        # KE_new = (momentum ** 2 / (mass + 1e-6)).sum()
        # E_new = PE_new + KE_new

        momentum = torch.where(mass < 1e-8, momentum.new_zeros(()), momentum + force * dt)

        # momentum *= E / E_new
        #momentum = torch.where(mass < 1e-8, momentum.new_zeros(()), momentum + (force + 0.01*new_force_momentum) * dt)
        velocity = torch.where(mass < 1e-8, momentum.new_zeros(()), momentum / mass)

        # Update mass
        propagation = torch.einsum('dn,nchw->cdhw', self.directions, velocity)
        propagation = softmax(propagation.mul(temp), dim=1)
        propagation = torch.where(propagation <= 0., propagation.new_zeros(()), propagation)

        mass_tot = mass.sum()
        mass_propagation = mass * propagation
        momentum_propagation = momentum * propagation

        mass = conv_pad(mass_propagation, self.moves)
        momentum = conv_pad(momentum_propagation, self.moves)
        mass = mass_tot * mass / mass.sum()

        return mass, momentum, force



class physicsCA(nn.Module):
    def __init__(self, CHANNELS, RADIUS):
        super().__init__()
        self.rule = Rule(CHANNELS=CHANNELS, RADIUS=RADIUS)

    def forward(self, mass, momentum, force, A, dt=1., temp=1e-1, fdelta=0.1):
        return self.rule(mass, momentum, force, A, dt, temp, fdelta)

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

def make_kernel(x_mul=1.0, y_mul=1.0, pow=0.2, swap=False, device="cuda"):
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


# def make_kernel(pow=None, device="cuda"):
    # return torch.tensor([[[-0.51608807, -0.3649294,  0.],
    #                       [-0.3649294,  0.,  0.3649294],
    #                       [0.,  0.3649294,  0.51608807]],
    #                      [[0., -0.3649294, -0.51608807],
    #                       [0.3649294,  0., -0.3649294],
    #                       [0.51608807,  0.3649294,  0.]]], dtype=torch.float32).to(device).unsqueeze(1)

    # return torch.tensor([[[-0.51608807, -0.3649294, 0.],
    #                       [-0.3649294, 0., 0.3649294],
    #                       [0., 0.3649294, 0.51608807]],
    #                      [[-0., 0.3649294, 0.51608807],
    #                       [-0.3649294, 0., 0.3649294],
    #                       [-0.51608807, -0.3649294, 0.]]], dtype=torch.float32).to(device).unsqueeze(1)

    # return torch.tensor([[[-0.3649294, .51608807, -0.3649294],
    #            [0., 0., 0.],
    #            [0.3649294, -.51608807, 0.3649294]],
    #           [[0.3649294, 0., -0.3649294],
    #            [-0.51608807, 0., 0.51608807],
    #            [0.3649294, 0., -0.3649294]]], dtype=torch.float32).to(device).unsqueeze(1)

    # return torch.tensor([[[-0.3649294, .51608807, -0.3649294],
    #                       [0., 0., 0.],
    #                       [0.3649294, -.51608807, 0.3649294]],
    #                      [[-0.51608807, -0.3649294,  0.],
    #                       [-0.3649294,  0.,  0.3649294],
    #                       [0.,  0.3649294,  0.51608807]]], dtype=torch.float32).to(device).unsqueeze(1)

    # return torch.tensor([[[.51, .51, .51],
    #                       [0., 0., 0.],
    #                       [0., 0., 0.]],
    #                      [[-0.51608807, -0.3649294, 0.],
    #                       [-0.3649294,  0.,  0.3649294],
    #                       [0.,  0.3649294,  0.51608807]],
    #                      [[0., 0, 0.3649294],
    #                       [0., 0., 0.51608807],
    #                       [0., 0., 0.3649294]],
    #                      [[0.,  0.3649294,  0.51608807],
    #                       [-0.3649294,  0.,  0.3649294],
    #                       [-0.51608807, -0.3649294, 0.]]], dtype=torch.float32).to(device).unsqueeze(1)

    # return torch.tensor([[[0., 0.3649294, 0.],
    #                           [0.3649294, -0.51608807, 0.3649294],
    #                           [0., 0.3649294, 0.]],
    #                          [[-0.51608807, -0.3649294, 0.],
    #                           [-0.3649294,  0.,  0.3649294],
    #                           [0.,  0.3649294,  0.51608807]],
    #                          [[0., -0.3649294, 0.],
    #                           [-0.3649294, 0.51608807, -0.3649294],
    #                           [0., -0.3649294, 0.]],
    #                          [[0.,  0.3649294,  0.51608807],
    #                           [-0.3649294,  0.,  0.3649294],
    #                           [-0.51608807, -0.3649294, 0.]]], dtype=torch.float32).to(device).unsqueeze(1)


def min_max(input: torch.Tensor) -> torch.Tensor:
    return (input - input.min()) / (input.max() - input.min() + 1e-6
                                    )