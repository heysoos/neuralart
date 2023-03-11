import numpy as np
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


HNM_STEPS = 3000  # Max number of steps to mine for a hard-negative sample before while-loop terminates


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
    if dim2:
        z = z - z.mean()
    else:
        z = z - z.mean(x_idx).mean(y_idx).unsqueeze(y_idx).unsqueeze(x_idx)
        # z = z - z.mean(x_idx).mean(y_idx).unsqueeze(y_idx).unsqueeze(x_idx) + torch.randn(z.shape[0], z.shape[1], 1, 1).cuda() * 0.001
        #z = z - z.mean(x_idx).mean(y_idx).unsqueeze(y_idx).unsqueeze(x_idx) + torch.randn(z.shape[0], z.shape[1], 1, 1).cuda() * 0.001

        # z = z - z.mean()

    return z


class Rule(nn.Module):
    def __init__(self, CHANNELS=8, FILTERS=1, HIDDEN=16, RADIUS=2, KCENTER=1, KSMOOTH=1, OUTR=1, INR=0, GAMP=0.1):
        super().__init__()
        self.channels = CHANNELS
        self.filters = FILTERS
        self.hidden = HIDDEN
        self.ksmooth = KSMOOTH  # kernel variance
        self.kcenter = KCENTER  # kernel center
        self.outr = OUTR  # outer radius
        self.inr = INR  # inner radius
        self.gamp = GAMP  # guassian kernel amplitude

        # toggle options
        self.totalistic = True
        self.use_growth_kernel = False

        Rk = RADIUS * 2 + 1
        ############################
        # mask/decay kernel
        # a gaussian curve used to radially decay the strength of the kernel
        # a inner/outer radial ring to mask the kernel
        decay_kernel = self.make_decay_kernel(Rk, KCENTER, KSMOOTH, OUTR, INR)
        # decay_kernel = decay_kernel - decay_kernel.mean()
        self.decay_kernel = decay_kernel
        GR = RADIUS
        GRK = GR*2 + 1
        growth_kernel = self.make_growth_kernel(Rk=GRK, KCENTER=KCENTER, KSMOOTH=KSMOOTH, OUTR=OUTR, INR=INR, GAMP=GAMP)
        # growth_kernel = growth_kernel - growth_kernel.mean()
        # growth_kernel = growth_kernel / growth_kernel.norm()
        # growth_kernel[GR, GR] = 1.
        self.growth_kernel = growth_kernel
        ############################

        ###########################################
        # for forward (general version)
        lower, upper = -(np.sqrt(6.) / np.sqrt(CHANNELS + FILTERS)), (np.sqrt(6.) / np.sqrt(CHANNELS + FILTERS))
        # kernel_weights = torch.randn(FILTERS, CHANNELS, Rk, Rk) / sqrt(Rk)
        kernel_weights = torch.rand(FILTERS, CHANNELS, Rk, Rk).cuda()
        kernel_weights = lower + kernel_weights * (upper - lower)
        kernel_weights = totalistic(kernel_weights * self.decay_kernel)

        kernel_weights /= kernel_weights.abs().sum(dim=(2, 3), keepdim=True)
        #kernel_weights /= kernel_weights.abs().sum(dim=(0), keepdim=True)
        #kernel_weights /= kernel_weights.abs().sum(dim=(1), keepdim=True)
        # kernel_weights = totalistic(kernel_weights * self.decay_kernel)

        # for iF in range(FILTERS):
        #     for iC in range(CHANNELS):
        #         # kernel_weights[iF, iC] -= kernel_weights[iF, iC].mean()
        #         kernel_weights[iF, iC] *= 1 + 0.001 * torch.randn(1).cuda()

        self.filter1 = nn.Parameter(kernel_weights)
        # self.filter1 = nn.Parameter(torch.randn(FILTERS, 4 * CHANNELS, Rk, Rk))
        # self.filter1 = nn.Parameter(torch.randn(1, 4 * CHANNELS * FILTERS, Rk, Rk))
        self.bias1 = nn.Parameter(0 * torch.randn(FILTERS))

        self.filter2 = nn.Conv2d(FILTERS, HIDDEN, 1, padding_mode='circular')
        #nn.init.orthogonal_(self.filter2.weight)
        nn.init.xavier_normal_(self.filter2.weight)
        # nn.init.dirac_(self.filter2.weight)
        # nn.init.normal_(self.filter2.weight)
        nn.init.zeros_(self.filter2.bias)
        # nn.init.zeros_(self.filter2.weight)
        # self.filter2.weight.data.zero_()
        # self.filter2.bias = nn.Parameter(self.filter2.bias * 0.1)

        # self.extra_filters = []
        # for i in range(10):
        #     self.extra_filters.append(nn.Conv2d(HIDDEN, HIDDEN, 1, padding_mode='circular', bias=False))
        #     nn.init.orthogonal_(self.extra_filters[-1].weight)
        #     # self.extra_filters.append(nn.ReLU())
        #     # self.extra_filters.append(nn.Tanh())
        #
        # self.extra_filters = nn.Sequential(*self.extra_filters)

        self.filter3 = nn.Conv2d(HIDDEN, CHANNELS, 1, padding_mode='circular', bias=False)
        #nn.init.orthogonal_(self.filter3.weight)
        nn.init.xavier_normal_(self.filter3.weight)
        # nn.init.dirac_(self.filter3.weight)
        # nn.init.normal_(self.filter3.weight)

        # nn.init.zeros_(self.filter3.bias)
        # nn.init.zeros_(self.filter3.weight)
        # self.filter3.weight.data.zero_()

        # activation functions
        thresh1 = np.random.random()*2. - 1
        thresh2 = thresh1 + (np.random.random() + 0.1)
        self.thresh1 = [thresh1, thresh2]

        thresh1 = np.random.random() * 2. - 1
        thresh2 = thresh1 + (np.random.random() + 0.1)
        self.thresh2 = [thresh1, thresh2]

        ###########################################

        ###########################################
        # for forward_perception
        self.ident = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]).cuda()
        self.sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).cuda() / 8.0
        self.lap = torch.tensor([[1.0, 2.0, 1.0], [2.0, -12, 2.0], [1.0, 2.0, 1.0]]).cuda() / 16.0

        self.filters = [nn.Parameter(torch.randn(3, 3).cuda())
                        for i in range(2)]

        self.w1 = torch.nn.Conv2d(CHANNELS * 4, HIDDEN, 1)
        self.w1.bias.data.zero_()
        self.w2 = torch.nn.Conv2d(HIDDEN, CHANNELS, 1, bias=False)
        # self.w2.weight.data.zero_()
        ###########################################

    def make_decay_kernel(self, Rk, KCENTER, KSMOOTH, OUTR, INR):
        # make a kernel of size Rk that decays like a gaussian and has a inner/outer cutoff radius
        xm, ym = torch.meshgrid(torch.linspace(-1, 1, Rk), torch.linspace(-1, 1, Rk))
        rm = torch.sqrt(xm ** 2 + ym ** 2).type(torch.double)
        decay = torch.exp(-(rm - KCENTER) ** 2 / (KSMOOTH ** 2 + 1e-6))
        decay = torch.where(rm <= OUTR, decay, 0.)
        decay = torch.where(rm >= INR, decay, 0.)
        decay = decay / decay.max()
        decay = decay.type(torch.cuda.FloatTensor)
        return decay

    def make_growth_kernel(self, Rk, KCENTER, KSMOOTH, OUTR, INR, GAMP):
        # make a kernel of size Rk that decays like a gaussian and has a inner/outer cutoff radius
        xm, ym = torch.meshgrid(torch.linspace(-1, 1, Rk), torch.linspace(-1, 1, Rk))
        rm = torch.sqrt(xm ** 2 + ym ** 2).type(torch.double)
        decay = torch.exp(-(rm - KCENTER) ** 2 / (KSMOOTH ** 2 + 1e-6))
        decay = torch.where(rm <= OUTR, decay, 0.)
        decay = torch.where(rm >= INR, decay, 0.)
        decay = decay / decay.sum() * GAMP
        # decay = decay - decay.mean()
        decay = decay.type(torch.cuda.FloatTensor)
        return decay

class CA(nn.Module):
    def __init__(self, CHANNELS=8, FILTERS=1, HIDDEN=16, RADIUS=2, KCENTER=0, KSMOOTH=1, OUTR=1, INR=0, GAMP=0.1):
        super().__init__()
        self.channels = CHANNELS
        self.filters = FILTERS
        self.hidden = HIDDEN
        self.radius = RADIUS


        self.rule = Rule(CHANNELS, FILTERS, HIDDEN, RADIUS, KCENTER, KSMOOTH, OUTR, INR, GAMP=GAMP)
        self.optim = torch.optim.Adam(self.parameters(), lr=1e-3)

    def initGrid(self, BS, RES):
        self.psi = torch.cuda.FloatTensor(2 * np.random.rand(BS, self.channels, RES, RES) - 1)

    def seed(self, RES, n):
        seed = torch.FloatTensor(np.zeros((n, self.channels, RES, RES)))
        # seed[:, 3:, RES // 2, RES // 2] = 1
        return seed

    def perchannel_conv(self, x, filters):
        '''filters: [filter_n, h, w]'''
        b, ch, h, w = x.shape
        y = x.reshape(b * ch, 1, h, w)
        y = torch.nn.functional.pad(y, [1, 1, 1, 1], 'circular')
        y = torch.nn.functional.conv2d(y, filters[:, None])
        return y.reshape(b, -1, h, w)

    def perchannel_conv_g(self, x, filters):
        '''filters: [filter_n, h, w]'''
        b, ch, h, w = x.shape
        Rx = int((filters.shape[-2] - 1)/2)
        Ry = int((filters.shape[-1] - 1)/2)
        y = x.reshape(b * ch, 1, h, w)
        y = torch.nn.functional.pad(y, [Rx, Rx, Ry, Ry], 'circular')
        y = torch.nn.functional.conv2d(y, filters[:, None])
        return y.reshape(b, -1, h, w)

    def perception(self, x):
        # filters = torch.stack([self.rule.ident, self.rule.sobel_x, self.rule.sobel_x.T, self.rule.lap])
        filters = [self.rule.ident, self.rule.sobel_x, self.rule.sobel_x.T, self.rule.lap]
        # custom kernels required to be the same size as the hard-coded filters for now to work
        # totalistic_filters = [totalistic(f, dim2=True) for f in self.rule.filters]
        # filters = torch.stack(filters + totalistic_filters)
        return self.perchannel_conv(x, torch.stack(filters))

    def get_living_mask(self, x, alive_thres=0, dead_thres=0.6):
        alpha_channel = x[:, 3:4, :, :]
        R = self.radius
        alpha_channel = F.pad(alpha_channel, (R, R, R, R), mode='circular')

        alive_mask = F.max_pool2d(alpha_channel, kernel_size=2 * R + 1, stride=1, padding=R) > alive_thres
        alive_mask = alive_mask[:, :, R:-R, R:-R]

        # death_mask = F.avg_pool2d(alpha_channel, kernel_size=2 * R + 1, stride=1, padding=R) < dead_thres
        # death_mask = death_mask[:, :, R:-R, R:-R]
        # return alive_mask & death_mask

        return alive_mask

    def forward(self, x, update_rate=1):
        '''
        The first filter applies a depthwise convolution to the CA grid. Each channel in the filter is applied to its corresponding channel in the CA grid.
        The second and third filters are 1x1 convolutions which act to mix the channels.
        If I understand this correctly, this is essentially applying a depthwise seperable convolution operation on the input (but I am a bit uncertain).
        PRETTY SURE THIS IS WRONG, RE-WRITE THIS!!
        '''
        # b, c, h, w = x.shape
        # circular/gaussian kernel mask
        # filter1 = totalistic(self.rule.filter1 * self.rule.decay_kernel)
        #
        # if self.rule.totalistic:
        #     filter1 = totalistic(filter1)
        weights = self.rule.filter1
        # weights = self.rule.filter1
        bias = self.rule.bias1
        R = self.radius
        # z = F.conv2d(self.psi, weight=weights, bias=bias, padding=2, groups=CHANNELS)
        z = F.pad(x, (R, R, R, R), 'circular')
        # z = F.pad(x, (R, R, R, R), 'constant', 0.)
        z = F.conv2d(z, weight=weights, bias=bias, padding=0)

        # selection_idx = torch.argmax(z.mean(dim=(2, 3)), dim=1)
        # z = z[:, [selection_idx], :, :].contiguous()



        z = F.leaky_relu(z)
        # z = ((z > self.rule.thresh1[0]) & (z < self.rule.thresh1[1])) * z

        if self.rule.use_growth_kernel:
            z = self.perchannel_conv_g(z, self.rule.growth_kernel.unsqueeze(0) )

        z = self.rule.filter2(z)
        z = F.leaky_relu(z)
        # z = ((z > self.rule.thresh2[0]) & (z < self.rule.thresh2[1])) * z

        # select max/min channel per pixel
        # selection_idx = torch.argmax(z, dim=1, keepdim=True)
        # z = torch.gather(z, 1, selection_idx)


        # random selection
        # z = torch.gather(z, 1, torch.randint(z.shape[1], size=(b, 1, h, w)).cuda())

        # z = self.rule.extra_filters(z)

        # select from top/bottom sorted channels
        # z = torch.gather(z, 1, torch.argsort(z, dim=1))[:, -12:, :, :]

        # update_mask = (torch.rand(b, 1, h, w) + update_rate).floor().cuda()
        # z = self.rule.filter3(z) * update_mask

        z = self.rule.filter3(z) * update_rate
        z = x + z
        # x = torch.clamp(z, 0, 255)
        x = torch.clamp(z, 0, 1)
        return x


    def forward_masked(self, x, dt=1):

        pre_life_mask = self.get_living_mask(x)
        # circular/gaussian kernel mask
        filter1 = self.rule.filter1 * self.rule.decay_kernel
        if self.rule.totalistic:
            filter1 = totalistic(filter1)
        weights = filter1
        bias = self.rule.bias1
        R = self.radius

        z = F.pad(x, (R, R, R, R), 'circular')
        z = F.conv2d(z, weight=weights, bias=bias, padding=0)

        z = F.leaky_relu(z)
        z = F.leaky_relu(self.rule.filter2(z))

        z = torch.gather(z, 1, torch.argsort(z, dim=1))[:, -12:, :, :]

        # self.psi = torch.tanh(self.psi[:, :, R:-R, R:-R] + dt*self.rule.filter3(z))
        z = self.rule.filter3(z)
        if self.rule.use_growth_kernel:
            z = self.perchannel_conv_g(z, self.rule.growth_kernel.unsqueeze(0))
        z = torch.clamp(x + dt*z, -127, 128)
        x = z * (pre_life_mask).type(torch.cuda.FloatTensor)


        # post_living_mask = self.get_living_mask(x)

        # x = z * (pre_life_mask).type(torch.cuda.FloatTensor)
        # x = z * (pre_life_mask & post_living_mask).type(torch.cuda.FloatTensor)

        return x

    # def forward_perception(self, dt=1, update_rate=0.5):
        # b, ch, h, w = self.psi.shape
        # pre_life_mask = self.get_living_mask(self.psi)
        # weights = totalistic(self.rule.filter1)
        # # weights = self.rule.filter1
        # bias = self.rule.bias1
        # R = self.radius
        #
        # y = self.perception(self.psi)
        #
        # y = F.pad(y, (R, R, R, R), 'circular')
        # y = F.conv2d(y, weight=weights, bias=bias, padding=0)
        #
        # y = F.leaky_relu(y)
        # y = F.leaky_relu(self.rule.filter2(y))
        #
        # update_mask = (torch.rand(b, 1, h, w)+update_rate).floor().cuda()
        # y = dt * self.rule.filter3(y) * update_mask
        # # self.psi = torch.clamp(self.psi + y, 0, 1)
        # self.psi = self.psi + y
        #
        # # post_living_mask = self.get_living_mask(self.psi)
        # #
        # # self.psi = self.psi * (pre_life_mask & post_living_mask).type(torch.cuda.FloatTensor)

    def forward_perception(self, x, dt=1, update_rate=0.5):
        b, ch, h, w = x.shape
        y = self.perception(x)

        y = torch.relu(self.rule.w1(y))
        y = self.rule.w2(y)

        update_mask = (torch.rand(b, 1, h, w) + update_rate).floor().cuda()
        # print(update_mask.shape, y.shape)
        y = dt * y * update_mask
        # res = torch.clamp(x + y, 0, 1)
        res = torch.tanh(x + y)
        return res



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