import torch
import torch.nn as nn
import numpy as np


def dimension_split(d, m):
    reminder = d % m
    quotient = int(d / m)
    dims_width = [quotient + 1 if i < reminder
                  else quotient for i in range(m)]
    ds = np.cumsum(dims_width)     # prefix sum
    ds = np.insert(ds, 0, 0)  # insert zero at beginning
    return ds


def product_add(x, m, ks):
    """
    :param x:  list of tensors with shape of [N, D] or [N, C, H, W]
    :param m:
    :param ks:
    :return: shape of [N, ks**m] or [N, ks**M, H, W]
    """
    N = x[0].shape[0]
    HW = list(x[0].shape[2:])

    s = [N] + [-1] + [1 for _ in range(m-1)] + HW

    sum_ = x[0].new_zeros([N] + [ks for _ in range(m)] + HW)
    for i, x_i in enumerate(x):
        x_i = x_i.reshape(s)
        if i == 0:
            sum_ += x_i
        else:
            view_ = sum_.transpose(1, i+1)
            view_ += x_i
    return sum_.view([N] + [ks**m] + HW)


class PQLinear(nn.Module):
    def __init__(self, d_input, d_output, bias=True, m=2, ks=32):
        super(PQLinear, self).__init__()
        assert ks ** m == d_output
        assert d_output**m == d_output
        self.ds = dimension_split(d_input, m)
        self.m = m
        self.ks = ks
        self.fcs = nn.ModuleList(
            [nn.Linear(self.ds[i+1] - self.ds[i], ks, bias=bias)
             for i in range(m)]
        )

    def forward(self, x):
        x = [fc(x[:, self.ds[i]: self.ds[i+1]])
             for i, fc in enumerate(self.fcs)]
        return product_add(x, self.m, self.ks)


class AQLinear(nn.Module):
    def __init__(self, d_input, d_output, bias=True, m=1, depth=2, ks=32):
        super(AQLinear, self).__init__()
        assert ks ** (m * depth) == d_output
        self.depth = depth
        self.ks = ks
        self.fcs = nn.ModuleList(
            [PQLinear(d_input, ks**m, bias, m, ks)
             for _ in range(depth)]
        )
        self.shapes = [-1] + [1 for _ in range(depth - 1)]

    def forward(self, x):
        x = [fc(x) for fc in self.fcs]
        return product_add(x, self.depth, self.ks)


class PQConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups, bias, m, ks):
        super(PQConv2d, self).__init__()
        assert out_channels == ks ** m
        self.ds = dimension_split(in_channels, m)
        self.m = m
        self.ks = ks
        self.convs = nn.ModuleList([
            nn.Conv2d(self.ds[i+1] - self.ds[i],
                       ks, kernel_size, stride,
                       padding, dilation, groups, bias)
            for i in range(m)])

    def forward(self, x):
        x = [conv(x[:, self.ds[i]: self.ds[i+1], :, :])
             for i, conv in enumerate(self.convs)]
        return product_add(x, self.m, self.ks)


class AQConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True, depth=2, m=1, ks=32):
        super(AQConv2d, self).__init__()
        assert out_channels == ks ** (m*depth)
        self.depth = depth
        self.ks = ks
        self.fcs = nn.ModuleList(
            [PQConv2d(in_channels, ks**m, kernel_size,
                         stride, padding, dilation, groups, bias,
                         m, ks)
             for _ in range(depth)])

    def forward(self, x):
        x = [fc(x) for fc in self.fcs]
        return product_add(x, self.depth, self.ks)


if __name__ == '__main__':
    ks = 2
    m = 4
    x = torch.Tensor([[1.0, 3.0]])
    x = [x for _ in range(m)]
    print(torch.sort(product_add(x, m, ks)))
