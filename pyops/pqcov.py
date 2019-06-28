import torch
import torch.nn as nn
from pyops.pqlayer import dimension_split

class KmeansConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups, bias, ks):
        super(KmeansConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, ks, kernel_size, stride,
            padding, dilation, groups, bias)
        self.idx = torch.randint(low=0,
                                 high=ks,
                                 size=(out_channels, ),
                                 dtype=torch.long).cuda()

    def forward(self, x):
        # torch.Size([BatchSize, ks])
        x = self.conv(x)
        # torch.Size([BatchSize, d_output])
        x = torch.index_select(x, 1, self.idx)
        return x


class PQConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups, bias, m, ks):
        super(PQConvLayer, self).__init__()
        self.ds = dimension_split(in_channels, m)
        self.convs = nn.ModuleList([
            KmeansConv(self.ds[i+1] - self.ds[i],
                       out_channels, kernel_size, stride,
                       padding, dilation, groups, bias, ks)
            for i in range(m)])

    def forward(self, x):
        x = [conv(x[:, self.ds[i]: self.ds[i+1], :, :])
             for i, conv in enumerate(self.convs)]
        return torch.stack(x, dim=0).sum(dim=0)


class RQConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True, depth=1, m=1, ks=256):
        super(RQConvLayer, self).__init__()
        # ks = out_channels
        self.fcs = nn.ModuleList(
            [PQConvLayer(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias,
                         m, ks)
             for _ in range(depth)])

    def forward(self, x):
        x = [fc(x) for fc in self.fcs]
        return torch.stack(x, dim=0).sum(dim=0)
