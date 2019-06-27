import torch
import torch.nn as nn


class KmeansConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups, bias, ks):
        super(KmeansConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride,
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
                 stride, padding, dilation, groups, bias, ks):
        super(PQConvLayer, self).__init__()
        self.convs = nn.ModuleList([
            KmeansConv(1, out_channels, kernel_size, stride,
                       padding, dilation, groups, bias, ks)
            for _ in range(in_channels)])

    def forward(self, x):
        x = [conv(x[:, :, i])
             for i, conv in enumerate(self.convs)]
        return torch.stack(x, dim=0).sum(dim=0)


class RQConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True, depth=1, ks=4):
        super(RQConvLayer, self).__init__()
        self.fcs = nn.ModuleList(
            [PQConvLayer(in_channels, out_channels,
                        kernel_size, stride, bias, ks)
             for _ in range(depth)])

    def forward(self, x):
        x = [fc(x) for fc in self.fcs]
        return torch.stack(x, dim=0).sum(dim=0)
