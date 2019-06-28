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


class KmeansDenseLayer(nn.Module):
    def __init__(self, d_input, d_output, ks):
        super(KmeansDenseLayer, self).__init__()
        self.fcs = nn.Linear(d_input, ks)
        self.idx = torch.randint(low=0,
                                 high=ks,
                                 size=(d_output, ),
                                 dtype=torch.long).cuda()

    def forward(self, x):
        # torch.Size([BatchSize, ks])
        x = self.fcs(x)
        # torch.Size([BatchSize, d_output])
        x = torch.index_select(x, 1, self.idx)
        return x


class PQDenseLayer(nn.Module):
    def __init__(self, d_input, d_output, m, ks):
        super(PQDenseLayer, self).__init__()
        self.ds = dimension_split(d_input, m)
        self.fcs = nn.ModuleList([
            KmeansDenseLayer(self.ds[i+1] - self.ds[i],
                             d_output, ks)
            for i in range(m)])

    def forward(self, x):
        x = [fc(x[:, self.ds[i]: self.ds[i+1]])
             for i, fc in enumerate(self.fcs)]
        return torch.stack(x, dim=0).sum(dim=0)


class RQDenseLayer(nn.Module):
    def __init__(self, d_input, d_output, m=1, depth=2, ks=1024):
        super(RQDenseLayer, self).__init__()
        self.fcs = nn.ModuleList(
            [PQDenseLayer(d_input, d_output, m, ks)
             for _ in range(depth)])

    def forward(self, x):
        x = [fc(x) for fc in self.fcs]
        return torch.stack(x, dim=0).sum(dim=0)
