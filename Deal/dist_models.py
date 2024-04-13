import torch
from torch import nn
from torch.nn import functional as F
from . import nn as dist_nn


class DistGNN(nn.Module):
    def __init__(self, n_layer, in_dim, hid_dim, out_dim) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.n_layer = n_layer
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.in_dims = [in_dim] + [hid_dim] * (n_layer - 1)
        self.out_dims = [hid_dim] * (n_layer - 1) + [out_dim]
        self.device = torch.device("cpu")

    def infer(self, g, x, layer, comm, part_size=None, group=None):
        raise NotImplementedError

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def to(self, *args, **kwargs):
        if isinstance(args[0], torch.device):
            self.device = args[0]
        return super().to(*args, **kwargs)


class DistGCN(DistGNN):
    def __init__(
        self,
        n_layer,
        in_dim,
        hid_dim,
        out_dim,
        norm="both",
        weight=True,
        bias=True,
        activation=F.relu,
        mini_batch=False,
    ):
        super().__init__(n_layer, in_dim, hid_dim, out_dim)
        layer = dist_nn.DistGraphConvMB if mini_batch else dist_nn.DistGraphConv
        self.layers = nn.ModuleList()
        for in_d, out_d in zip(self.in_dims, self.out_dims):
            self.layers.append(
                layer(
                    in_d,
                    out_d,
                    norm=norm,
                    weight=weight,
                    bias=bias,
                    activation=activation,
                )
            )

    def infer(self, g, x, layer, comm, part_size=None, group=None):
        return self.layers[layer](g, x, comm, part_size, group=group, layer=layer)


class DistGAT(DistGNN):
    def __init__(
        self,
        n_layer,
        in_dim,
        hid_dim,
        out_dim,
        heads,
        mini_batch=False,
    ):
        super().__init__(n_layer, in_dim, hid_dim, out_dim)
        assert mini_batch, "GAT only supports mini-batch training"
        layer = dist_nn.DistGATConvMB
        self.layers = nn.ModuleList()
        for i, (in_d, out_d) in enumerate(zip(self.in_dims, self.out_dims)):
            self.layers.append(
                layer(
                    in_d,
                    out_d // heads[i] if i < n_layer - 1 else out_d,
                    heads[i],
                    feat_drop=0.6,
                    attn_drop=0.6,
                    activation=F.elu if i < n_layer - 1 else None,
                )
            )

    def infer(self, g, x, layer, comm, part_size=None, group=None):
        rst, new_part_size = self.layers[layer](g, x, comm, part_size, group=group, layer=layer)
        if layer == self.n_layer - 1:
            rst = rst.mean(1)
        else:
            rst = rst.flatten(1)
        return rst, new_part_size
