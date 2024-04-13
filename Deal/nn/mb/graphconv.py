import gc
import time
import dgl.function as fn
import torch
import torch.distributed as dist
from torch import nn

from ..ops import dist_mm, indexed_add
import dgl

import psutil
def get_mem():
    return f'RAM memory % used: {psutil.virtual_memory()[2]} {psutil.virtual_memory()[3] >> 20}'
from memory_profiler import profile   


class DistGraphConv(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        norm="both",
        weight=True,
        bias=True,
        activation=None,
    ):
        super().__init__()
        assert norm in ("none", "both", "right", "left"), "Invalid norm value"
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._activation = activation

        if weight:
            self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        else:
            self.register_parameter("weight", None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


    def forward(
        self,
        graph,
        local_feat,
        dataloader,
        part_size,
        weight=None,
        # edge_weight=None,
        group=None,
        layer=None,
    ):
        with graph.local_scope():
            inner_nodes = torch.where(graph.ndata["inner_node"])[0]
            aggregate_fn = fn.copy_u("h", "m")

            if self._norm in ["left", "both"]:
                assert graph.include_out_edge, "Partition has no out edges"
                degs = graph.out_degrees(inner_nodes).to(local_feat).clamp(min=1)
                if self._norm == "both":
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (local_feat.dim() - 1)
                norm = torch.reshape(norm, shp)
                local_feat = local_feat * norm

            if weight:
                assert self.weight is None, "External weight is provided"
            else:
                weight = self.weight
            
            rst = torch.empty_like(local_feat)
            # aggregate first. The aggregtaion is in batch
            dataloader.set_local_feat(local_feat)
            dataloader.set_hop(layer)
            for block, comm in dataloader:
                block.ndata["h"] = comm.sync()      
                block.update_all(aggregate_fn, fn.sum(msg="m", out="h"))
                h = block.ndata.pop("h")
                indexed_add(rst, comm.rst_dst, h, comm.rst_src)

            if weight is not None:
                rst, new_part_size = dist_mm(rst, weight, part_size, group)

            if self._norm in ["right", "both"]:
                degs = graph.in_degrees(inner_nodes).to(local_feat).clamp(min=1)
                if self._norm == "both":
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (local_feat.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rank = dist.get_rank(group)
                begin = new_part_size[:rank].sum()
                end = begin + new_part_size[rank]
                rst = rst + self.bias[begin:end]

            if self._activation is not None:
                rst = self._activation(rst)

            return rst, new_part_size
