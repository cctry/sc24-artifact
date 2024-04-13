import dgl.function as fn
import torch
from torch import nn
from .ops import dist_mm


class DistSAGEConv(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        aggregator_type,
        feat_drop=0.0,
        bias=True,
        norm=None,
        activation=None,
    ):
        super().__init__()
        valid_aggre_types = {"mean", "gcn", "pool", "lstm"}
        assert aggregator_type in valid_aggre_types, "Invalid aggregator type"

        self.in_feats = in_feats
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation

        # aggregator type: mean/pool/lstm/gcn
        if aggregator_type == "pool":
            self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        if aggregator_type == "lstm":
            self.lstm = nn.LSTM(
                self._in_src_feats, self._in_src_feats, batch_first=True
            )

        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)

        if aggregator_type != "gcn":
            self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        elif bias:
            self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        else:
            self.register_buffer("bias", None)

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
        comm,
        part_size,
        weight=None,
        edge_weight=None,
        group=None,
    ):
        with graph.local_scope():
            inner_nodes = torch.where(graph.ndata["inner_node"])[0]
            aggregate_fn = fn.copy_u("h", "m")
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.num_edges()
                graph.edata["_edge_weight"] = edge_weight
                aggregate_fn = fn.u_mul_e("h", "_edge_weight", "m")

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

            if self._in_feats > self._out_feats:
                # mult W first to reduce the feature size for aggregation.
                if weight is not None:
                    local_feat, new_part_size = dist_mm(
                        local_feat, weight, part_size, group
                    )
                graph.srcdata["h"] = comm.sync_tensor(local_feat)
                graph.update_all(aggregate_fn, fn.sum(msg="m", out="h"))
                rst = graph.dstdata["h"]
                rst = rst[graph.ndata["inner_node"]]
            else:
                # aggregate first then mult W
                graph.srcdata["h"] = comm.sync_tensor(local_feat)
                graph.update_all(aggregate_fn, fn.sum(msg="m", out="h"))
                rst = graph.dstdata["h"]
                rst = rst[graph.ndata["inner_node"]]
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
                offset = torch.cumsum(
                    torch.cat([torch.zeros(1, dtype=torch.long), part_size]), dim=0
                )
                rst = rst + self.bias[offset[comm.rank] : offset[comm.rank + 1]]

            if self._activation is not None:
                rst = self._activation(rst)

            return rst, new_part_size
