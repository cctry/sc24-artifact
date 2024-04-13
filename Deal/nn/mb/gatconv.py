import time
import dgl.function as fn
import torch
import torch.distributed as dist
from torch import nn

from ..ops import dist_mm, indexed_add, to_dist_linear

from dgl.utils import expand_as_pair
from dgl.nn.functional import edge_softmax

import dgl


class DistGATConv(dgl.nn.GATConv):      
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc = to_dist_linear(self.fc)
        if self.res_fc is not None:
            self.res_fc = to_dist_linear(self.res_fc)


    def forward(self, graph, local_feat, dataloader, part_size, group=None, layer=None):
        local_num_head = self._num_heads // dist.get_world_size(group)
        assert self._num_heads % dist.get_world_size(group) == 0, "GAT needs head parallelism"
        prefix_shape = local_feat.shape[:-1]
        h = self.feat_drop(local_feat)

        feat = self.fc(h, part_size, group)
        feat = feat.view(
            *prefix_shape, local_num_head, self._out_feats
        )
        assert feat.ndim == 3, "GATConv requires 2D input"

        local_head_start = dist.get_rank(group) * local_num_head
        local_head_end = local_head_start + local_num_head
        attn_l = self.attn_l[:, local_head_start:local_head_end, :]
        attn_r = self.attn_r[:, local_head_start:local_head_end, :]
        attn_lr = torch.vstack([attn_l, attn_r])
        e_lr = torch.einsum("nhd,ehd->neh", feat, attn_lr)
        
        dataloader.set_local_feat(e_lr)
        dataloader.set_hop(layer)

        efeat_shape = [graph.num_edges(), local_num_head, 1]
        efeat = torch.empty(efeat_shape, dtype=local_feat.dtype, device=local_feat.device)
        
        # SDDMM
        for block, comm in dataloader:
            e_lr_ = comm.sync().unsqueeze(-1)
            el, er = e_lr_[:, 0, :, :], e_lr_[:, 1, :, :]
            block.ndata.update({"el": el, "er": er})
            block.apply_edges(fn.u_add_v("el", "er", "e"))
            block.ndata.pop("el")
            block.ndata.pop("er")
            e = self.leaky_relu(block.edata.pop("e"))
            efeat[block.edata[dgl.EID]] = e
        
        e_lr = None
        efeat = edge_softmax(graph, efeat)
        efeat = self.attn_drop(efeat)

        rst_shape = (feat.shape[0], feat.shape[1] * feat.shape[2])
        rst = torch.empty(rst_shape, dtype=feat.dtype, device=feat.device)
        # SPMM
        dataloader.set_local_feat(feat)
        for block, comm in dataloader:
            block.edata["a"] = efeat[block.edata[dgl.EID]]
            block.ndata['ft'] = comm.sync()
            block.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
            block.edata.pop("a")
            ft = block.ndata.pop("ft")
            indexed_add(rst, comm.rst_dst, ft.view(ft.shape[0], -1), comm.rst_src)
            
        new_part_size = self.fc.new_part_size
        rst = rst.view(feat.shape)
        if isinstance(self.res_fc, nn.Linear):
            rst_flat = rst.view(-1, self.res_fc.in_features)
            resval = self.res_fc(rst_flat, new_part_size, group)
            rst = rst + resval.view(rst.shape)
            new_part_size = self.res_fc.new_part_size
        
        if self.activation:
            rst = self.activation(rst)

        return rst, new_part_size


