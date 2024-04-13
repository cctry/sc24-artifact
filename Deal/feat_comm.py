from collections import defaultdict
from typing import Optional

import dgl
import torch
import torch.distributed as dist

from . import utils


class FeatCommunicator:
    def __init__(self, g: dgl.DGLGraph, group: Optional[dist.ProcessGroup] = None):
        self.group = group
        self.g = g
        self.world_size = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.part2idx = [
            torch.where(g.ndata["part_id"] == i)[0] for i in range(self.world_size)
        ]
        assert (
            g.ndata["part_id"].max() == self.world_size - 1
        ), "part_id cannot match world_size"
        # required global ID from each partition
        part2gid = [g.ndata[dgl.NID][t] for t in self.part2idx]
        # required global ID by each partition
        part2gid, future = utils.all_to_all(part2gid, is_async=True, group=group)
        sorted_nid, indices = torch.sort(g.ndata[dgl.NID])

        inner_nodes = torch.where(g.ndata["inner_node"])[0]
        self.part2size = defaultdict(lambda: None)
        self.local_nids = [None] * self.world_size
        self.local_indices = [None] * self.world_size
        for i in future.as_completed():
            self.local_nids[i] = indices[torch.searchsorted(sorted_nid, part2gid[i])]
            self.local_indices[i] = torch.searchsorted(inner_nodes, self.local_nids[i])
            part2gid[i] = None

    def sync(self, feat_name: str):
        """sync and populate nodes features

        Parameters
        ----------
        feat_name : str
            The name of the feature to be synchronized.
        """
        if self.world_size == 1:
            return
        send_tensors = [self.g.ndata[feat_name][idx] for idx in self.local_nids]
        if self.part2size[feat_name] is None:
            self.part2size[feat_name] = utils.compute_part2size(
                send_tensors, self.group
            )
        recv_tensors, future = utils.all_to_all(
            send_tensors, self.part2size[feat_name], True, self.group
        )
        for i in future.as_completed():
            if i != self.rank:
                self.g.ndata[feat_name][self.part2idx[i]] = recv_tensors[i]
            recv_tensors[i] = None

    def sync_tensor(self, tensor: torch.Tensor, part_size=None) -> torch.Tensor:
        if dist.get_backend(self.group) == dist.Backend.NCCL:
            assert tensor.is_cuda, "NCCL backend requires CUDA tensors"
        if self.world_size == 1:
            return tensor
        send_tensors = [tensor[idx] for idx in self.local_indices]
        recv_tensors, future = utils.all_to_all(
            send_tensors, part2size=part_size, is_async=True, group=self.group
        )
        buf_shape = self.g.num_nodes(), *tensor.shape[1:]
        buffer = torch.empty(buf_shape, dtype=tensor.dtype, device=tensor.device)
        for i in future.as_completed():
            buffer[self.part2idx[i]] = recv_tensors[i]
            recv_tensors[i] = None
        return buffer
