import math
import threading
from typing import List

import dgl
import torch
import torch.distributed as dist

from . import utils


class FeatCommunicator:
    def __init__(self, g: dgl.DGLGraph, group: dist.ProcessGroup):
        self.group = group
        self.part_id = dist.get_rank(group)
        self.num_parts = dist.get_world_size(group)
        self.part2idx = [
            torch.where(g.ndata["part_id"] == i)[0] for i in range(self.num_parts)
        ]
        # required global ID from each partition
        self.part2gid = [g.ndata[dgl.NID][t] for t in self.part2idx]
        # The local feat idx sent to each partition
        self.local_ids = [None] * self.num_parts
        # The recv buffer and future for all-to-all
        self.recv_buffer, self.future = None, None
        # Returned by the dataloader
        self.g = g
        self.local_feat = None
        self.rst_rsc = None
        self.rst_dst = None

    def prepare(self, local_feat):
        """
        Prepares the comm for the next iteration by setting the input features,
        exchanging global IDs between partitions, and computing the local IDs for each partition.

        Args:
            local_feat (torch.Tensor): The local features for the current partition.
            sorted_nid (torch.Tensor): The sorted global IDs of all nodes in the current partition.
            feat_nid_indices (torch.Tensor): The indices of the global IDs matching the local features.

        Returns:
            None
        """
        # part2size used for feature all-to-all
        send_tensors = [local_feat[idx] for idx in self.local_ids]
        if self.num_parts > 1:
            part2size = [[None] * self.num_parts for _ in range(self.num_parts)]
            for i in range(self.num_parts):
                part2size[i][self.part_id] = torch.Size(
                    (len(self.part2gid[i]), *local_feat.shape[1:])
                )
            self.recv_buffer, self.future = utils.all_to_all(
                send_tensors, part2size, True, self.group
            )
        self.local_feat = send_tensors[self.part_id]

    def sync_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if dist.get_backend(self.group) == dist.Backend.NCCL:
            assert tensor.is_cuda, "NCCL backend requires CUDA tensors"
        assert (
            tensor is self.local_feat
        ), "Cannot sync tensors not prepared by this comm"
        if self.num_parts == 1:
            return tensor
        res_tensor = torch.empty(
            (self.g.num_nodes(), *self.local_feat.shape[1:]),
            dtype=self.local_feat.dtype,
            device=self.local_feat.device,
        )
        for i in self.future.as_completed():
            res_tensor[self.part2idx[i]] = self.recv_buffer[i]
            self.recv_buffer[i] = None
        return res_tensor


class EdgeBatchCommunicator:
    __slots__ = [
        "group",
        "part_id",
        "num_parts",
        "part2idx",
        "part2gid",
        "local_ids",
        "recv_buffer",
        "future",
        "graph",
        "local_feat",
        "rst_src",
        "rst_dst",
    ]

    def __init__(self, graph: dgl.DGLGraph, group: dist.ProcessGroup):
        self.group = group
        self.part_id = dist.get_rank(group)
        self.num_parts = dist.get_world_size(group)
        # local ID of nodes receiving
        self.part2idx = [
            torch.where(graph.ndata["part_id"] == i)[0] for i in range(self.num_parts)
        ]
        # required global ID from each partition
        self.part2gid = [graph.ndata[dgl.NID][t] for t in self.part2idx]
        # The local feat idx sent to each partition
        self.local_ids = [None] * self.num_parts
        # The recv buffer and future for all-to-all
        self.recv_buffer, self.future = None, None
        # Returned by the dataloader
        self.graph = graph
        self.local_feat = None
        self.rst_src = None
        self.rst_dst = None

    def prepare(self, local_feat):
        """
        Prepares the comm for the next iteration by setting the input features,
        exchanging global IDs between partitions, and computing the local IDs for each partition.

        Args:
            local_feat (torch.Tensor): The local features for the current partition.
            sorted_nid (torch.Tensor): The sorted global IDs of all nodes in the current partition.
            feat_nid_indices (torch.Tensor): The indices of the global IDs matching the local features.

        Returns:
            None
        """
        # part2size used for feature all-to-all
        send_tensors = [local_feat[idx] for idx in self.local_ids]
        # self.local_ids = None
        if self.num_parts > 1:
            part2size = [[None] * self.num_parts for _ in range(self.num_parts)]
            for i in range(self.num_parts):
                part2size[i][self.part_id] = torch.Size(
                    (len(self.part2gid[i]), *local_feat.shape[1:])
                )
            # if dist.get_rank(self.group) == 0:
            #     print(part2size)
            #     print("local_ids", [len(i) for i in self.local_ids])
            #     self.local_ids = None
            self.recv_buffer, self.future = utils.all_to_all(
                send_tensors, part2size, True, self.group
            )
        self.local_feat = send_tensors[self.part_id]

    def sync(self) -> torch.Tensor:
        assert self.local_feat is not None, "local_feat is not set"
        if self.num_parts == 1:
            return self.local_feat
        res_tensor = torch.empty(
            (self.graph.num_nodes(), *self.local_feat.shape[1:]),
            dtype=self.local_feat.dtype,
            device=self.local_feat.device,
        )
        for i in self.future.as_completed():
            # res_tensor[self.part2idx[i]] = self.recv_buffer[i]
            res_tensor.index_put_((self.part2idx[i],), self.recv_buffer[i])
            # self.recv_buffer[i] = None
        # self.local_feat = None
        # self.future = None
        return res_tensor

PRINT = True

class EdgeBatchCommunicatorP2P:
    __slots__ = [
        "group",
        "part_id",
        "num_parts",
        "part2idx",
        "part2gid",
        "local_ids",
        "recv_buffer",
        "future",
        "graph",
        "local_feat",
        "rst_src",
        "rst_dst",
        "op_list",
        "reqs",
        "recv_from",
    ]

    def __init__(self, graph: dgl.DGLGraph, group: dist.ProcessGroup):
        self.group = group
        self.part_id = dist.get_rank(group)
        self.num_parts = dist.get_world_size(group)
        # local ID of nodes receiving
        self.part2idx = [
            torch.where(graph.ndata["part_id"] == i)[0] for i in range(self.num_parts)
        ]
        # required global ID from each partition
        self.part2gid = [graph.ndata[dgl.NID][t] for t in self.part2idx]
        # The local feat idx sent to each partition
        self.local_ids = [None] * self.num_parts
        # The recv buffer and future for all-to-all
        self.recv_buffer, self.future = None, None
        # Returned by the dataloader
        self.graph = graph
        self.local_feat = None
        self.rst_src = None
        self.rst_dst = None
        self.op_list = [None, None]
        self.reqs = [None, None]
        self.recv_from = -1

    def prepare(self, local_feat):
        """
        Prepares the comm for the next iteration by setting the input features,
        exchanging global IDs between partitions, and computing the local IDs for each partition.

        Args:
            local_feat (torch.Tensor): The local features for the current partition.
            sorted_nid (torch.Tensor): The sorted global IDs of all nodes in the current partition.
            feat_nid_indices (torch.Tensor): The indices of the global IDs matching the local features.

        Returns:
            None
        """
        send_tensors = [local_feat[idx] for idx in self.local_ids]
        if self.num_parts > 1:
            recv_from = send_to = -1
            for i in range(dist.get_world_size(self.group)):
                if i != self.part_id:
                    if len(self.local_ids[i]) != 0:
                        send_to = i
                    if len(self.part2gid[i]) != 0:
                        recv_from = i
            size = torch.Size((len(self.part2gid[recv_from]), *local_feat.shape[1:]))
            self.recv_buffer = torch.empty(
                size, dtype=local_feat.dtype, device=local_feat.device
            )
            # global PRINT
            # if PRINT:
            #     # PRINT = False
            #     print(f"[{dist.get_rank(self.group)}] local_ids", [len(i) for i in self.local_ids])
            #     print(f"[{dist.get_rank(self.group)}] part2gid", [len(i) for i in self.part2gid])
            #     print(f"[{dist.get_rank(self.group)}] size", size, "from", recv_from) 
            #     print(f"[{dist.get_rank(self.group)}] size", send_tensors[send_to].size(), "to", send_to)

            assert (send_to == -1) == (recv_from == -1), f"[{dist.get_rank(self.group)}] send_to {send_to} recv_from {recv_from} {self.part2gid[recv_from]} {self.local_ids[send_to]}"
            if send_to != -1 and recv_from != -1:
                recv_from_g = dist.get_global_rank(self.group, recv_from)
                send_to_g = dist.get_global_rank(self.group, send_to)
                self.op_list[0] = dist.P2POp(
                    dist.irecv,
                    self.recv_buffer,
                    recv_from_g,
                    self.group,
                )
                self.op_list[1] = dist.P2POp(
                    dist.isend,
                    send_tensors[send_to].contiguous(),
                    send_to_g,
                    self.group,
                )
                self.reqs = dist.batch_isend_irecv(self.op_list)
            self.recv_from = recv_from
        # self.local_ids = None
        self.local_feat = send_tensors[self.part_id]

    def sync(self) -> torch.Tensor:
        assert self.local_feat is not None, "local_feat is not set"
        if self.num_parts == 1:
            return self.local_feat
        res_tensor = torch.empty(
            (self.graph.num_nodes(), *self.local_feat.shape[1:]),
            dtype=self.local_feat.dtype,
            device=self.local_feat.device,
        )
        res_tensor.index_put_((self.part2idx[self.part_id],), self.local_feat)
        if self.reqs[0] is not None:
            self.reqs[0].wait()
            self.reqs[0] = None
        # for i, req in enumerate(self.reqs):
        #     if req is not None:
        #         req.wait()
        #     self.reqs[i] = None
        res_tensor.index_put_((self.part2idx[self.recv_from],), self.recv_buffer)
        if self.reqs[1] is not None:
            self.reqs[1].wait()
            self.reqs[1] = None
        return res_tensor


class BaseScheduler:
    def __init__(self, graph, eid, batch_size, group, sync_num=True) -> None:
        self.graph = graph
        self.eid = eid
        self.src, self.dst = graph.find_edges(eid)
        self.num_edges = len(eid)
        self.batch_size = batch_size
        self.group = group
        self.current_batch = 0
        self.batches = None
        self.num_batches = torch.tensor(math.ceil(self.num_edges / batch_size))
        if sync_num:
            dist.all_reduce(self.num_batches, op=dist.ReduceOp.MIN, group=self.group)
        assert (
            self.num_batches.item() > 0
        ), f"num_batches is {self.num_batches.item()} num_edges is {self.num_edges}"

    def __iter__(self):
        self.current_batch = 0
        edge_id = torch.arange(self.num_edges)
        self.batches = torch.tensor_split(edge_id, self.num_batches)
        return self

    def __next__(self):
        if self.current_batch >= len(self.batches):
            raise StopIteration
        batch = self.batches[self.current_batch]
        self.current_batch += 1
        return self.src[batch], self.dst[batch], batch


class SrcSortScheduler(BaseScheduler):
    def __iter__(self):
        self.current_batch = 0
        sorted_edge = torch.argsort(self.src, stable=True)  # sort by src
        self.batches = torch.tensor_split(sorted_edge, self.num_batches)
        return self


class RingScheduler(BaseScheduler):
    def __init__(self, graph, eid, batch_size, group) -> None:
        super().__init__(graph, eid, batch_size, group, False)
        self.recv_tgts = []
        rank = dist.get_rank(self.group)
        M = dist.get_world_size(self.group)
        for b in range(M):
            recv_from = (rank - b - 1) % M
            target = torch.where(self.graph.ndata["part_id"][self.src] == recv_from)[0]
            assert target.numel() > 0, f"Receive nothing from partition {recv_from}"
            self.recv_tgts.append(target)

        self.num_batches = torch.tensor(
            [math.ceil(tgt.numel() / batch_size) for tgt in self.recv_tgts]
        )
        dist.all_reduce(self.num_batches, op=dist.ReduceOp.MIN, group=self.group)

    def __iter__(self):
        for num_batches, target in zip(self.num_batches, self.recv_tgts):
            target = target[torch.argsort(self.src[target], stable=True)]
            batches = torch.tensor_split(target, num_batches)
            for batch in batches:
                yield self.src[batch], self.dst[batch], batch


class PairScheduler(BaseScheduler):

    def generate_pairs_schedule(self, num_machines):
        """Generate a schedule where each machine communicates with every other machine exactly once."""
        # The schedule will have num_machines - 1 rounds
        rounds = [set() for _ in range(num_machines - 1)]

        # Function to check if a machine is already in a round
        def is_machine_in_round(machine, round):
            return any(machine in pair for pair in rounds[round])

        # Generate pairs for each round
        for machine1 in range(num_machines):
            for machine2 in range(machine1 + 1, num_machines):
                # Find a round where both machines are not yet scheduled
                for round_num in range(num_machines - 1):
                    if not is_machine_in_round(machine1, round_num) and not is_machine_in_round(machine2, round_num):
                        rounds[round_num].add((machine1, machine2))
                        break

        return rounds

    def func(self, rank, world_size):
        """
        Generate a list of machines for a given machine to interact with in the following rounds.

        :param rank: The rank of the current machine.
        :param world_size: Total number of machines.
        :return: A list of machines to interact with in the following rounds.
        """
        # Generate the full schedule using the previous function
        full_schedule = self.generate_pairs_schedule(world_size)

        # Extract the specific schedule for the given machine
        machine_schedule = []
        for round in full_schedule:
            for pair in round:
                if rank in pair:
                    # Add the other machine in the pair
                    machine_schedule.append(pair[0] if pair[1] == rank else pair[1])
                    break

        return machine_schedule + [dist.get_rank(self.group)]

    def __init__(self, graph, eid, batch_size, group) -> None:
        super().__init__(graph, eid, batch_size, group, False)
        
        self.recv_tgts = []
        rank = dist.get_rank(self.group)
        M = dist.get_world_size(self.group)
        assert M % 2 == 0, "pair-wise only works for even number of partitions"
        assert M in [4,8,16], "pair-wise only works for 4, 8, 16 partitions"
        for other in self.func(rank, M):
            target = torch.where(self.graph.ndata["part_id"][self.src] == other)[0]
            assert target.numel() > 0, f"Receive nothing from partition {other}"
            self.recv_tgts.append(target)

        self.num_batches = torch.tensor(
            [math.ceil(tgt.numel() / batch_size) for tgt in self.recv_tgts]
        )
        dist.all_reduce(self.num_batches, op=dist.ReduceOp.MIN, group=self.group)

    def __iter__(self):
        for num_batches, target in zip(self.num_batches, self.recv_tgts):
            target = target[torch.argsort(self.src[target], stable=True)]
            batches = torch.tensor_split(target, num_batches)
            for batch in batches:
                yield self.src[batch], self.dst[batch], batch

class BatchLoader:
    def __init__(
        self,
        graph: dgl.DGLGraph,
        fanout: List[int],
        batch_size: int,
        feat_group: dist.ProcessGroup,
        graph_group: dist.ProcessGroup,
        scheduler=BaseScheduler,
    ):
        # set random seed to make sure the sampled edges are the same within feature group
        dgl.seed(
            42
        )  # TODO: Cannot ensure the same sampled edges across different feature groups
        self.nid = graph.ndata[dgl.NID]
        self.part_id = graph.ndata["part_id"]
        self.inner_node = graph.ndata["inner_node"]

        self.num_parts = dist.get_world_size(feat_group)
        self.batch_size = batch_size
        self.fanout = fanout
        self.feat_group = feat_group
        self.graph_group = graph_group

        inner_nodes = torch.where(self.inner_node)[0]
        self.sorted_nid, self.feat_nid_indices = torch.sort(self.nid[inner_nodes])
        self.num_hop = len(fanout)

        self.batches: List[List[FeatCommunicator]] = [[] for _ in range(self.num_hop)]
        self.all_part2gid = [[] for _ in range(self.num_hop)]
        self.schedulers = []

        self.sampled_graphs = [None] * self.num_hop

        for l, f in enumerate(fanout):
            # g_ = [None]
            # src_rank = dist.get_global_rank(graph_group, 0)
            # if dist.get_rank(graph_group) == 0:
            #     g_[0] = graph.sample_neighbors(inner_nodes, f)
            # dist.broadcast_object_list(g_, src=src_rank, group=graph_group) # TODO: optimize this
            # sampled_g = g_[0]

            sampled_g = graph.sample_neighbors(inner_nodes, f)

            # print(f"[{dist.get_rank()}] {sampled_g.edata[dgl.EID]}")

            assert inner_nodes.numel() > 0, "No inner nodes"
            assert sampled_g.num_nodes() > 0, "No sampled nodes"
            assert sampled_g.num_edges() > 0, "No sampled edges" + str(
                graph.in_degrees(inner_nodes)
            )
            sampled_edge = sampled_g.edata[dgl.EID]
            self.sampled_graphs[l] = sampled_g
            self.schedulers.append(
                scheduler(graph, sampled_edge, batch_size, feat_group)
            )

        self.threads = []
        self.thread_flags = [True] * self.num_hop
        for l in range(self.num_hop):
            args = (
                self.schedulers[l],
                self.batches[l],
                self.all_part2gid[l],
            )
            self.threads.append(threading.Thread(target=self._setup, args=args))
            self.threads[-1].start()

        self.local_feat = None
        self.current_batch = 0
        self.current_hop = -1

    def _setup(
        self,
        scheduler: BaseScheduler,
        batches: List[FeatCommunicator],
        all_part2gid: List[torch.Tensor],
    ):
        _dispatch = {
            RingScheduler: EdgeBatchCommunicatorP2P,
            PairScheduler: EdgeBatchCommunicatorP2P,
            BaseScheduler: EdgeBatchCommunicator,
            SrcSortScheduler: EdgeBatchCommunicator,
        }
        comm_cls = _dispatch[type(scheduler)]
        # comm_cls = EdgeBatchCommunicator
        for batch_src, batch_dst, batch_eid in scheduler:
            nodes = utils.cat_unique(
                (batch_src, batch_dst)
            )  # local node ID for this batch

            # batch node ID
            batch_src = torch.searchsorted(nodes, batch_src)
            batch_dst = torch.searchsorted(nodes, batch_dst)

            g = dgl.graph((batch_src, batch_dst))
            g.ndata[dgl.NID] = self.nid[nodes]
            g.edata[dgl.EID] = batch_eid
            g.ndata["part_id"] = self.part_id[nodes]
            g.ndata["inner_node"] = self.inner_node[nodes]
            batch = comm_cls(g, self.feat_group)
            # batch node ID for dst nodes
            batch.rst_src = batch_dst
            # local node ID for dst nodes
            batch.rst_dst = torch.searchsorted(nodes, batch_dst)
            batches.append(batch)
        for p in range(self.num_parts):
            all_part2gid.append(
                utils.pad_and_cat([b.part2gid[p] for b in batches], True)[0]
            )


    def _post_init(self, hop):
        all_part2gid_sent, future = utils.all_to_all(
            self.all_part2gid[hop], group=self.feat_group, is_async=True
        )
        for i in future.as_completed():
            for b, batch in enumerate(self.batches[hop]):
                length = all_part2gid_sent[i][b, -1].item()
                idx = all_part2gid_sent[i][b, :length]
                idx = torch.searchsorted(self.sorted_nid, idx)
                batch.local_ids[i] = self.feat_nid_indices[idx]


    def set_local_feat(self, local_feat):
        self.local_feat = local_feat

    def set_hop(self, hop):
        assert hop > self.current_hop, "Cannot go back to previous layers"
        if self.current_hop >= 0:
            self.batches[self.current_hop] = None
            self.sampled_graphs[self.current_hop] = None
        self.current_hop = hop

    def __iter__(self):
        assert self.local_feat is not None, "local_feat is not set"
        assert self.current_hop < self.num_hop, "No more layers"
        assert self.current_hop >= 0, "current_hop is not set"
        self.current_batch = 0
        if self.thread_flags[self.current_hop]:
            self.threads[self.current_hop].join()
            self.threads[self.current_hop] = None
            self.schedulers[self.current_hop] = None
            self._post_init(self.current_hop)
            self.thread_flags[self.current_hop] = False
            self.all_part2gid[self.current_hop] = None
        self.batches[self.current_hop][self.current_batch].prepare(self.local_feat)
        return self

    def __next__(self) -> EdgeBatchCommunicator:
        if self.current_batch >= len(self.batches[self.current_hop]):
            raise StopIteration
        batch = self.batches[self.current_hop][self.current_batch]
        self.current_batch += 1
        if self.current_batch < len(self.batches[self.current_hop]):
            self.batches[self.current_hop][self.current_batch].prepare(self.local_feat)
        return batch.graph, batch
