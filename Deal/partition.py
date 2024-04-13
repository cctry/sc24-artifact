import math
import os
import queue
from abc import ABC, abstractmethod
from concurrent import futures
from threading import Thread
from typing import List, Optional, Tuple

import dgl
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist

import utils


class Partitioner(ABC):
    @abstractmethod
    def __call__(
        self,
        inner_nodes: np.ndarray,
        batch_size: int,
        include_out_edge: bool,
    ) -> Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor]:
        pass


def read_batches(
    edge_file: pq.ParquetFile,
    num_workers: int,
    batch_size: int,
    src_name: str,
    dst_name: str,
    data_queue: queue.Queue,
):
    for batch_id, batch in enumerate(
        edge_file.iter_batches(batch_size=batch_size, columns=[src_name, dst_name])
    ):
        src_col = batch.column(src_name).to_numpy(zero_copy_only=False)
        dst_col = batch.column(dst_name).to_numpy(zero_copy_only=False)
        data_queue.put((batch_id, src_col, dst_col))
    for _ in range(num_workers):
        data_queue.put(None)


def stream_partition(
    edge_file: pq.ParquetFile,
    src_name: str,
    dst_name: str,
    inner_nodes: np.ndarray,
    batch_size: int = 1024 * 1024,
    include_out_edge: bool = False,
) -> Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor]:
    """Stream partition a graph from the edge list stored in a Parquet file.

    Parameters
    ----------
    edge_file : pq.ParquetFile
        The file containing the edge list.
    src_name : str
        The name of the source node ID column.
    dst_name : str
        The name of the destination node ID column.
    inner_nodes : np.ndarray
        The node IDs of the inner nodes.

    Returns
    -------
    DGLGraph
        The local partitioned graph.
    torch.Tensor
        The mapping from local node IDs to node IDs in edge file.
    torch.Tensor
        The local edge IDs.

    Notes
    -----
    The local partition has only the in-edge of the inner nodes.
    """
    num_batches = math.ceil(edge_file.metadata.num_rows / batch_size)
    data_queue = queue.Queue(maxsize=10)

    local_src = [None] * num_batches
    local_dst = [None] * num_batches
    local_edge = [None] * num_batches
    unique_nodes = [None] * num_batches

    def process_batch():
        while True:
            data = data_queue.get(timeout=10)
            if data is None:
                break
            batch_id, src_col, dst_col = data
            if include_out_edge:
                srcdst = np.concatenate((src_col, dst_col))
                edge_mask = utils.fast_isin(inner_nodes, srcdst)
                edge_mask = edge_mask[: len(dst_col)] | edge_mask[len(dst_col) :]
            else:
                edge_mask = utils.fast_isin(inner_nodes, dst_col)
            local_src[batch_id] = torch.from_numpy(src_col[edge_mask])
            local_dst[batch_id] = torch.from_numpy(dst_col[edge_mask])
            local_edge[batch_id] = torch.from_numpy(
                edge_mask.nonzero()[0] + (batch_id * batch_size)
            )
            unique_nodes[batch_id] = utils.cat_unique(
                [local_src[batch_id], local_dst[batch_id]]
            )

    reader_thread = Thread(
        target=read_batches,
        args=(
            edge_file,
            os.cpu_count() // 2,
            batch_size,
            src_name,
            dst_name,
            data_queue,
        ),
    )
    reader_thread.start()
    inner_nodes = np.sort(inner_nodes)
    with futures.ThreadPoolExecutor() as executor:
        future = [executor.submit(process_batch) for _ in range(os.cpu_count() // 2)]
        # Block until all tasks are complete
        utils.check_exception(future)

    local_edge = torch.cat(local_edge)
    # map node
    node_mapping = utils.cat_unique(unique_nodes)
    src_local_id = utils.cat_searchsorted(node_mapping, local_src)
    dst_local_id = utils.cat_searchsorted(node_mapping, local_dst)
    # add remaining nodes
    mask = utils.fast_isin(node_mapping.numpy(), inner_nodes)
    diff_tensor = torch.from_numpy(inner_nodes[~mask])
    node_mapping = torch.cat((node_mapping, diff_tensor))
    # create graph
    graph = dgl.graph((src_local_id, dst_local_id), num_nodes=len(node_mapping))
    return graph, node_mapping, local_edge


class SingleFilePartitioner(Partitioner):
    def __init__(
        self,
        edge_file: pq.ParquetFile,
        src_name: str,
        dst_name: str,
    ) -> None:
        self.edge_file = edge_file
        self.src_name = src_name
        self.dst_name = dst_name

    def __call__(
        self,
        inner_nodes: np.ndarray,
        batch_size: int,
        include_out_edge: bool = False,
    ) -> Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor]:
        return stream_partition(
            self.edge_file,
            self.src_name,
            self.dst_name,
            inner_nodes,
            batch_size,
            include_out_edge,
        )


def compute_degree(
    src: torch.Tensor,
    dst: torch.Tensor,
    unique_nodes: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the in-degree and out-degree of nodes in a graph.

    Args:
        src (torch.Tensor): The source nodes of the graph.
        dst (torch.Tensor): The destination nodes of the graph.
        unique_nodes (torch.Tensor): The unique nodes in the graph.
        group (Optional[dist.ProcessGroup]): The process group to use for distributed computation.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the in-degree and out-degree of nodes in the graph.
    """
    num_nodes = torch.tensor(unique_nodes.numel())
    dist.all_reduce(num_nodes, group=group)
    min_index = unique_nodes.min()
    dist.all_reduce(min_index, op=dist.ReduceOp.MIN, group=group)
    num_nid = num_nodes + min_index
    in_degree = utils.sorted_count(dst, unique_nodes, num_nid).float()
    op1 = dist.all_reduce(in_degree, group=group, async_op=True)
    out_degree = utils.sorted_count(src, unique_nodes, num_nid).float()
    op2 = dist.all_reduce(out_degree, group=group, async_op=True)
    op1.wait()
    op2.wait()
    return in_degree, out_degree


def stream_partition_multiple_files(
    edge_files: List[pq.ParquetFile],
    src_name: str,
    dst_name: str,
    inner_nodes: np.ndarray,
    batch_size: int = 1024 * 1024,
    include_out_edge: bool = False,
) -> Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor]:
    """
    Reads edge data from multiple Parquet files and generate a DGL graph object from edges that are connected to
    inner nodes. All workers read all edges from multiple files and then exchange them.

    Args:
        edge_files (List[pq.ParquetFile]): List of ParquetFile objects containing edge data.
        src_name (str): Name of the source column in the Parquet files.
        dst_name (str): Name of the destination column in the Parquet files.
        inner_nodes (np.ndarray): Array of node IDs that should be included in the resulting graph.
        batch_size (int, optional): Number of edges to read at a time. Defaults to 1024 * 1024.
        include_out_edge (bool, optional): Whether to include out-edges in the resulting graph. Defaults to False.

    Returns:
        Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor]: A tuple containing the resulting DGL graph object,
        a tensor mapping node IDs to global IDs, and the ID of local edges.
    """
    # compute the base on Edge ID
    partial_num_edges = torch.tensor([f.metadata.num_rows for f in edge_files])
    num_edges = [None] * dist.get_world_size()
    dist.all_gather_object(num_edges, partial_num_edges)
    num_edges = torch.cat(num_edges)
    edge_offset = torch.cat([torch.zeros(1, dtype=torch.int64), num_edges.cumsum(0)])
    data_queues = [queue.Queue(maxsize=10) for _ in edge_files]
    num_workers = max(1, os.cpu_count() // 2 // len(edge_files))
    threads = []
    for i, edge_file in enumerate(edge_files):
        args = (edge_file, num_workers, batch_size, src_name, dst_name, data_queues[i])
        threads.append(Thread(target=read_batches, args=args))
        threads[-1].start()
    part2inner = [None] * dist.get_world_size()
    inner_nodes = np.sort(inner_nodes)

    dist.all_gather_object(part2inner, inner_nodes)

    def process_batch(
        data_queue: queue.Queue, res_list: List[queue.Queue], eid_base: int
    ):
        while True:
            data = data_queue.get()
            if data is None:
                break
            batch_id, src_col, dst_col = data
            for i, inner in enumerate(part2inner):
                if include_out_edge:
                    srcdst = np.hstack((src_col, dst_col))
                    edge_mask = utils.fast_isin(inner, srcdst)
                    edge_mask = edge_mask[: len(dst_col)] | edge_mask[len(dst_col) :]
                else:
                    edge_mask = utils.fast_isin(inner, dst_col)
                if not edge_mask.any():
                    continue
                local_src = torch.from_numpy(src_col[edge_mask])
                local_dst = torch.from_numpy(dst_col[edge_mask])
                local_edge = (
                    torch.from_numpy(edge_mask.nonzero()[0] + (batch_id * batch_size))
                    + eid_base
                )
                unique_nodes = utils.cat_unique([local_src, local_dst])
                res_list[i].put((local_src, local_dst, local_edge, unique_nodes))

    res_lists = []
    for i, edge_file in enumerate(edge_files):
        res_lists.append([queue.Queue() for _ in range(dist.get_world_size())])
        for _ in range(num_workers):
            threads.append(
                Thread(
                    target=process_batch,
                    args=(data_queues[i], res_lists[i], edge_offset[i]),
                )
            )
            threads[-1].start()
    _ = [t.join() for t in threads]

    srcs, dsts, local_edges, unique_nodes = [], [], [], []  # per partition
    for i in range(dist.get_world_size()):
        src, dst, local_edge, unique_node = [], [], [], []
        for j in range(len(edge_files)):
            data_list = list(res_lists[j][i].queue)
            if len(data_list) == 0:
                continue
            src.append(torch.cat([data[0] for data in data_list]))
            dst.append(torch.cat([data[1] for data in data_list]))
            local_edge.append(torch.cat([data[2] for data in data_list]))
            unique_node.append(torch.cat([data[3] for data in data_list]))

        srcs.append(torch.cat(src))
        dsts.append(torch.cat(dst))
        local_edges.append(torch.cat(local_edge))
        unique_nodes.append(torch.cat(unique_node))

    part2size = utils.compute_part2size(srcs)
    unique_nodes, un_future = utils.all_to_all(unique_nodes, is_async=True)
    local_edge, edge_future = utils.all_to_all(local_edges, part2size, is_async=True)
    local_src, src_future = utils.all_to_all(srcs, part2size, is_async=True)
    local_dst, dst_future = utils.all_to_all(dsts, part2size, is_async=True)
    un_future.wait()
    node_mapping = utils.cat_unique(unique_nodes)
    edge_future.wait()
    local_edge = torch.cat(local_edge)
    local_edge, local_edge_idx = torch.sort(local_edge, stable=True)
    src_future.wait()
    local_src_id = utils.cat_searchsorted(node_mapping, local_src)
    local_src_id = local_src_id[local_edge_idx]
    dst_future.wait()
    local_dst_id = utils.cat_searchsorted(node_mapping, local_dst)
    local_dst_id = local_dst_id[local_edge_idx]

    # add remaining nodes
    mask = utils.fast_isin(node_mapping.numpy(), inner_nodes)
    diff_tensor = torch.from_numpy(inner_nodes[~mask])
    node_mapping = torch.cat((node_mapping, diff_tensor))
    # create graph
    graph = dgl.graph((local_src_id, local_dst_id), num_nodes=len(node_mapping))    

    return graph, node_mapping, local_edge


class MulitpleFilePartitioner(Partitioner):
    def __init__(
        self,
        edge_file: List[pq.ParquetFile],
        src_name: str,
        dst_name: str,
    ) -> None:
        self.edge_files = edge_file
        self.src_name = src_name
        self.dst_name = dst_name

    def __call__(
        self,
        inner_nodes: np.ndarray,
        batch_size: int,
        include_out_edge: bool = False,
    ) -> Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor]:
        return stream_partition_multiple_files(
            self.edge_files,
            self.src_name,
            self.dst_name,
            inner_nodes,
            batch_size,
            include_out_edge,
        )
