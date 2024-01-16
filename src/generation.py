from typing import Optional, Tuple

import dgl
import torch
import torch.distributed as dist

from . import partition, utils


def generate_graph(
    node_ids: torch.Tensor,
    partition_algo: partition.Partitioner,
    batch_size: int = 1024 * 1024,
    group: Optional[dist.ProcessGroup] = None,
    include_out_edge: bool = False,
) -> Tuple[dgl.DGLGraph, torch.Tensor]:
    """Generate a DGLGraph by partitioning the edge list.

    Parameters
    ----------
    node_ids : torch.Tensor
        The ID of all nodes used to generate the graph from files.

    partition_algo : partition.Partitioner
        The partition algorithm.

    group : dist.ProcessGroup
        Processes in the group will have different graphs.

    Returns
    -------
    DGLGraph
        The generated graph.
    torch.Tensor
        The mapping from global node IDs to the ID in the node file.

    Notes
    -----
    The partition graph contains additional node data:

    - “inner_node” indicates whether a node belongs to the partition.

    - “part_id” indicates the partition ID of a node.

    - “dgl.NID” indicates the global node ID of a node.

    Processes in the group will have different graphs.
    Processes in different groups but with the same rank will have the same graph.
    """
    part_rank = dist.get_rank(group)
    with utils.same_random_seed():
        node_mapping = node_ids[torch.randperm(len(node_ids))]
    part_offset = utils.balanced_split(dist.get_world_size(group), len(node_ids))
    inner_nodes = node_mapping[part_offset[part_rank] : part_offset[part_rank + 1]]
    sort_thread, (sorted_nid, indices) = utils.async_sort(node_mapping)
    # partition edges
    graph, origin_id, local_edge = partition_algo(
        inner_nodes.numpy(), batch_size, include_out_edge
    )
    # map original node id to global id
    sort_thread.join()
    graph.ndata[dgl.NID] = indices[torch.searchsorted(sorted_nid, origin_id)]
    graph.ndata["part_id"] = torch.bucketize(
        graph.ndata[dgl.NID], part_offset[1:], right=True, out_int32=True
    )
    graph.ndata["inner_node"] = graph.ndata["part_id"] == part_rank
    graph.ndata["origin_id"] = origin_id
    # map original edge id to global id
    graph.edata[dgl.EID] = local_edge
    graph.include_out_edge = include_out_edge
    return graph, node_mapping


def generate_graph_intelligent(
    partition_algo: partition.Partitioner,
    batch_size: int = 1024 * 1024,
    group: Optional[dist.ProcessGroup] = None,
    include_out_edge: bool = False,
) -> Tuple[dgl.DGLGraph, torch.Tensor]:
    """Generate a DGLGraph by partitioning the edge list.

    Parameters
    ----------
    node_ids : torch.Tensor
        The ID of all nodes used to generate the graph from files.

    partition_algo : partition.Partitioner
        The partition algorithm.

    group : dist.ProcessGroup
        Processes in the group will have different graphs.

    Returns
    -------
    DGLGraph
        The generated graph.
    torch.Tensor
        The mapping from global node IDs to the ID in the node file.

    Notes
    -----
    The partition graph contains additional node data:

    - “inner_node” indicates whether a node belongs to the partition.

    - “part_id” indicates the partition ID of a node.

    - “dgl.NID” indicates the global node ID of a node.

    Processes in the group will have different graphs.
    Processes in different groups but with the same rank will have the same graph.
    """
    part_rank = dist.get_rank(group)
    with utils.same_random_seed():
        node_mapping = node_ids[torch.randperm(len(node_ids))]
    part_offset = utils.balanced_split(dist.get_world_size(group), len(node_ids))
    inner_nodes = node_mapping[part_offset[part_rank] : part_offset[part_rank + 1]]
    sort_thread, (sorted_nid, indices) = utils.async_sort(node_mapping)
    # partition edges
    graph, origin_id, local_edge = partition_algo(
        inner_nodes.numpy(), batch_size, include_out_edge
    )
    # map original node id to global id
    sort_thread.join()
    graph.ndata[dgl.NID] = indices[torch.searchsorted(sorted_nid, origin_id)]
    graph.ndata["part_id"] = torch.bucketize(
        graph.ndata[dgl.NID], part_offset[1:], right=True
    )
    graph.ndata["inner_node"] = graph.ndata["part_id"] == part_rank
    graph.ndata["origin_id"] = origin_id
    # map original edge id to global id
    graph.edata[dgl.EID] = local_edge
    graph.include_out_edge = include_out_edge
    return graph, node_mapping