import argparse
from contextlib import contextmanager
import datetime
import math
import os
import queue
import threading
from concurrent import futures
import os
import sys
import time
from typing import List, Union

import dgl
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist

from deal import dist_models, utils
from deal.feat_comm import FeatCommunicator
from deal.generation import generate_graph
from deal.load_node_feat import read_batches, load_node_feat_single_file
from deal.partition import MulitpleFilePartitioner, SingleFilePartitioner
from deal.dataloader import BatchLoader, BaseScheduler, SrcSortScheduler, RingScheduler

torch.set_grad_enabled(False)

SCHEDULER_CLS = {
    "src": SrcSortScheduler,
    "ring": RingScheduler,
    "base": BaseScheduler,
}

TIMER = None
OUT_FILE = None


def init_timer(args):
    output_file = f"{dist.get_rank()}_{args.graph_name}_{args.model}_{args.scheduler}_{args.fanout}_{args.feat_group}_{args.graph_group}.log"
    global TIMER, OUT_FILE
    OUT_FILE = open(output_file, "w+")
    template = f"[{dist.get_rank()}] {{}} time: {{}} seconds"

    @contextmanager
    def _timer(note: str) -> None:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            print(template.format(note, elapsed), file=OUT_FILE)
            print(template.format(note, elapsed))

    TIMER = _timer

def load_node_feat_partitioned_file(
    node_files: List[pq.ParquetFile],
    feat_cols: Union[str, List[str]],
    node_id_col: str,
    target_id: torch.Tensor,
    batch_size: int = 8192,
    group: dist.ProcessGroup = None,
) -> torch.Tensor:
    def process_batch(data_queue, res_list):
        while True:
            data = data_queue.get()
            if data is None:
                break
            batch_id, node_ids, *feats = data
            node_ids = node_ids.to_numpy(zero_copy_only=False)
            feats_list = []
            for feat in feats:
                feat = feat.to_numpy(zero_copy_only=False).tolist()
                feats_list.append(utils.fast_vstack(feat))
            res_list[batch_id] = (node_ids, torch.hstack(feats_list))

    num_files = len(node_files)
    data_queues = [queue.Queue(maxsize=64) for _ in node_files]
    num_batches = [math.ceil(f.metadata.num_rows / batch_size) for f in node_files]
    res_lists = [[None] * num_batches[i] for i in range(num_files)]
    workers_per_file = max(1, os.cpu_count() // 2 // num_files)
    with futures.ThreadPoolExecutor(num_files * (workers_per_file + 1)) as executor:
        res = []
        for i in range(num_files):
            res.append(
                executor.submit(
                    read_batches,
                    node_files[i],
                    workers_per_file,
                    batch_size,
                    node_id_col,
                    feat_cols if isinstance(feat_cols, list) else [feat_cols],
                    data_queues[i],
                )
            )
            for _ in range(workers_per_file):
                res.append(executor.submit(process_batch, data_queues[i], res_lists[i]))
        sorted_target_id, reverse_ind = torch.sort(target_id)
        utils.check_exception(res)
    node_id = np.concatenate(
        [node_id for res_list in res_lists for node_id, _ in res_list]
    )
    node_id = torch.from_numpy(node_id)
    node_id, ind = torch.sort(node_id)
    all_node_id = [None] * dist.get_world_size(group)
    dist.all_gather_object(all_node_id, node_id, group)
    needed_nid = []  # node id needed from each rank
    for i in range(dist.get_world_size(group)):
        res = utils.fast_isin(sorted_target_id.numpy(), all_node_id[i].numpy())
        needed_nid.append(all_node_id[i][res])
    # node sent to each rank
    sent_nid, handles = utils.all_to_all(needed_nid, is_async=True, group=group)
    feats = torch.cat([feat for res_list in res_lists for _, feat in res_list])
    # shuffle the feats based on ind
    feats = feats[ind]
    send_tensors = [None] * dist.get_world_size(group)
    for i in handles.as_completed():
        local_idx = torch.searchsorted(node_id, sent_nid[i])
        send_tensors[i] = feats[local_idx]
        sent_nid[i] = None
    recv_tensors, handles = utils.all_to_all(send_tensors, is_async=True, group=group)
    # compute the indices of the received tensors in target_id
    indices = []
    for i in range(dist.get_world_size(group)):
        ind = torch.searchsorted(sorted_target_id, needed_nid[i])
        indices.append(reverse_ind[ind])
    # create the final tensor
    out_shape = [len(target_id)] + list(feats.shape[1:])
    out_tensor = torch.empty(out_shape, dtype=feats.dtype)
    # rearrange feat based on the order of target_id
    for i in handles.as_completed():
        out_tensor[indices[i]] = recv_tensors[i]
        recv_tensors[i] = None
    return out_tensor

def load_node_feat_partitioned_file_wo_redistribution(
    node_files: List[pq.ParquetFile],
    feat_cols: Union[str, List[str]],
    node_id_col: str,
    target_id: torch.Tensor,
    batch_size: int = 8192,
    group: dist.ProcessGroup = None,
) -> torch.Tensor:
    def process_batch(data_queue, res_list):
        while True:
            data = data_queue.get()
            if data is None:
                break
            batch_id, node_ids, *feats = data
            node_ids = node_ids.to_numpy(zero_copy_only=False)
            feats_list = []
            for feat in feats:
                feat = feat.to_numpy(zero_copy_only=False).tolist()
                feats_list.append(utils.fast_vstack(feat))
            res_list[batch_id] = (node_ids, torch.hstack(feats_list))

    num_files = len(node_files)
    data_queues = [queue.Queue(maxsize=64) for _ in node_files]
    num_batches = [math.ceil(f.metadata.num_rows / batch_size) for f in node_files]
    res_lists = [[None] * num_batches[i] for i in range(num_files)]
    workers_per_file = max(1, os.cpu_count() // 2 // num_files)
    with futures.ThreadPoolExecutor(num_files * (workers_per_file + 1)) as executor:
        res = []
        for i in range(num_files):
            res.append(
                executor.submit(
                    read_batches,
                    node_files[i],
                    workers_per_file,
                    batch_size,
                    node_id_col,
                    feat_cols if isinstance(feat_cols, list) else [feat_cols],
                    data_queues[i],
                )
            )
            for _ in range(workers_per_file):
                res.append(executor.submit(process_batch, data_queues[i], res_lists[i]))
        sorted_target_id, reverse_ind = torch.sort(target_id)
        utils.check_exception(res)
    node_id = np.concatenate(
        [node_id for res_list in res_lists for node_id, _ in res_list]
    )
    node_id = torch.from_numpy(node_id)
    feats = torch.cat([feat for res_list in res_lists for _, feat in res_list])
    return feats

def main(
    node_files: List[pq.ParquetFile],
    edge_files: List[pq.ParquetFile],
    node_id_col: str,
    src_id_col: str,
    dst_id_col: str,
    feature_col: str,
    feat_group: dist.ProcessGroup,
    graph_group: dist.ProcessGroup,
):
    with TIMER("Graph generation"):
        node_ids = []
        for node_file in node_files:
            id_col = node_file.read(columns=[node_id_col]).column(0).to_numpy()
            node_ids.append(id_col)
        partial_node_ids = torch.from_numpy(np.concatenate(node_ids))
        node_ids = [None] * dist.get_world_size()
        dist.all_gather_object(node_ids, partial_node_ids)
        node_ids = torch.cat(node_ids)
        del partial_node_ids

        part_rank = dist.get_rank(feat_group)
        with utils.same_random_seed():
            node_mapping = node_ids[torch.randperm(len(node_ids))]
        part_offset = utils.balanced_split(dist.get_world_size(feat_group), len(node_ids))
        local_nid = node_mapping[part_offset[part_rank] : part_offset[part_rank + 1]]

    [f.close() for f in edge_files]


        

    node_file = pq.ParquetFile(f"{args.node_file.split('.')[0]}_split.parquet")
    with TIMER("scan through"):
        num_columns = len(node_files[0].schema.names) - 1
        assert (
            dist.get_world_size(graph_group) <= num_columns
        ), "too many ranks for feature columns"

        offset = utils.balanced_split(
            dist.get_world_size(graph_group), num_columns
        ).tolist()
        feat_rank = dist.get_rank(graph_group)
        feature_cols = [
            f"{feature_col}{i}" for i in range(offset[feat_rank], offset[feat_rank + 1])
        ]
        inputs = load_node_feat_single_file(
            node_file,
            feature_cols,
            node_id_col,
            local_nid,
            # group=feat_group,
        )
        dist.barrier()


    with TIMER("w/ redistribution"):
        num_columns = len(node_files[0].schema.names) - 1
        assert (
            dist.get_world_size(graph_group) <= num_columns
        ), "too many ranks for feature columns"

        offset = utils.balanced_split(
            dist.get_world_size(graph_group), num_columns
        ).tolist()
        feat_rank = dist.get_rank(graph_group)
        feature_cols = [
            f"{feature_col}{i}" for i in range(offset[feat_rank], offset[feat_rank + 1])
        ]
        inputs = load_node_feat_partitioned_file(
            node_files,
            feature_cols,
            node_id_col,
            local_nid,
            group=feat_group,
        )
        dist.barrier()

    with TIMER("w/o redistribution"):
        num_columns = len(node_files[0].schema.names) - 1
        assert (
            dist.get_world_size(graph_group) <= num_columns
        ), "too many ranks for feature columns"

        offset = utils.balanced_split(
            dist.get_world_size(graph_group), num_columns
        ).tolist()
        feat_rank = dist.get_rank(graph_group)
        feature_cols = [
            f"{feature_col}{i}" for i in range(offset[feat_rank], offset[feat_rank + 1])
        ]
        inputs = load_node_feat_partitioned_file_wo_redistribution(
            node_files,
            feature_cols,
            node_id_col,
            local_nid,
            group=feat_group,
        )
        dist.barrier()

    [f.close() for f in node_files]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_name", type=str, required=True)
    parser.add_argument("--edge_file", type=str, required=True)
    parser.add_argument("--source_id_col", type=str, required=True)
    parser.add_argument("--dest_id_col", type=str, required=True)
    parser.add_argument("--node_file", type=str, required=True)
    parser.add_argument("--node_id_col", type=str, required=True)
    parser.add_argument("--feature_col", type=str, required=True)
    parser.add_argument("--feat_group", type=int, required=True)
    parser.add_argument("--graph_group", type=int, required=True)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--model", type=str, required=True, choices=["gcn", "gat"])
    parser.add_argument("--num_node_files", type=int, required=True)
    parser.add_argument("--num_edge_files", type=int, required=True)
    parser.add_argument(
        "--scheduler", type=str, choices=["src", "ring", "base"], default="src"
    )
    parser.add_argument("--fanout", type=int, default=50)
    args = parser.parse_args()

    dist.init_process_group(backend="gloo")
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    assert world_size == args.feat_group * args.graph_group, "world size mismatch"
    graph_groups = [
        utils.create_group(rank % args.graph_group == i)
        for i in range(args.graph_group)
    ]
    feat_groups = [
        utils.create_group(rank // args.graph_group == i)
        for i in range(args.feat_group)
    ]
    feat_group = [g for g in feat_groups if dist.get_rank(g) != -1][0]
    graph_group = [g for g in graph_groups if dist.get_rank(g) != -1][0]

    part_offset = utils.balanced_split(
        dist.get_world_size(), args.num_node_files
    ).tolist()
    node_files = [
        pq.ParquetFile(f"{args.node_file.split('.')[0]}_{i}_split.parquet")
        for i in range(part_offset[rank], part_offset[rank + 1])
    ]
    part_offset = utils.balanced_split(
        dist.get_world_size(), args.num_edge_files
    ).tolist()
    edge_files = [
        pq.ParquetFile(f"{args.edge_file.split('.')[0]}_{i}.parquet")
        for i in range(part_offset[rank], part_offset[rank + 1])
    ]

    init_timer(args)

    main(
        node_files,
        edge_files,
        args.node_id_col,
        args.source_id_col,
        args.dest_id_col,
        args.feature_col,
        feat_group,
        graph_group,
    )

    OUT_FILE.close()
