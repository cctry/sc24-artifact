import argparse
from contextlib import contextmanager
import datetime
import math
import os
import sys
import time
from typing import List

import dgl
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist

from deal import dist_models, utils
from deal.feat_comm import FeatCommunicator
from deal.generation import generate_graph
from deal.load_node_feat import load_node_feat_partitioned_file
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
        part_algo = MulitpleFilePartitioner(edge_files, src_id_col, dst_id_col)
        graph, node_mapping = generate_graph(node_ids, part_algo, group=feat_group)
        local_nid = node_mapping[graph.ndata[dgl.NID][graph.ndata["inner_node"]]]
        [f.close() for f in edge_files]
        dist.barrier()

    # build up dataloader
    with TIMER("Build dataloader"):
        fanout = [args.fanout] * args.num_layers
        dataloader = BatchLoader(
            graph, fanout, 2 * 1024 * 1024, feat_group, graph_group, SCHEDULER_CLS[args.scheduler]
        )
        graph = None

        with TIMER("Load node features"):
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
            [f.close() for f in node_files]
            dist.barrier()
        dataloader.threads[dataloader.current_hop].join()

    with TIMER("Init models"):
        part_size = utils.all_gather_into_tensor(
            torch.tensor([inputs.shape[1]]), "cat", graph_group
        )
        dim = part_size.sum().item()
        if args.model == "gcn":
            model = dist_models.DistGCN(
                args.num_layers, dim, dim, dim, "right", mini_batch=True
            )
        elif args.model == "gat":
            heads = [4] * args.num_layers
            model = dist_models.DistGAT(
                args.num_layers, dim, dim, dim, heads, mini_batch=True
            )
        else:
            raise NotImplementedError
        with utils.same_random_seed():
            model.reset_parameters()

    with torch.no_grad():
        model.eval()
        for i in range(model.n_layer):
            with TIMER(f"GNN computation layer {i+1}"):
                outputs, part_size = model.infer(
                    dataloader.sampled_graphs[i],
                    inputs,
                    i,
                    dataloader,
                    part_size,
                    graph_group,
                )
                inputs = outputs

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
