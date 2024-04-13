import math
import os
import queue
import threading
from concurrent import futures
from typing import List, Union

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist

from . import utils


def read_batches(
    node_file: pq.ParquetFile,
    num_workers: int,
    batch_size: int,
    node_id_col: str,
    feat_cols: List[str],
    data_queue: queue.Queue,
):
    """
    Reads batches of data from a Parquet file and puts them into a queue for processing.

    Args:
        node_file (pq.ParquetFile): The Parquet file to read from.
        num_workers (int): The number of worker threads that will be processing the data.
        batch_size (int): The number of rows to read per batch.
        node_id_col (str): The name of the column containing the node IDs.
        feat_col (str): The name of the column containing the node features.
        data_queue (queue.Queue): The queue to put the data batches into.
    """
    columns = [node_id_col] + feat_cols
    for batch_id, batch in enumerate(
        node_file.iter_batches(batch_size=batch_size, columns=columns)
    ):
        node_ids = batch.column(0)
        feats = [c for c in batch.columns[1:]]
        assert len(feats) != 0, f"no feature columns {node_file.metadata} {columns}"
        data_queue.put((batch_id, node_ids, *feats))
    for _ in range(num_workers):
        data_queue.put(None, timeout=30)


def load_node_feat_single_file(
    node_file: pq.ParquetFile,
    feat_cols: Union[str, List[str]],
    node_id_col: str,
    target_id: torch.Tensor,
    batch_size: int = 8192,
) -> torch.Tensor:
    num_nodes = len(target_id)
    assert (
        num_nodes <= node_file.metadata.num_rows
    ), "target_id contains invalid node IDs"
    num_batches = math.ceil(node_file.metadata.num_rows / batch_size)
    data_queue = queue.Queue(maxsize=10)

    feats_list = [None] * num_batches
    nid_list = [None] * num_batches

    num_workers = os.cpu_count() // 2
    reader_thread = threading.Thread(
        target=read_batches,
        args=(
            node_file,
            num_workers,
            batch_size,
            node_id_col,
            feat_cols if isinstance(feat_cols, list) else [feat_cols],
            data_queue,
        ),
    )
    reader_thread.start()

    targets = np.sort(target_id.numpy())

    def process_batch():
        while True:
            data = data_queue.get()
            if data is None:
                break
            batch_id, node_ids, *feats = data
            node_ids = node_ids.to_numpy(zero_copy_only=False)
            mask = utils.fast_isin(targets, node_ids)
            mask_nz = np.nonzero(mask)[0]
            valid_feats = []
            for feat in feats:
                feat = feat.to_numpy(zero_copy_only=False)
                valid_feat = [feat[i] for i in mask_nz]
                valid_feats.append(utils.fast_vstack(valid_feat))
            feats_list[batch_id] = np.hstack(valid_feats)
            nid_list[batch_id] = node_ids[mask]

    with futures.ThreadPoolExecutor() as executor:
        future_to_batch = {
            executor.submit(process_batch): i for i in range(num_workers)
        }
        indices = np.argsort(target_id.numpy())
        utils.check_exception(future_to_batch)

    feats = np.concatenate(feats_list)
    nid = np.concatenate(nid_list)
    assert len(nid) == len(targets), f"len(nid)={len(nid)}, len(targets)={len(targets)}"
    # rearrange feat based on the order of target_id
    return torch.from_numpy(feats[indices[np.searchsorted(targets, nid)]])


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
