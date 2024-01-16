import os
import threading
import time
import traceback
from concurrent import futures
from contextlib import contextmanager
from typing import List, Optional, Tuple, Union

import numba
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load

# Load the C++ extension module "node_map"
node_map = load(
    name="node_map",
    sources=[f"{os.path.dirname(__file__)}/node_map.cpp"],
    extra_cflags=["-O3", "-march=native", "-fopenmp"],
)

def indexed_add(out_data, out_index, in_data, in_index):
    """
    Adds the values of `in_data` to `out_data` at the indices specified by `in_index` and `out_index`.

    Args:
        out_data (torch.Tensor): The output tensor to which the values will be added.
        out_index (torch.Tensor): The indices in `out_data` where the values will be added.
        in_data (torch.Tensor): The input tensor containing the values to be added.
        in_index (torch.Tensor): The indices in `in_data` from where the values will be added.

    Returns:
        torch.Tensor: The updated `out_data` tensor.
    """
    assert out_data.dtype == in_data.dtype == torch.float32
    assert out_index.dtype == in_index.dtype == torch.int64
    assert out_data.ndim == in_data.ndim == 2
    assert out_data.shape[1] == in_data.shape[1]
    node_map.indexed_add(out_data, out_index, in_data, in_index)

def balanced_split(
    num_parts: int, num_elements: int, per_part_size=False
) -> torch.Tensor:
    """
    Splits a number of elements evenly across a given number of parts.

    Parameters
    ----------
    num_parts : int
        The number of parts to split the elements across.

    num_nodes : int
        The total number of elements to split.

    Returns
    -------
    torch.Tensor
        A tensor containing the offsets of each partition.

    Notes
    -----
    The returned tensor contains the offsets of each partition, which can be used to
    index into a larger tensor to extract the elements for each partition.
    """
    # assert num_parts <= num_elements, f"{num_parts} > {num_elements}"
    part_size = torch.full((num_parts,), num_elements // num_parts, dtype=torch.int64)
    residual = num_elements % num_parts
    if residual > 0:
        part_size[:residual] += 1
    if per_part_size:
        return part_size
    part_offset = torch.cat([torch.tensor([0]), part_size.cumsum(dim=0)])
    return part_offset


def async_sort(
    tensor: torch.Tensor,
) -> Tuple[threading.Thread, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Sorts a tensor asynchronously using a separate thread.

    Args:
        t: The tensor to be sorted.

    Returns:
        A tuple containing the sort thread and a tuple of the sorted tensor and its indices.
    """
    sorted_t = torch.empty_like(tensor)
    indices = torch.empty_like(tensor)
    sort_thread = threading.Thread(
        target=lambda x, sorted_x, indices: torch.sort(x, out=(sorted_x, indices)),
        args=(tensor, sorted_t, indices),
    )
    sort_thread.start()
    return sort_thread, (sorted_t, indices)


bool_t = np.dtype(bool)
ro_arr_t = numba.types.Array(numba.types.int64, 1, "A", readonly=True)


@numba.jit((ro_arr_t, ro_arr_t), nogil=True, nopython=True)
def fast_isin(search_target: np.ndarray, inputs: np.ndarray) -> np.ndarray:
    """
    Fast implementation of np.isin for sorted arrays.
    Equivalent to `np.isin(inputs, search_target)`.

    Args:
        search_target: The sorted array to search in.
        inputs: The array of values to search for.

    Returns:
        A boolean array indicating whether each element of `inputs` is in `search_target`.
    """
    indices = np.searchsorted(search_target, inputs)
    found = indices < len(search_target)
    mask = np.zeros_like(inputs, dtype=bool_t)
    mask[found] = search_target[indices[found]] == inputs[found]
    return mask


def sorted_count(
    content: torch.Tensor, target: torch.Tensor, target_space: int = -1
) -> torch.Tensor:
    """
    Count the apperaence of each element in `target` in `content`.

    Args:
        content (torch.Tensor): The tensor to search in.
        target (torch.Tensor): The tensor to search for.
        target_space (int): The size of return tensor. If -1, the size is the same as `target`.

    Returns:
        A tensor containing the counts of each element in `target` in `content`.
        If target_space is positive, the returned tensor contains the counts of elements not in `target` but within (0, `target_space-1`).
        The target tensor is assumed to be sorted and have no duplicates.
    """
    counts = torch.bincount(
        content,
        minlength=target_space
        if target_space > 0
        else max(content.max(), target.max()) + 1,
    )
    if target_space < 0:
        out = counts[target]
    else:
        out = torch.zeros(target_space, dtype=content.dtype)
        out[target] = counts[target]
    return out
    # assert content.dtype == target.dtype == torch.int64
    # assert not content.is_cuda and not target.is_cuda
    # assert content.ndim == target.ndim == 1
    # assert content.numel() > 0 and target.numel() > 0
    # return node_map.sorted_count(content, target, target_space)


def cat_unique(tensor_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Equvalent to `torch.cat(tensor_list).unique()`.

    Args:
        tensor_list: The list of tensors to concatenate.

    Returns:
        A tensor containing the concatenated unique elements of the input tensors.
    """
    return node_map.cat_unique(tensor_list)


def cat_searchsorted(
    sorted_tensor: torch.Tensor, tensor_list: List[torch.Tensor]
) -> torch.Tensor:
    """
    Equivalent to `torch.searchsorted(sorted_tensor, torch.cat(tensor_list))`.

    Args:
        sorted_tensor: The tensor to search for.
        tensor_list: The list of tensors to concatenate.

    Returns:
        A tensor containing the indices of elements in `sorted_tensor`.
    """
    return node_map.cat_searchsorted(sorted_tensor, tensor_list)


def check_exception(future_to_batch):
    """
    Checks for exceptions in a batch of futures.

    Args:
        future_to_batch: A dictionary mapping futures to their corresponding batches.
    """
    for future in futures.as_completed(future_to_batch):
        try:
            _ = future.result()
        except Exception as exception:  # pylint: disable=broad-except
            print("Catched exception:", type(exception), exception)
            traceback.print_exc()
            raise exception


NP_TH_DTYPE = {
    bool: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}


def fast_vstack(arrays: List[np.ndarray]) -> torch.Tensor:
    """
    Equivalent to `np.vstack(arrays)` bur returns a torch tensor.
    """
    dtype = arrays[0].dtype.type
    return node_map.vstack(arrays, NP_TH_DTYPE[dtype])


def all_gather_into_tensor(
    in_tensor: torch.Tensor, method="stack", group: Optional[dist.ProcessGroup] = None
) -> torch.Tensor:
    """
    Gathers a tensor from all processes and stack them into a single tensor.

    Args:
        in_tensor: The input tensor to gather.

        method: The method to gather the tensor. 'stack' or 'cat'.

    Returns:
        A tensor containing the stacked input tensors from all processes.
    """
    world_size = dist.get_world_size(group)
    out_shape = [world_size] + list(in_tensor.shape)
    out_tensor = torch.empty(
        out_shape,
        dtype=in_tensor.dtype,
        device=in_tensor.device,
    )
    dist.all_gather(
        list(out_tensor.chunk(world_size, 0)), in_tensor.unsqueeze(0), group
    )
    if method == "cat":
        out_tensor = out_tensor.view(-1, *in_tensor.shape[1:])
    return out_tensor


def compute_part2size(
    in_tensors: List[torch.Tensor], group: Optional[dist.ProcessGroup] = None
) -> List[List[torch.Size]]:
    """
    Computes the size of each tensor in `in_tensors` for all ranks.

    Args:
        in_tensors: A list of input tensors.

    Returns:
        A list of lists, where `part2size[i][j]` is the size of the tensor that rank `i` sends to rank `j`.
    """
    part2size = [None] * dist.get_world_size(group)
    dist.all_gather_object(
        part2size, [in_tensor.shape for in_tensor in in_tensors], group
    )
    return part2size



class MPIAllToAllFuture:
    def __init__(self, req, group) -> None:
        self.req = req
        self.group = group

    def wait(self):
        if not self.req.is_completed():
            self.req.wait()

    def wait_recv(self):
        if not self.req.is_completed():
            self.req.wait()

    def wait_send(self):
        if not self.req.is_completed():
            self.req.wait()

    def as_completed(self, wait=True):
        world_size = dist.get_world_size(self.group)
        rank = dist.get_rank(self.group)
        yield rank
        self.wait()
        for i in range(1, world_size):
            recv_rank = (rank + world_size - i) % world_size
            yield recv_rank


def mpi_all_to_all_impl(
    in_tensors: List[torch.Tensor],
    part2size: Optional[List[List[torch.Size]]],
    is_async: bool,
    group: Optional[dist.ProcessGroup],
) -> Union[Tuple[List[torch.Tensor], MPIAllToAllFuture], List[torch.Tensor]]:
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    buffer = [None] * world_size
    for recv_rank in range(world_size):
        if recv_rank == rank:
            buffer[rank] = in_tensors[rank]
        else:
            buffer[recv_rank] = torch.empty(
                part2size[recv_rank][rank],
                dtype=in_tensors[recv_rank].dtype,
                device=in_tensors[recv_rank].device,
            )
    req = dist.all_to_all(buffer, in_tensors, group, async_op=is_async)
    if is_async:
        return buffer, MPIAllToAllFuture(req, group)
    return buffer

class AllToAllFuture:
    def __init__(self, reqs, group, part2size) -> None:
        self.reqs = reqs
        self.group = group
        self.part2size = part2size

    def wait(self):
        for req in self.reqs:
            req.wait()

    def wait_recv(self):
        for req in self.reqs[1::2]:
            req.wait()

    def wait_send(self):
        for req in self.reqs[::2]:
            req.wait()

    def as_completed(self, wait=True):
        world_size = dist.get_world_size(self.group)
        rank = dist.get_rank(self.group)
        yield rank

        for i, req in zip(range(1, world_size), self.reqs[1::2]):
            recv_rank = (rank + world_size - i) % world_size

        # req_ptr = 0
        # for recv_rank in range(world_size):
        #     if recv_rank == rank:
        #         continue
        #     req = self.reqs[req_ptr*2 + 1]
        #     req_ptr += 1

            recv_size = torch.prod(torch.tensor(self.part2size[recv_rank][rank]))
            if recv_size.item() != 0:
                req.wait()
            yield recv_rank
        if wait:
            self.wait_send()

def gloo_all_to_all_impl(
    in_tensors: List[torch.Tensor],
    part2size: Optional[List[List[torch.Size]]],
    is_async: bool,
    group: Optional[dist.ProcessGroup],
) -> Union[Tuple[List[torch.Tensor], AllToAllFuture], List[torch.Tensor]]:
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    op_list = []
    buffer = [None] * world_size

    buffer[rank] = in_tensors[rank]

    for i in range(1, world_size):
        send_rank = (rank + i) % world_size
        recv_rank = (rank + world_size - i) % world_size

    # for i in range(0, world_size):
    #     send_rank = recv_rank = i
    #     if i == rank: # This fix the crash
    #        buffer[i] = in_tensors[i]
    #        continue

        send_rank_g = dist.get_global_rank(group, send_rank) if group else send_rank
        recv_rank_g = dist.get_global_rank(group, recv_rank) if group else recv_rank
        op_list.append(
            dist.P2POp(
                dist.isend,
                in_tensors[send_rank].contiguous(),
                send_rank_g,
                group,
            )
        )
        buffer[recv_rank] = torch.empty(
            part2size[recv_rank][rank],
            dtype=in_tensors[recv_rank].dtype,
            device=in_tensors[recv_rank].device,
        )
        op_list.append(
            dist.P2POp(
                dist.irecv,
                buffer[recv_rank],
                recv_rank_g,
                group,
            )
        )
    if world_size != 1:
        reqs = dist.batch_isend_irecv(op_list)
    else:
        reqs = []
    future = AllToAllFuture(reqs, group, part2size)
    if is_async:
        return buffer, future
    future.wait()
    return buffer


def all_to_all(
    in_tensors: List[torch.Tensor],
    part2size: Optional[List[List[torch.Size]]] = None,
    is_async: bool = False,
    group: Optional[dist.ProcessGroup] = None,
) -> Union[Tuple[List[torch.Tensor], AllToAllFuture], List[torch.Tensor]]:
    """
    Performs an all-to-all communication between all ranks using batch_isend_irecv.

    Args:
        in_tensors: A list of input tensors to send to all ranks.

    Note:
        part2size is a PxP 2D-List where P is the number of partitions
        part2size[i][j] is the size of the tensor that rank i sends to rank j

        All tensor types are assumed to be the same.

    Returns:
        A list of output tensors received from all ranks.
    """
    world_size = dist.get_world_size(group)
    assert (
        len(in_tensors) == world_size
    ), f"len(in_tensors)={len(in_tensors)} != {world_size}"
    if dist.get_backend(group) == dist.Backend.NCCL:
        assert all(
            in_tensor.is_cuda for in_tensor in in_tensors
        ), "Input tensors must be CUDA tensors"
    if part2size is None:
        part2size = compute_part2size(in_tensors, group)

    if dist.get_backend(group) == dist.Backend.MPI:
        return mpi_all_to_all_impl(in_tensors, part2size, is_async, group)
    if dist.get_backend(group) == dist.Backend.GLOO:
        return gloo_all_to_all_impl(in_tensors, part2size, is_async, group)
    if dist.get_backend(group) == dist.Backend.NCCL:  # use manual all-to-all for NCCL
        return gloo_all_to_all_impl(in_tensors, part2size, is_async, group)
    raise NotImplementedError


def create_group(pred: bool) -> dist.ProcessGroup:
    """
    Creates a new process group based on a boolean predicate.

    Args:
        pred: A boolean predicate used to determine if the current rank is in the new group.

    Returns:
        A new process group containing all ranks that satisfy the predicate.
    """
    rank_mask = torch.zeros(dist.get_world_size(), dtype=torch.uint8)
    rank_mask[dist.get_rank()] = pred
    rank_mask = all_gather_into_tensor(rank_mask)
    rank_mask = torch.where(rank_mask.any(0))[0]
    return dist.new_group(rank_mask.tolist())


@contextmanager
def same_random_seed(group: Optional[dist.ProcessGroup] = None) -> None:
    """
    Operate with the same random seed for all processes in the group.

    Parameters
    ----------
    group : dist.ProcessGroup
        The process group to set the random seed for.
    """
    origin_seed = torch.initial_seed()
    seed_tensor = torch.tensor([int(time.time())], dtype=torch.int64)
    dist.broadcast(seed_tensor, src=0, group=group)
    torch.manual_seed(seed_tensor.item())
    try:
        yield
    finally:
        torch.manual_seed(origin_seed)


def pad_and_cat(
    tensors: List[torch.Tensor], append_len: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pads a list of tensors and concatenates them along the first dimension.

    Args:
        tensors (List[torch.Tensor]): The list of tensors to pad and concatenate.
        append_len (bool, optional): Whether to append the lengths of the tensors as an extra element. 
            Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the padded tensor and the lengths of the tensors.
    """
    assert not append_len or tensors[0].dtype == torch.int64
    lengths = torch.tensor([t.size(0) for t in tensors])
    max_length = lengths.max().item()
    if append_len:
        max_length = max_length + 1
    padded = torch.empty(
        (len(tensors), max_length), device=tensors[0].device, dtype=tensors[0].dtype
    )
    for i, t in enumerate(tensors):
        padded[i, : t.size(0)] = t
        if append_len:
            padded[i, -1] = lengths[i]
    return padded, lengths
