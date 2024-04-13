from typing import Optional

import torch
import torch.distributed as dist
from torch import nn

from .. import utils

indexed_add = utils.indexed_add

def dist_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    part_size: Optional[torch.Tensor] = None,
    group: Optional[dist.ProcessGroup] = None,
):
    """
    Distributed matrix multiplication of two tensors `a` and `b`.

    Args:
        a (torch.Tensor): The first matrix to multiply. This matrix should be partitioned.
        b (torch.Tensor): The second matrix to multiply.
        part_size (Optional[torch.Tensor], optional): The size of each partition. It will be calculated if None.
        group (Optional[dist.ProcessGroup], optional): The process group used to communicate.

    Returns:
        torch.Tensor: The result of the matrix multiplication.
        torch.Tensor: The new part size.
    """
    assert a.dim() == 2, "a should be a matrix"
    assert b.dim() == 2, "b should be a matrix"
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    if world_size == 1:
        c = torch.mm(a, b)
        return c, torch.tensor([c.shape[1]])

    a_split = a.tensor_split(world_size, dim=0)
    if part_size is None:
        part2size = utils.compute_part2size(a_split, group=group)
        part_size = torch.tensor([part2size[i][rank][1] for i in range(world_size)])
    else:
        part2size = [[None] * world_size for _ in range(world_size)]
        for i in range(world_size):
            part2size[i][rank] = torch.Size((a_split[rank].shape[0], part_size[i]))
    a_gather, future1 = utils.all_to_all(a_split, part2size, is_async=True, group=group)
    b_split = b.split(part_size.tolist(), dim=0)
    # transpose to make c_split contiguous
    c_t = torch.zeros(
        (b.shape[1], a_split[rank].shape[0]), device=a.device, dtype=a.dtype
    )
    for i in future1.as_completed(False):
        c_t.addmm_(b_split[i].t(), a_gather[i].t())
        a_gather[i] = None
    c_split = c_t.tensor_split(world_size, dim=0)
    part2size = [[None] * world_size for _ in range(world_size)]
    for i in range(world_size):
        part2size[i][rank] = torch.Size((c_split[rank].shape[0], a_split[i].shape[0]))
    c_gather, future2 = utils.all_to_all(c_split, part2size, is_async=True, group=group)
    c_ = torch.empty(
        (a.shape[0], c_split[rank].shape[0]), device=a.device, dtype=a.dtype
    )
    offset = torch.cumsum(torch.tensor([0] + [a.shape[0] for a in a_split]), dim=0)
    new_part_size = torch.tensor([c.shape[0] for c in c_split])
    for i in future2.as_completed():
        c_[offset[i] : offset[i + 1], :] = c_gather[i].t()
        c_gather[i] = None
    future1.wait_send()
    return c_, new_part_size


class dist_Linear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.new_part_size = None

    def forward(
        self,
        input: torch.Tensor,
        part_size: Optional[torch.Tensor] = None,
        group: Optional[dist.ProcessGroup] = None,
    ):
        if input.ndim > 2:
            input_ = input.view(-1, input.shape[-1])
        else:
            input_ = input
        out, self.new_part_size = dist_mm(input_, self.weight.t(), part_size, group)
        rank = dist.get_rank(group)
        if self.bias is not None:
            begin = self.new_part_size[:rank].sum()
            end = begin + self.new_part_size[rank]
            out += self.bias[begin:end]
        out_shape = input.shape[:-1] + (self.new_part_size[rank],)
        return out.view(out_shape)


def to_dist_linear(module: nn.Linear):
    dist_lin = dist_Linear(
        module.in_features,
        module.out_features,
        module.bias is not None,
        module.weight.device,
        module.weight.dtype,
    )
    dist_lin.weight = module.weight
    if module.bias is not None:
        dist_lin.bias = module.bias
    return dist_lin
