import typing

import torch

from .equation import Equation, AutomaticBlockSize, AUTOMATIC_BLOCK_SIZE
from .extend import semiring_einsum_forward
from .utils import add_in_place

def log_viterbi_einsum_forward(
        equation: Equation,
        *args: torch.Tensor,
        block_size: typing.Union[int, AutomaticBlockSize]=AUTOMATIC_BLOCK_SIZE
    ) -> typing.Tuple[torch.Tensor, torch.LongTensor]:
    r"""Viterbi einsum, where addition :math:`a + b` is replaced with
    :math:`(\max(a, b), \arg \max(a, b))`, and multiplication
    :math:`a \times b` is replaced with log-space multiplication
    :math:`a + b`.

    :param equation: A pre-compiled equation.
    :param args: Input tensors. The number of input tensors must be compatible
        with ``equation``.
    :param block_size: Block size used to control memory usage.
    :return: A tuple containing the max and argmax of the einsum operation.
        The first element of the tuple simply contains the maximum values
        of the terms "summed" over by einsum. The second element contains
        the values of the summed-out variables that maximized those terms.
        If the max tensor has dimension
        :math:`N_1 \times \cdots \times N_m`,
        and :math:`k` variables were summed out, then the argmax tensor has
        dimension
        :math:`N_1 \times \cdots \times N_m \times k`,
        where the :math:`(m+1)`\ th dimension is a :math:`k`-tuple of indexes
        representing the argmax. The variables in the `k`-tuple are ordered
        by first appearance in the einsum equation.
    """
    return semiring_einsum_forward(equation, args, block_size, _callback)

ARGMAX_DTYPE = torch.int64

def _callback(compute_sum):
    return compute_sum(
        viterbi_max_in_place,
        viterbi_max_block,
        add_in_place,
        include_indexes=True,
        output_dtypes=(None, ARGMAX_DTYPE))

def viterbi_max_in_place(a, b):
    # a_max : X1 x ... x Xn
    # a_argmax : X1 x ... x Xn x m
    # b_max : X1 x ... x Xn
    # b_argmax : X1 x ... x Xn x m
    a_max, a_argmax = a
    b_max, b_argmax = b
    # Get a mask for elements where a < b.
    # `torch.lt` has been available since PyTorch 0.1.12.
    # a_is_less : X1 x ... x Xn
    a_is_less = torch.lt(a_max, b_max)
    # `torch.where` was introduced in PyTorch 0.4.0.
    # Replace elements in a with the new maximum.
    a_max[...] = torch.where(a_is_less, b_max, a_max)
    # Replace elements in the argmax tensor with the updated index.
    # Unfortunately there is no in-place version of where() (yet).
    a_argmax[...] = torch.where(
        a_is_less.unsqueeze(-1),
        b_argmax,
        a_argmax)

def viterbi_max_block(a, dims, var_values):
    # Given a tensor of values `a`, return the max and argmax of `a` over
    # the dimensions specified in `dims`. Make sure to offset the argmax
    # indexes according to the ranges specified in `var_values`.
    # Let `a` have n output dimensions and m reduced dimensions.
    # a : X1 x ... x Xn x K1 x ... Km
    # The dimensions are not necessarily in that order.
    # max_values : X1 x ... x Xn
    # argmax : X1 x ... x Xn x m
    max_values, argmax = max_argmax_block(a, dims)
    # Offset the argmax indexes.
    # offset : 1 x ... x 1 (n times) x m
    n = argmax.dim() - 1
    offset_size = [1] * n
    offset_size.append(-1)
    # `.new_tensor` was introduced in PyTorch 0.4.0.
    offset = argmax.new_tensor([r.start for r in var_values]).view(offset_size)
    argmax.add_(offset)
    return max_values, argmax

def max_argmax_block(a, dims):
    # a : X1 x ... x Xn x K1 x ... Km (not necessarily in this order)
    dim_max = a
    argmaxes = []
    # Iterate over dimensions in reverse so we don't need to adjust the
    # remaining dimensions after reducing each one.
    for dim in reversed(dims):
        # `torch.max` has been available since PyTorch 0.1.12.
        # dim_max : X1 x ... x Xn x K1 x ... x Ki
        # argmaxes : m-i x [X1 x ... x Xn x K1 x ... x Ki]
        dim_max, dim_argmax = torch.max(dim_max, dim=dim)
        # dim_argmax : X1 x ... x Xn x K1 x ... x Ki-1
        argmaxes = [lookup_dim(x, dim_argmax, dim) for x in argmaxes]
        # argmaxes : m-i x [X1 x ... x Xn x K1 x ... x Ki-1]
        argmaxes.append(dim_argmax)
        # argmaxes : m-i+1 x [X1 x ... x Xn x K1 x ... x Ki-1]
    # dim_max : X1 x ... x Xn
    # argmaxes : m x [X1 x ... x Xn]
    # Remember to reverse the argmaxes, since we iterated in reverse.
    argmaxes.reverse()
    if argmaxes:
        # `torch.stack` has been available since PyTorch 0.1.12.
        argmax = torch.stack(argmaxes, dim=argmaxes[0].dim())
    else:
        # Handle the case where there are no summed variables.
        # `torch.empty` has been available since PyTorch 0.4.0.
        argmax = torch.empty(dim_max.size() + (0,), dtype=ARGMAX_DTYPE, device=a.device)
    # argmax : X1 x ... x Xn x m
    return dim_max, argmax

def lookup_dim(x, i, dim):
    # x : X1 x ... x Xdim x ... x Xn
    # i : X1 x ... x Xn, with int values in [0, dim)
    # return : X1 x ... x Xn
    # return[x1, ..., xn] = x[x1, ..., i[x1, ..., xn], ..., xn]
    index = i.unsqueeze(dim)
    # index : X1 x ... x 1 x ... x Xn
    result = torch.gather(x, dim, index)
    # result : X1 x ... x 1 x ... x Xn
    return result.squeeze(dim)
