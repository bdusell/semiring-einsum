import typing

import torch

from .extend import semiring_einsum_forward, EquationForForward

def logspace_viterbi_einsum_forward(
        equation: EquationForForward,
        *args: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.LongTensor]:
    r"""Einsum where addition :math:`a + b` is replaced with
    :math:`(\max(a, b), \arg \max(a, b))`, and multiplication
    :math:`a \times b` is replaced with addition :math:`a + b`.

    :param equation: A pre-compiled equation.
    :param args: Input tensors. The number of input tensors must be compatible
        with ``equation``.
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
    return semiring_einsum_forward(equation, args, _callback)

def _callback(compute_sum):
    return compute_sum(
        _viterbi_max_in_place,
        _add_in_place,
        _viterbi_max_init,
        include_indexes=True)

def _add_in_place(a, b):
    a.add_(b)

def _viterbi_max_init(a):
    a, a_indexes = a
    index_tensor_size = list(a.size())
    index_tensor_size.append(len(a_indexes))
    # First index is always zeros.
    index_tensor = a.new_zeros(index_tensor_size, dtype=torch.long)
    return a, index_tensor

def _viterbi_max_in_place(a, b):
    # a : X1 x ... x Xn
    # a_index_tensor : X1 x ... x Xn x K
    # b : X1 x ... x Xn
    # b_indexes : K x [int]
    a, a_index_tensor = a
    b, b_indexes = b
    # Get a mask for elements where a < b.
    # a_is_less : X1 x ... x Xn
    a_is_less = torch.lt(a, b)
    # Replace elements in a with the new maximum.
    a[:] = torch.where(a_is_less, b, a)
    # Replace elements in the argmax tensor with the updated index.
    n = a.dim()
    K = len(b_indexes)
    view_size = [1] * n
    view_size.append(K)
    expand_size = list(a_is_less.size())
    expand_size.append(-1)
    # b_index_tensor : 1 x ... x 1 (n times) x K
    b_index_tensor = (
        torch.tensor(b_indexes, device=b.device)
            .view(view_size)
            .expand(expand_size))
    # a_is_less.unsqueeze(-1) : X1 x ... x Xn x 1
    # Unfortunately there is no in-place version of where() (yet).
    a_index_tensor[:] = torch.where(
        a_is_less.unsqueeze(-1),
        b_index_tensor,
        a_index_tensor)
    # This would be a hacky but correct method of doing the same thing
    # in-place:
    # a_index_tensor.masked_scatter_(a_is_less.unsqueeze(-1), b_index_tensor)
