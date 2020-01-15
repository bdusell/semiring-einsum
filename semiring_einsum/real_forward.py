import torch

from .equation import Equation
from .extend import semiring_einsum_forward

def real_einsum_forward(
        equation: Equation,
        *args: torch.Tensor,
        block_size : int) -> torch.Tensor:
    r"""Einsum where addition and multiplication have their usual meanings.

    This function has different memory and runtime characteristics than
    :py:func:`torch.einsum`, which can be tuned with ``block_size``. Higher
    values of ``block_size`` result in faster runtime and higher memory usage.

    In some cases, when dealing with summations over more than two input
    tensors at once, this implementation can have better space complexity than
    :py:func:`torch.einsum`.

    :param equation: A pre-compiled equation.
    :param args: Input tensors. The number of input tensors must be compatible
        with ``equation``.
    :param block_size: Block size used to control memory usage.
    :return: Output of einsum.
    """
    return semiring_einsum_forward(equation, args, block_size, _callback)

def _callback(compute_sum):
    return compute_sum(_add_in_place, _sum_block, _multiply_in_place)

def _add_in_place(a, b):
    a.add_(b)

def _sum_block(a, dims):
    return torch.sum(a, dim=dims)

def _multiply_in_place(a, b):
    a.mul_(b)
