import typing

import torch

from .equation import Equation, AutomaticBlockSize, AUTOMATIC_BLOCK_SIZE
from .extend import semiring_einsum_forward
from .utils import add_in_place, sum_block, multiply_in_place

def real_einsum_forward(
        equation: Equation,
        *args: torch.Tensor,
        block_size: typing.Union[int, AutomaticBlockSize]=AUTOMATIC_BLOCK_SIZE
    ) -> torch.Tensor:
    r"""Einsum where addition and multiplication have their usual meanings.

    This function has different memory and runtime characteristics than
    :py:func:`torch.einsum`, which can be tuned with ``block_size``. Higher
    values of ``block_size`` result in faster runtime and higher memory usage.

    In some cases, when dealing with summations over more than two input
    tensors at once, this implementation can have better space complexity than
    :py:func:`torch.einsum`, because it does not create intermediate tensors
    whose sizes are proportional to the dimensions being summed over.

    :param equation: A pre-compiled equation.
    :param args: Input tensors. The number of input tensors must be compatible
        with ``equation``.
    :param block_size: Block size used to control memory usage.
    :return: Output of einsum.
    """
    return semiring_einsum_forward(equation, args, block_size, _callback)

def _callback(compute_sum):
    return compute_sum(add_in_place, sum_block, multiply_in_place)
