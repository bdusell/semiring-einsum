import typing

import torch

from .equation import Equation, AutomaticBlockSize, AUTOMATIC_BLOCK_SIZE
from .extend import semiring_einsum_forward
from .utils import min_in_place, min_block, add_in_place

def min_tropical_einsum_forward(
        equation: Equation,
        *args: torch.Tensor,
        block_size: typing.Union[int, AutomaticBlockSize]=AUTOMATIC_BLOCK_SIZE
    ) -> torch.Tensor:
    r"""Min tropical einsum, where addition (:math:`a + b`) is replaced with
    min (:math:`\min(a, b)`) and multiplication (:math:`a \times b`) is
    replaced with addition (:math:`a + b`).

    :param equation: A pre-compiled equation.
    :param args: Input tensors. The number of input tensors must be compatible
        with ``equation``.
    :param block_size: Block size used to control memory usage.
    :return: Output of einsum.
    """
    return semiring_einsum_forward(equation, args, block_size, _callback)

def _callback(compute_sum):
    return compute_sum(min_in_place, min_block, add_in_place)
