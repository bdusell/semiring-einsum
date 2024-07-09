import typing

import torch

from .equation import Equation, AutomaticBlockSize, AUTOMATIC_BLOCK_SIZE
from .extend import semiring_einsum_forward
from .utils import logical_or_in_place, logical_or_block, logical_and_in_place

def boolean_einsum_forward(
        equation: Equation,
        *args: torch.Tensor,
        block_size: typing.Union[int, AutomaticBlockSize]=AUTOMATIC_BLOCK_SIZE
    ) -> torch.Tensor:
    r"""Boolean einsum, where addition (:math:`a + b`) is replaced with logical
    or (:math:`a \vee b`) and multiplication (:math:`a \times b`) is replaced
    with logical and (:math:`a \wedge b`).

    Note that prior to PyTorch 1.2.0, the dtype :py:data:`torch.uint8` is used
    for Boolean tensors instead of :py:data:`torch.bool`.

    :param equation: A pre-compiled equation.
    :param args: Input tensors. The number of input tensors must be compatible
        with ``equation``.
    :param block_size: Block size used to control memory usage.
    :return: Output of einsum.
    """
    return semiring_einsum_forward(equation, args, block_size, _callback)

def _callback(compute_sum):
    return compute_sum(logical_or_in_place, logical_or_block, logical_and_in_place)
