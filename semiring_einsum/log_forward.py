import torch

from .equation import Equation
from .extend import semiring_einsum_forward
from .utils import (
    max_in_place,
    max_block,
    add_in_place,
    sum_block,
    clip_max_values,
    resize_max_values
)

def log_einsum_forward(
        equation: Equation,
        *args: torch.Tensor,
        block_size : int) -> torch.Tensor:
    r"""Log-space einsum, where addition :math:`a + b` is replaced with
    :math:`\log(\exp a + \exp b)`, and multiplication :math:`a \times b` is
    replaced with addition :math:`a + b`.

    :param equation: A pre-compiled equation.
    :param args: Input tensors. The number of input tensors must be compatible
        with ``equation``.
    :param block_size: Block size used to control memory usage.
    :return: Output of einsum.
    """
    def callback(compute_sum):
        # Make an initial pass to compute the maximum terms.
        # max_values has the same size as the reduced variables.
        # TODO Add an option to save max_values for the backward pass.
        max_values = compute_sum(max_in_place, max_block, add_in_place)
        clip_max_values(max_values)
        resized_max_values = resize_max_values(
            max_values,
            len(equation.reduce_input_to_output.reduced_variables))

        def sumexpsub_block(a, dims):
            a.sub_(resized_max_values)
            a.exp_()
            return sum_block(a, dims)

        # Now compute the logsumexp.
        # This implements y = max(x) + log \sum_i exp(x_i - max(x))
        result = compute_sum(add_in_place, sumexpsub_block, add_in_place)
        result.log_()
        result.add_(max_values)
        return result

    return semiring_einsum_forward(equation, args, block_size, callback)
