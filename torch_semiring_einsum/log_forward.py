import typing

import torch

from .equation import Equation, AutomaticBlockSize, AUTOMATIC_BLOCK_SIZE
from .extend import (
    semiring_einsum_forward,
    semiring_einsum_forward_impl
)
from .utils import (
    max_in_place,
    max_block,
    add_in_place,
    sum_block,
    clip_inf_in_place
)

def log_einsum_forward(
        equation: Equation,
        *args: torch.Tensor,
        block_size: typing.Union[int, AutomaticBlockSize]=AUTOMATIC_BLOCK_SIZE,
        return_max: bool=False,
        return_sumexpsub: bool=False
    ) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor]]:
    r"""Log-space einsum, where addition :math:`a + b` is replaced with
    :math:`\log(\exp a + \exp b)`, and multiplication :math:`a \times b` is
    replaced with addition :math:`a + b`.

    :param equation: A pre-compiled equation.
    :param args: Input tensors. The number of input tensors must be compatible
        with ``equation``.
    :param block_size: Block size used to control memory usage.
    :param return_max: If true, also return the tensor of maximum terms, which
        can be reused when computing the gradient.
    :param return_sumexpsub: If true, also return the tensor of sums of terms
        (where the maximum term has been subtracted from each term), which can
        be reused when computing the gradient.
    :return: Output of einsum. If ``return_max`` or ``return_sumexpsub`` is
        true, the output will be a list containing the extra outputs.
    """

    def callback(compute_sum):
        # Make an initial pass to compute the maximum terms.
        # max_values has the same size as the output.
        max_values = compute_max(equation, args, block_size)
        # Now compute the logsumexp.
        # This implements y = max(x) + log \sum_i exp(x_i - max(x))
        output = compute_sumexpsub(equation, args, block_size, max_values)
        if return_sumexpsub:
            # Optionally save the sumexpsub for the backward pass.
            sumexpsub = output.clone()
        output.log_()
        output.add_(max_values)
        results = [output]
        if return_max:
            results.append(max_values)
        if return_sumexpsub:
            results.append(sumexpsub)
        if len(results) == 1:
            return results[0]
        else:
            return tuple(results)

    return semiring_einsum_forward(equation, args, block_size, callback)

def compute_max(equation, args, block_size):
    max_values = semiring_einsum_forward_impl(
        equation,
        args,
        block_size,
        args,
        add_in_place=max_in_place,
        sum_block=max_block,
        multiply_in_place=add_in_place,
        reduce_info=equation.reduce_input_to_output,
        include_indexes=False
    )
    # This will clip +inf to the max float value, and -inf to the min float
    # value. Clipping to the min/max float fixes an edge case where all terms
    # are -inf/+inf (the problem is that (-inf - -inf) or (+inf - +inf)
    # produces nan). Values of nan are left as-is, although it should be
    # harmless to replace them with 0, because the terms to the logsumexp would
    # be nan anyway, so the nans would not be silently suppressed.
    clip_inf_in_place(max_values)
    return max_values

def resize_max_values(max_values, equation):
    # Resize max_values so it can broadcast with the shape
    # output_vars + reduced_vars.
    num_reduced_vars = len(equation.reduce_input_to_output.reduced_variables)
    return max_values.view(list(max_values.size()) + [1] * num_reduced_vars)

def compute_sumexpsub(equation, args, block_size, max_values):

    resized_max_values = resize_max_values(max_values, equation)

    def sumexpsub_block(a, dims):
        a.sub_(resized_max_values)
        a.exp_()
        return sum_block(a, dims)

    return semiring_einsum_forward_impl(
        equation,
        args,
        block_size,
        args,
        add_in_place=add_in_place,
        sum_block=sumexpsub_block,
        multiply_in_place=add_in_place,
        reduce_info=equation.reduce_input_to_output,
        include_indexes=False
    )
