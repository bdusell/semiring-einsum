import itertools
import typing

import torch

from .equation import Equation, get_ranges
from .extend import semiring_einsum_forward_impl, reduce_in_place, adjust_size
from .utils import (
    max_in_place,
    max_block,
    add_in_place,
    sum_block,
    clip_max_values,
    resize_max_values
)

def log_einsum_backward(
        equation: Equation,
        args: typing.Sequence[torch.Tensor],
        needs_grad: typing.Sequence[bool],
        grad: torch.Tensor,
        block_size: int) -> typing.List[typing.Optional[torch.Tensor]]:
    r"""Compute the derivative of
    :py:func:`~torch_semiring_einsum.log_einsum_forward`.

    Like the forward pass, the backward pass is done in memory-efficient
    fashion by doing summations in fixed-size chunks.

    :param equation: Pre-compiled einsum equation. The derivative of the
        log-space einsum operation specified by this equation will be computed.
    :param args: The inputs to the log-space einsum operation whose derivative
        is being computed.
    :param needs_grad: Indicates which inputs in ``args`` require gradient.
    :param grad: The gradient of the loss function with respect to the output
        of the log-space einsum operation.
    :param block_size: Block size used to control memory usage.
    :return: The gradients with respect to each of the inputs to the log-space
        einsum operation. Returns ``None`` for inputs that do not require
        gradient.
    """
    # grad : same size as output of equation
    if len(args) != len(needs_grad):
        raise ValueError('length of args is not equal to length of needs_grad')
    equation.validate_sizes(args)
    equation.prepare_for_forward()
    equation.prepare_for_log_backward()
    output_size = tuple(equation.get_sizes(args, equation.output_variables))
    grad_size = tuple(grad.size())
    if grad_size != output_size:
        raise ValueError(
            'size of gradient {} does not match expected size {}'.format(
                grad_size, output_size))
    # The gradient of logsumexp is softmax (logsumexp is a soft version of max,
    # and softmax is a soft version of argmax). So essentially we're computing
    # a softmax here. In order to avoid overflow in the exp() function, we need
    # to exploit the identity
    #     \frac{ \exp(x) }{ \sum_{x'} \exp(x') } =
    #         \frac{ \exp(x-c) }{ \sum_{x'} \exp(x'-c) }
    # where c = \max_{x'} x'.
    # Z is the denominator of the softmax. We first do a separate pass through
    # the inputs to compute the maximums, then we use those to compute Z and
    # later the numerators.
    # max_values : same size as output of equation
    max_values = semiring_einsum_forward_impl(
        equation,
        args,
        block_size,
        args,
        add_in_place=max_in_place,
        sum_block=max_block,
        multiply_in_place=add_in_place,
        reduce_info=equation.reduce_input_to_output,
        include_indexes=False)
    clip_max_values(max_values)
    resized_max_values = resize_max_values(
        max_values,
        len(equation.reduce_input_to_output.reduced_variables))

    def sumexpsub_block(a, dims):
        a.sub_(resized_max_values)
        a.exp_()
        return sum_block(a, dims)

    # Z : same size as output of equation
    Z = semiring_einsum_forward_impl(
        equation,
        args,
        block_size,
        args,
        add_in_place=add_in_place,
        sum_block=sumexpsub_block,
        multiply_in_place=add_in_place,
        reduce_info=equation.reduce_input_to_output,
        include_indexes=False)
    # C : same size as output of equation
    C = grad / Z
    del Z
    arg_grads = []
    for i, arg in enumerate(args):
        if needs_grad[i]:
            reduce_info, output_lookup_info = equation.reduce_all_to_input[i]
            var_ranges = reduce_info.get_ranges(equation, args, block_size)

            def generate_terms():
                for var_values in itertools.product(*var_ranges):

                    def generate_factors():
                        for arg, arg_info in zip(args, reduce_info.lookup_info):
                            yield arg_info.lookup(arg, var_values)

                    term_size = reduce_info.get_term_size(equation, args, var_values)
                    term = reduce_in_place(
                        add_in_place,
                        generate_factors(),
                        lambda x: adjust_size(x, term_size))
                    # Subtract the maximum values to avoid overflow in exp().
                    term.sub_(output_lookup_info.lookup(max_values, var_values))
                    term.exp_()
                    term.mul_(output_lookup_info.lookup(C, var_values))
                    yield sum_block(term, reduce_info.reduced_dims)

            arg_grad = reduce_in_place(add_in_place, generate_terms())
        else:
            arg_grad = None
        arg_grads.append(arg_grad)
    return arg_grads

def _sumexp_block(a, dims):
    a.exp_()
    return sum_block(a, dims)
