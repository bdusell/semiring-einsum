import itertools
import typing

import torch

from .equation import Equation, get_ranges
from .extend import semiring_einsum_forward_impl, reduce_in_place, adjust_size

def log_einsum_backward(
        equation: Equation,
        args: typing.Sequence[torch.Tensor],
        needs_grad: typing.Sequence[bool],
        grad: torch.Tensor,
        block_size: int) -> typing.List[typing.Optional[torch.Tensor]]:
    r"""Compute the derivative of
    :py:func:`~semiring_einsum.log_einsum_forward`.

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
    # Z : same size as output of equation
    Z = semiring_einsum_forward_impl(
        equation,
        args,
        block_size,
        args,
        add_in_place=_add_in_place,
        sum_block=_sumexp_block,
        multiply_in_place=_add_in_place,
        reduce_info=equation.reduce_input_to_output,
        include_indexes=False)
    # C : same size as output of equation
    C = grad / Z
    del Z
    arg_grads = []
    for i, arg in enumerate(args):
        if needs_grad[i]:
            reduce_info, C_lookup_info = equation.reduce_all_to_input[i]
            var_ranges = reduce_info.get_ranges(equation, args, block_size)

            def generate_terms():
                for var_values in itertools.product(*var_ranges):

                    def generate_factors():
                        for arg, arg_info in zip(args, reduce_info.lookup_info):
                            yield arg_info.lookup(arg, var_values)

                    term_size = reduce_info.get_term_size(equation, args, var_values)
                    term = reduce_in_place(
                        _add_in_place,
                        generate_factors(),
                        lambda x: adjust_size(x, term_size))
                    term.exp_()
                    term.mul_(C_lookup_info.lookup(C, var_values))
                    yield _sum_block(term, reduce_info.reduced_dims)

            arg_grad = reduce_in_place(_add_in_place, generate_terms())
        else:
            arg_grad = None
        arg_grads.append(arg_grad)
    return arg_grads

def _add_in_place(a, b):
    a.add_(b)

def _sumexp_block(a, dims):
    a.exp_()
    return torch.sum(a, dim=dims)

def _sum_block(a, dims):
    return torch.sum(a, dim=dims)
