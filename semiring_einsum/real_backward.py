import typing

import torch

from .equation import Equation
from .extend import semiring_einsum_forward_impl
from .real_forward import _add_in_place, _sum_block, _multiply_in_place

def real_einsum_backward(
        equation: Equation,
        args: typing.Sequence[torch.Tensor],
        needs_grad: typing.Sequence[bool],
        grad: torch.Tensor,
        block_size: int) -> typing.List[typing.Optional[torch.Tensor]]:
    r"""Compute the derivative of
    :py:func:`~semiring_einsum.real_einsum_forward`.

    Like the forward pass, the backward pass is done in memory-efficient
    fashion by doing summations in-place.

    :param equation: Pre-compiled einsum equation. The derivative of the
        einsum operation specified by this equation will be computed.
        The equation must have been compiled with ``backward=True``.
    :param args: The inputs to the einsum operation whose derivative
        is being computed.
    :param needs_grad: Indicates which inputs in ``args`` require gradient.
    :param grad: The gradient of the loss function with respect to the output
        of the einsum operation.
    :param block_size: Block size used to control memory usage.
    :return: The gradients with respect to each of the inputs to the
        einsum operation. Returns ``None`` for inputs that do not require
        gradient.
    """
    # grad : same size as output of equation
    if len(args) != len(needs_grad):
        raise ValueError('length of args is not equal to length of needs_grad')
    equation.validate_sizes(args)
    equation.prepare_for_backward()
    output_size = tuple(equation.get_sizes(args, equation.output_variables))
    grad_size = tuple(grad.size())
    if grad_size != output_size:
        raise ValueError(
            'size of gradient {} does not match expected size {}'.format(
                grad_size, output_size))
    arg_grads = []
    for i, (arg, arg_needs_grad, arg_reduce_info) in enumerate(zip(
            args, needs_grad, equation.reduce_others_to_input)):
        if arg_needs_grad:
            inputs = list(args)
            inputs[i] = grad
            arg_grad = semiring_einsum_forward_impl(
                equation,
                args,
                block_size,
                inputs,
                _add_in_place,
                _sum_block,
                _multiply_in_place,
                arg_reduce_info,
                False)
        else:
            arg_grad = None
        arg_grads.append(arg_grad)
    return arg_grads
