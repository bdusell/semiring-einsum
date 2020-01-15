import typing

import torch

def combine(
        forward: typing.Callable,
        backward: typing.Callable) -> typing.Callable:
    r"""Combine an einsum implementation and its derivative into a single
    function that works with PyTorch's autograd mechanics.

    Combining separate forward and backward implementations allows more
    memory efficiency than would otherwise be possible.

    :param forward: The forward implementation.
    :param backward: The backward implementation. Its signature should be
        ``backward(equation, args, needs_grad, grad, block_size)``, and it
        should return a :py:class:`tuple` of :py:class:`~torch.Tensor`
        containing the gradients with respect to ``args``. The :math:`i`\ th
        gradient may be ``None`` if ``needs_grad[i]`` is ``False``.
    :return: A function whose return value is compatible with PyTorch's
        autograd mechanics.
    """

    class EinsumFunction(torch.autograd.Function):

        @staticmethod
        def forward(ctx, equation, block_size, *args):
            ctx.equation = equation
            ctx.block_size = block_size
            ctx.save_for_backward(*args)
            return forward(equation, *args, block_size=block_size)

        @staticmethod
        def backward(ctx, grad):
            args = ctx.saved_tensors
            needs_grad = ctx.needs_input_grad[2:]
            input_grads = backward(ctx.equation, args, needs_grad, grad,
                ctx.block_size)
            return (None, None, *input_grads)

    def result(equation, *args, block_size):
        return EinsumFunction.apply(equation, block_size, *args)

    return result
