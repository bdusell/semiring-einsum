import typing

import torch

def combine(
        forward: typing.Callable,
        backward: typing.Callable) -> typing.Callable:
    r"""Combine an einsum implementation and its derivative into an
    auto-differentiable :py:class:`~torch.autograd.Function`.

    Combining separate forward and backward implementations allows more
    memory efficiency than would otherwise be possible.

    :param forward: The forward implementation.
    :param backward: The backward implementation. Its signature should be
        ``backward(equation, args, needs_grad, grad)``, and it should return a
        :py:class:`tuple` of :py:class:`~torch.Tensor` containing the gradients
        with respect to ``args``. The :math:`i`\ th gradient may be ``None`` if
        ``needs_grad[i]`` is ``False``.
    :return: The ``apply`` method of a new auto-differentiable function.
    """

    class EinsumFunction(torch.autograd.Function):

        @staticmethod
        def forward(ctx, equation, *args):
            ctx.equation = equation
            ctx.save_for_backward(*args)
            return forward(equation, *args)

        @staticmethod
        def backward(ctx, grad):
            args = ctx.saved_tensors
            needs_grad = ctx.needs_input_grad[1:]
            input_grads = backward(ctx.equation, args, needs_grad, grad)
            return (None,) + tuple(input_grads)

    return EinsumFunction.apply
