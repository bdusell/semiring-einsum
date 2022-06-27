import typing

import torch

def combine(
        forward: typing.Callable,
        backward: typing.Callable,
        forward_options: typing.Tuple[str]=(),
        backward_options: typing.Tuple[str]=()
    ) -> typing.Callable:
    r"""Combine an einsum implementation and its derivative into a single
    function that works with PyTorch's autograd mechanics.

    Combining separate forward and backward implementations allows more
    memory efficiency than would otherwise be possible.

    :param forward: The forward implementation.
    :param backward: The backward implementation. Its signature must be
        ``backward(equation, args, needs_grad, grad, block_size)``, and it
        must return a :py:class:`tuple` of :py:class:`~torch.Tensor`
        containing the gradients with respect to ``args``. The :math:`i`\ th
        gradient may be ``None`` if ``needs_grad[i]`` is ``False``.
    :param forward_options: A list of optional keyword arguments that should
        be passed to the forward function.
    :param backward_options: A list of optional keyword arguments that should
        be passed to the backward function.
    :return: A function whose return value is compatible with PyTorch's
        autograd mechanics.
    """

    # The Function API has been available since PyTorch 0.2.0.
    class EinsumFunction(torch.autograd.Function):

        @staticmethod
        def forward(ctx, equation, forward_kwargs, backward_kwargs, *args):
            ctx.equation = equation
            ctx.backward_kwargs = backward_kwargs
            ctx.save_for_backward(*args)
            return forward(equation, *args, **forward_kwargs)

        @staticmethod
        def backward(ctx, grad):
            args = ctx.saved_tensors
            needs_grad = ctx.needs_input_grad[3:]
            input_grads = backward(ctx.equation, args, needs_grad, grad, **ctx.backward_kwargs)
            return (None, None, None, *input_grads)

    def result(equation, *args, block_size, **kwargs):
        forward_kwargs = take_kwargs(forward_options, kwargs)
        backward_kwargs = take_kwargs(backward_options, kwargs)
        for k in kwargs:
            raise TypeError(f'got an unexpected keyword argument {k!r}')
        forward_kwargs['block_size'] = backward_kwargs['block_size'] = block_size
        return EinsumFunction.apply(equation, forward_kwargs, backward_kwargs, *args)

    return result

def take_kwargs(keys, kwargs):
    result = {}
    for k in keys:
        if k in kwargs:
            result[k] = kwargs.pop(k)
    return result
