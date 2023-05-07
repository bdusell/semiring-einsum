import math
import typing

import torch

from .equation import Equation, AutomaticBlockSize, AUTOMATIC_BLOCK_SIZE
from .log_forward import log_einsum_forward
from .log_backward import log_einsum_backward

# The Function API has been available since PyTorch 0.2.0.
class LogEinsumFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, equation, block_size, save_max, save_sumexpsub,
            grad_of_neg_inf, *args):
        # If the gradient will not be needed, do not save any extra tensors
        # for the backward pass (this avoids some unnecessary cloning).
        # It looks like needs_input_grad is still true when some inputs have
        # needs_grad=True even when torch.no_grad() is used, but I don't know
        # a better way to detect this (torch.is_grad_enabled() is always false
        # in forward()).
        if not any(ctx.needs_input_grad):
            save_max = save_sumexpsub = False
        ctx.equation = equation
        ctx.block_size = block_size
        ctx.save_max = save_max
        ctx.save_sumexpsub = save_sumexpsub
        ctx.grad_of_neg_inf = grad_of_neg_inf
        output = log_einsum_forward(
            equation,
            *args,
            block_size=block_size,
            return_max=save_max,
            return_sumexpsub=save_sumexpsub
        )
        if save_max or save_sumexpsub:
            output, *extra = output
        else:
            extra = []
        ctx.save_for_backward(*args, *extra)
        return output

    @staticmethod
    def backward(ctx, grad):
        saved_tensors = list(ctx.saved_tensors)
        needs_grad = ctx.needs_input_grad[5:]
        if ctx.save_sumexpsub:
            saved_sumexpsub = saved_tensors.pop()
        else:
            saved_sumexpsub = None
        if ctx.save_max:
            saved_max = saved_tensors.pop()
        else:
            saved_max = None
        args = saved_tensors
        input_grads = log_einsum_backward(
            ctx.equation,
            args,
            needs_grad,
            grad,
            ctx.block_size,
            ctx.grad_of_neg_inf,
            saved_max,
            saved_sumexpsub
        )
        return (None, None, None, None, None, *input_grads)

def log_einsum(
        equation: Equation,
        *args: torch.Tensor,
        block_size: typing.Union[int, AutomaticBlockSize]=AUTOMATIC_BLOCK_SIZE,
        save_max: bool=True,
        save_sumexpsub: bool=True,
        grad_of_neg_inf=math.nan
    ) -> torch.Tensor:
    r"""Differentiable version of log-space einsum.

    This combines :py:func:`log_einsum_forward` and
    :py:func:`log_einsum_backward` into one auto-differentiable function.

    :param save_max: If true, save the tensor of maximum terms computed in the
        forward pass and reuse it in the backward pass. This tensor has the
        same size as the output tensor. Setting this to false will save memory
        but increase runtime.
    :param save_sumexpsub: If true, save the tensor of sums of terms computed
        in the forward pass and reuse it in the backward pass. This tensor has
        the same size as the output tensor. Setting this to false will save
        memory but increase runtime.
    """
    return LogEinsumFunction.apply(
        equation, block_size, save_max, save_sumexpsub, grad_of_neg_inf, *args)
