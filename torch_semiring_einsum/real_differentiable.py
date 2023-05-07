from .function import combine
from .real_forward import real_einsum_forward
from .real_backward import real_einsum_backward

# TODO It should be possible to avoid saving some args for the backward pass
# when needs_grad is false for other args.

einsum = combine(real_einsum_forward, real_einsum_backward)
einsum.__doc__ = \
r"""Differentiable version of ordinary (real) einsum.

This combines :py:func:`real_einsum_forward` and
:py:func:`real_einsum_backward` into one auto-differentiable function.
"""
