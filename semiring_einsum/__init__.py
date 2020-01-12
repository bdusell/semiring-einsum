from .equation import compile_equation
from .extend import semiring_einsum_forward
from .function import combine
from .real_forward import real_einsum_forward
from .real_backward import real_einsum_backward
from .logspace_forward import logspace_einsum_forward
from .logspace_backward import logspace_einsum_backward
from .logspace_viterbi_forward import logspace_viterbi_einsum_forward

einsum = combine(real_einsum_forward, real_einsum_backward)
r"""Differentiable version of real-space einsum.

This combines :py:func:`real_einsum_forward` and
:py:func:`real_einsum_backward` into one auto-differentiable
:py:class:`~torch.autograd.Function`.
"""

logspace_einsum = combine(logspace_einsum_forward, logspace_einsum_backward)
r"""Differentiable version of logspace einsum.

This combines :py:func:`logspace_einsum_forward` and
:py:func:`logspace_einsum_backward` into one auto-differentiable
:py:class:`~torch.autograd.Function`.
"""

__all__ = [
    'compile_equation',
    'real_einsum_forward',
    'real_einsum_backward',
    'einsum',
    'logspace_einsum_forward',
    'logspace_einsum_backward',
    'logspace_einsum',
    'logspace_viterbi_einsum_forward',
    'semiring_einsum_forward',
    'combine'
]
