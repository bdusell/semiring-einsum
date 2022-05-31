from .equation import compile_equation
from .extend import semiring_einsum_forward
from .function import combine
from .real_forward import real_einsum_forward
from .real_backward import real_einsum_backward
from .log_forward import log_einsum_forward
from .log_backward import log_einsum_backward
from .log_viterbi_forward import log_viterbi_einsum_forward

einsum = combine(real_einsum_forward, real_einsum_backward)
r"""Differentiable version of ordinary (real) einsum.

This combines :py:func:`real_einsum_forward` and
:py:func:`real_einsum_backward` into one auto-differentiable function.
"""

log_einsum = combine(log_einsum_forward, log_einsum_backward,
    backward_options=('grad_of_neg_inf',))
r"""Differentiable version of log-space einsum.

This combines :py:func:`log_einsum_forward` and
:py:func:`log_einsum_backward` into one auto-differentiable function. It
accepts all options to either function.
"""

__all__ = [
    'compile_equation',
    'real_einsum_forward',
    'real_einsum_backward',
    'einsum',
    'log_einsum_forward',
    'log_einsum_backward',
    'log_einsum',
    'log_viterbi_einsum_forward',
    'semiring_einsum_forward',
    'combine'
]
