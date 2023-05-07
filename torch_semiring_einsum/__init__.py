import torch

from .equation import (
    AutomaticBlockSize,
    AUTOMATIC_BLOCK_SIZE,
    compile_equation,
    Equation
)
from .extend import semiring_einsum_forward
from .function import combine
from .real_forward import real_einsum_forward
from .real_backward import real_einsum_backward
from .real_differentiable import einsum
from .log_forward import log_einsum_forward
from .log_backward import log_einsum_backward
from .log_differentiable import log_einsum
from .log_viterbi_forward import log_viterbi_einsum_forward

# Note that module variables need to be documented in this file, or else
# Sphinx won't pick them up.
# See https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#directive-automodule

AUTOMATIC_BLOCK_SIZE = AUTOMATIC_BLOCK_SIZE
r"""Use this as ``block_size`` to determine block size automatically based on
available memory, according to the default arguments for
:py:func:`AutomaticBlockSize.__init__`."""

__all__ = [
    'compile_equation',
    'Equation',
    'einsum',
    'real_einsum_forward',
    'real_einsum_backward',
    'log_einsum',
    'log_einsum_forward',
    'log_einsum_backward',
    'log_viterbi_einsum_forward',
    'AUTOMATIC_BLOCK_SIZE',
    'AutomaticBlockSize',
    'semiring_einsum_forward',
    'combine'
]
