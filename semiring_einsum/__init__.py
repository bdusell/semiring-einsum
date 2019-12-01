from .extend import (
    semiring_einsum_forward,
    parse_equation,
    compile_equation_for_forward,
    ParsedEquation)
from .function import combine
from .real_forward import real_einsum_forward
from .logspace_forward import logspace_einsum_forward
from .logspace_backward import (
    compile_equation_for_logspace_backward,
    logspace_einsum_backward)
from .logspace_viterbi_forward import logspace_viterbi_einsum_forward

def compile_equation(
        equation: str,
        forward: bool=True,
        logspace_backward: bool=False) -> ParsedEquation:
    r"""Pre-compile an einsum equation for use with the einsum functions in
    this package.

    :param equation: An equation in einsum syntax.
    :param forward: Compile the equation for use with
        :py:func:`real_einsum_forward`,
        :py:func:`logspace_einsum_forward`,
        or any function implemented with
        :py:func:`semiring_einsum_forward`.
    :param logspace_backward: Compile the equation for use with
        :py:func:`logspace_einsum_backward` or
        :py:func:`logspace_einsum` (things that require computing the
        derivative of logspace einsum).
    :return: A pre-compiled equation.
    """
    equation = parse_equation(equation)
    if forward or logspace_backward:
        equation = compile_equation_for_forward(equation)
    if logspace_backward:
        equation = compile_equation_for_logspace_backward(equation)
    return equation

logspace_einsum = combine(logspace_einsum_forward, logspace_einsum_backward)
r"""Differentiable version of logspace einsum.

This combines :py:func:`logspace_einsum_forward` and
:py:func:`logspace_einsum_backward` into one auto-differentiable
:py:class:`~torch.autograd.Function`.
"""

__all__ = [
    'compile_equation',
    'real_einsum_forward',
    'logspace_einsum_forward',
    'logspace_einsum_backward',
    'logspace_einsum',
    'logspace_viterbi_einsum_forward',
    'semiring_einsum_forward',
    'combine'
]
