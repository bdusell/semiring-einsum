import itertools
import typing

import torch

from .extend import (
    adjust_size,
    reduce_in_place,
    EquationForBackward)

def real_einsum_backward(
        equation: EquationForBackward,
        args: typing.Sequence[torch.Tensor],
        needs_grad: typing.Sequence[bool],
        grad: torch.Tensor) -> typing.List[typing.Optional[torch.Tensor]]:
    r"""Compute the derivative of
    :py:func:`~semiring_einsum.real_einsum_forward`.

    Like the forward pass, the backward pass is done in memory-efficient
    fashion by doing summations in-place.

    :param equation: Pre-compiled einsum equation. The derivative of the
        logspace einsum operation specified by this equation will be computed.
        The equation must have been compiled with ``backward=True``.
    :param args: The inputs to the logspace einsum operation whose derivative
        is being computed.
    :param needs_grad: Indicates which inputs in ``args`` require gradient.
    :param grad: The gradient of the loss function with respect to the output
        of the einsum operation.
    :return: The gradients with respect to each of the inputs to the
        einsum operation. Returns ``None`` for inputs that do not require
        gradient.
    """
    # grad : same size as output of equation
    if not isinstance(equation, EquationForBackward):
        raise TypeError
    if len(args) != len(needs_grad):
        raise ValueError('length of args is not equal to length of needs_grad')
    equation.validate_sizes(args)
    output_size = tuple(equation.get_sizes(args, equation.output_variables))
    grad_size = tuple(grad.size())
    if grad_size != output_size:
        raise ValueError(
            'size of gradient {} does not match expected size {}'.format(
                grad_size, output_size))
    arg_grads = []
    output_to_input_ranges = [
        x.get_ranges(equation, args)
        for x in equation.reduce_output_to_input
    ]
    other_to_input_ranges = [
        [range(y) for y in equation.get_sizes(args, x)]
        for x in equation.other_reduced_variables
    ]
    other_args = [
        [
            arg
            for j, arg in enumerate(args)
            if j != i
        ]
        for i in range(len(args))
    ]
    for i, arg in enumerate(args):
        if needs_grad[i]:
            arg_size = tuple(equation.get_sizes(args, equation.input_variables[i]))

            def generate_terms():
                for var_values in itertools.product(*output_to_input_ranges[i]):
                    lookup_info = equation.reduce_output_to_input[i].lookup_info[0]
                    grad_slice = lookup_info.lookup(grad, var_values)

                    def generate_inner_terms():
                        for other_var_values in itertools.product(*other_to_input_ranges[i]):
                            reduced_var_values = var_values + other_var_values

                            def generate_factors():
                                lookup_info_list = equation.reduce_others_to_input[i].lookup_info
                                for other_arg, lookup_info in zip(other_args[i], lookup_info_list):
                                    yield lookup_info.lookup(other_arg, reduced_var_values)

                            yield reduce_in_place(
                                _multiply_in_place,
                                generate_factors(),
                                lambda x: adjust_size(x, arg_size))

                    term = reduce_in_place(
                        _add_in_place,
                        generate_inner_terms())
                    term.mul_(grad_slice)
                    yield term

            arg_grad = reduce_in_place(
                _add_in_place,
                generate_terms())
        else:
            arg_grad = None
        arg_grads.append(arg_grad)
    return arg_grads

def _add_in_place(a, b):
    a.add_(b)

def _multiply_in_place(a, b):
    a.mul_(b)