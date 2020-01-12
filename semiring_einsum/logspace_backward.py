import itertools
import typing

import torch

from .equation import Equation, get_ranges
from .extend import semiring_einsum_forward_impl, reduce_in_place

def logspace_einsum_backward(
        equation: Equation,
        args: typing.Sequence[torch.Tensor],
        needs_grad: typing.Sequence[bool],
        grad: torch.Tensor,
        block_size: int) -> typing.List[typing.Optional[torch.Tensor]]:
    r"""Compute the derivative of
    :py:func:`~semiring_einsum.logspace_einsum_forward`.

    Like the forward pass, the backward pass is done in memory-efficient
    fashion by doing summations in-place.

    :param equation: Pre-compiled einsum equation. The derivative of the
        logspace einsum operation specified by this equation will be computed.
        The equation must have been compiled with ``backward=True``.
    :param args: The inputs to the logspace einsum operation whose derivative
        is being computed.
    :param needs_grad: Indicates which inputs in ``args`` require gradient.
    :param grad: The gradient of the loss function with respect to the output
        of the logspace einsum operation.
    :return: The gradients with respect to each of the inputs to the logspace
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
    input_to_output_ranges = equation.reduce_input_to_output.get_ranges(
        equation, args)
    # Z : same size as output of equation
    Z = semiring_einsum_forward_impl(
        args,
        initialize_sum=_init_addexp,
        add_in_place=_addexp_in_place,
        multiply_in_place=_add_in_place,
        variable_ranges=input_to_output_ranges,
        output_size=output_size,
        lookup_info=equation.reduce_input_to_output.lookup_info,
        include_indexes=False)
    # C : same size as output of equation
    C = grad / Z
    del Z
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

            def generate_terms():
                # Sum over all variables in the output but not args[i].
                for var_values in itertools.product(*output_to_input_ranges[i]):
                    lookup_info = equation.reduce_output_to_input[i].lookup_info[0]
                    C_slice = lookup_info.lookup(C, var_values)

                    def generate_inner_terms():
                        # Sum over all leftover variables not in the output or args[i].
                        for other_var_values in itertools.product(*other_to_input_ranges[i]):
                            reduced_var_values = var_values + other_var_values

                            def generate_factors():
                                # Take the product over all args[i].
                                # No need to reshape the current input -- it
                                # already matches the shape of its gradient!
                                # NOTE: Important! The first tensor yielded
                                # *must* be a copy, or else the in-place
                                # operations will modify the original input,
                                # leading to incorrect results.
                                yield arg.clone()
                                lookup_info_list = equation.reduce_others_to_input[i].lookup_info
                                for other_arg, lookup_info in zip(other_args[i], lookup_info_list):
                                    yield lookup_info.lookup(other_arg, reduced_var_values)

                            yield reduce_in_place(
                                _add_in_place,
                                generate_factors())

                    term = reduce_in_place(
                        _addexp_in_place,
                        generate_inner_terms(),
                        _init_addexp)
                    term.mul_(C_slice)
                    yield term

            arg_grad = reduce_in_place(
                _add_in_place,
                generate_terms())
        else:
            arg_grad = None
        arg_grads.append(arg_grad)
    return arg_grads

def _init_addexp(a):
    a.exp_()
    return a

def _addexp_in_place(a, b):
    b.exp_()
    a.add_(b)

def _add_in_place(a, b):
    a.add_(b)
