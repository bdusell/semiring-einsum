import itertools
import typing

import torch

from .equation import Equation

def semiring_einsum_forward(
        equation: Equation,
        args: typing.Sequence[torch.Tensor],
        block_size: int,
        func: typing.Callable):
    r"""Implement a custom version of einsum using the callback ``func``.

    This function is the main workhorse used to implement einsum for different
    semirings. It takes away the burden of figuring out how to index the input
    tensors and sum terms in a memory-efficient way, and only requires
    callbacks for performing addition and multiplication. It is also flexible
    enough to support multiple passes through the input tensors, which is
    required when the summation operation is logsumexp. This function is used
    internally by the real, logspace, and Viterbi semiring einsum
    implementations in this package and can be used to implement einsum in
    other semirings as well.

    Note that this function only implements the *forward* aspect of einsum and
    is not differentiable. To turn it into a differentiable PyTorch
    :py:class:`~torch.autograd.Function`, implement its derivative and use
    :py:func:`~semiring_einsum.combine` to combine the forward and backward
    functions into one function.

    The signature of ``func`` is ``func(compute_sum)``, where
    ``compute_sum`` is a function that, when called, runs einsum on the inputs
    with given addition and multiplication operators.
    :py:func:`semiring_einsum_forward` returns the return value of ``func``.
    ``func`` will often consist of a single call to ``compute_sum()``, but
    there are cases where multiple passes over the inputs with different
    semirings is useful (e.g. computing maximum values and then using them
    for a subsequent logsumexp step).

    Here is a quick example that implements the equivalent of
    :py:func:`torch.einsum`:

    .. code-block:: python

        def regular_einsum(equation, *args):
            def func(compute_sum):
                def add_in_place(a, b):
                    a += b
                def multiply_in_place(a, b):
                    a *= b
                return compute_sum(add_in_place, multiply_in_place)
            return semiring_einsum_forward(equation, args, func)

    The signature of ``compute_sum`` is
    ``compute_sum(add_in_place, multiply_in_place, initialize_sum=None,
    include_indexes=False)``.
    The ``+`` and ``*`` operators are customized using ``add_in_place`` and
    ``multiply_in_place``. ``add_in_place(a, b)`` should be a function that
    accepts two :py:class:`~torch.Tensor`\ s and implements ``a += b``, for
    the desired definition of ``+``. Likewise, ``multiply_in_place(a, b)``
    should implement ``a *= b`` for the desired definition of ``*``. These
    functions must modify the tensor object ``a`` *in-place* and not return a
    new tensor.

    The optional function ``initialize_sum(a)`` should be used to modify
    the first term of the einsum summation so that it becomes suitable as
    the accumulator ``a`` in subsequent calls to ``add_in_place``.
    ``initialize_sum()`` must *return* its result. By default, the first
    term is left as-is.

    If ``include_indexes`` is ``True``, then the parameter ``a`` in
    ``initialize_sum(a)`` and the parameter ``b`` in
    ``add_in_place(a, b)`` will become a :py:class:`tuple` of the form
    ``(term, var_values)``, where ``term`` is the usual tensor value, and
    ``var_values`` is a :py:class:`tuple` of :py:class:`int` representing
    the current values of the variables being summed over in the einsum
    summation. This is necessary for implementing argmax for the Viterbi
    semiring.

    :param equation: A pre-compiled equation.
    :param args: A list of input tensors.
    :param func: A callback of the form described above.
    """
    equation.prepare_for_forward()
    equation.validate_sizes(args)

    def compute_sum(add_in_place, sum_block, multiply_in_place,
            initialize_sum=None, include_indexes=False):
        return semiring_einsum_forward_impl(equation, block_size, args,
            initialize_sum, add_in_place, sum_block, multiply_in_place,
            equation.reduce_input_to_output, include_indexes)

    return func(compute_sum)

def semiring_einsum_forward_impl(equation, block_size, args, initialize_sum,
        add_in_place, sum_block, multiply_in_place, reduce_info,
        include_indexes):
    var_ranges = reduce_info.get_ranges(equation, args, block_size)

    def generate_terms():
        for var_values in itertools.product(*var_ranges):

            def generate_factors():
                for arg, arg_info in zip(args, reduce_info.lookup_info):
                    # Get a slice of arg based on the current values of the
                    # reduced variables. The result has a shape of
                    # output_vars x reduced_vars.
                    #yield arg_info.lookup(arg, var_values)
                    yield arg_info.lookup(arg, var_values)

            term_size = reduce_info.get_term_size(equation, args, var_values)
            # Multiply the args together.
            term = reduce_in_place(
                multiply_in_place,
                generate_factors(),
                # Make sure to clone and resize the first factor so that it
                # has the correct shape. Subsequent multiplications will
                # automatically broadcast to the correct shape.
                lambda x: adjust_size(x, term_size))
            # Sum over the reduced variables to get a tensor with the shape of
            # output_vars.
            reduced_term = sum_block(term, reduce_info.reduced_dims)
            if include_indexes:
                # TODO This needs to change for argmax
                yield reduced_term, var_values
            else:
                yield reduced_term

    # Add all the terms together.
    return reduce_in_place(add_in_place, generate_terms(), initialize_sum)

def adjust_size(arg, size):
    repeat_size = []
    for output_size, arg_size in zip(size, arg.size()):
        repeat_size.append(output_size if arg_size == 1 else 1)
    # NOTE: Important! This call to .repeat() creates a copy of the first
    # tensor used to compute each product, so the subsequent in-place
    # operations will not affect the original tensor.
    return arg.repeat(*repeat_size)

def reduce_in_place(func, args, initialize=None):
    # NOTE: Important! This modifies the first element of `args` in-place.
    it = iter(args)
    result = next(it)
    if initialize is not None:
        result = initialize(result)
    for arg in it:
        func(result, arg)
    return result
