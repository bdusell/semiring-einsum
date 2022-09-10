import typing

import torch

from .equation import Equation, AutomaticBlockSize

def semiring_einsum_forward(
        equation: Equation,
        args: typing.Sequence[torch.Tensor],
        block_size: typing.Union[int, AutomaticBlockSize],
        func: typing.Callable):
    r"""Implement a custom version of einsum using the callback ``func``.

    This function is the main workhorse used to implement einsum for different
    semirings. It takes away the burden of figuring out how to index the input
    tensors and sum terms in a memory-efficient way, and only requires
    callbacks for performing addition and multiplication. It is also flexible
    enough to support multiple passes through the input tensors (this feature
    is required for implementing logsumexp). This function is used internally
    by the real, log, and Viterbi semiring einsum implementations in this
    package and can be used to implement einsum in other semirings as well.

    Note that this function only implements the *forward* aspect of einsum and
    is not differentiable. To turn your instantiation of einsum in a
    particular semiring into a differentiable PyTorch
    :py:class:`~torch.autograd.Function`, implement its derivative and use
    :py:func:`~torch_semiring_einsum.combine` to combine the forward and
    backward functions into one function. Odds are,
    :py:func:`~semiring_einsum_forward` can be used to implement the derivative
    efficiently as well (despite including "forward" in the name, there is
    nothing preventing you from using it as a tool in the backward step).

    :py:func:`~semiring_einsum_forward` will call ``func`` as
    ``func(compute_sum)``, where ``compute_sum`` is itself another function.
    Calling ``compute_sum`` executes a single einsum pass over the input
    tensors, where you supply custom functions for addition and
    multiplication; this is where the semiring customization really takes
    place. :py:func:`~semiring_einsum_forward` returns whatever you return from
    ``func``, which will usually be what is returned from ``compute_sum``.
    ``func`` will often consist of a single call to ``compute_sum()``, but
    there are cases where multiple passes over the inputs with different
    semirings is useful (e.g. for a numerically stable logsumexp
    implementation, one must first compute maximum values and then use them
    for a subsequent logsumexp step).

    Here is a quick example that implements the equivalent of
    :py:func:`torch.einsum`:

    .. code-block:: python

        def regular_einsum(equation, *args, block_size):
            def func(compute_sum):
                def add_in_place(a, b):
                    a += b
                def sum_block(a, dims):
                    if dims:
                        return torch.sum(a, dim=dims)
                    else:
                        # This is an edge case that `torch.sum` does not
                        # handle correctly.
                        return a
                def multiply_in_place(a, b):
                    a *= b
                return compute_sum(add_in_place, sum_block, multiply_in_place)
            return semiring_einsum_forward(equation, args, block_size, func)

    The full signature of ``compute_sum`` is
    ``compute_sum(add_in_place, sum_block, multiply_in_place,
    include_indexes=False, output_dtypes=(None,))``.
    The ``+`` and ``*`` operators are customized using ``add_in_place``,
    ``sum_block``, and ``multiply_in_place``.

    ``add_in_place(a, b)`` must be a function that accepts two
    values and implements ``a += b`` for the desired definition of ``+``.
    Likewise, ``multiply_in_place(a, b)`` must implement ``a *= b`` for the
    desired definition of ``*``. The arguments ``a`` and ``b`` are values
    returned from ``sum_block`` (see below) and are usually of type
    :py:class:`~torch.Tensor`, although they can be something fancier for
    cases like Viterbi (which involves a pair of tensors: max and argmax).
    These functions must modify the object ``a`` *in-place*; the return value
    is ignored.

    ``sum_block(a, dims)`` should be a function that "sums" over multiple
    dimensions in a tensor at once. It must return its result. ``a`` is always
    a :py:class:`~torch.Tensor`. ``dims`` is a :py:class:`tuple` of
    :py:class:`int`\ s representing the dimensions in ``a`` to sum out. Take
    special care to handle the case where ``dims`` is an empty tuple --
    in particular, keep in mind that :py:func:`torch.sum` returns a *scalar*
    when ``dims`` is an empty tuple. Simply returning ``a`` is sufficient to
    handle this edge case. Note that it is always safe to return a *view* of
    ``a`` from ``sum_block``, since ``a`` is itself never a view of the input
    tensors, but always a new tensor.

    If ``include_indexes`` is ``True``, then ``sum_block`` will receive a
    third argument, ``var_values``, which contains the current indexes of the
    parts of the input tensors being summed over (``sum_block`` is called
    multiple times on different slices of the inputs). ``var_values`` is a
    :py:class:`tuple` of :py:class:`range` objects representing the *ranges*
    of indexes representing the current slice. ``var_values`` contains an
    entry for each summed variable, in order of first appearance in the
    equation.

    The optional argument ``output_dtypes`` should be a list of PyTorch dtypes
    that represents the components of the output tensor. It is used to
    calculate the memory required by the output tensor when performing
    automatic block sizing. In most cases, the output tensor simply has the
    same dtype as the input tensor. In some cases, like Viterbi, the output
    tensor has multiple components (e.g. a tensor of floats for the max and a
    tensor of ints for the argmax). A dtype of ``None`` can be used to indicate
    the same dtype as the input tensors. The default value for
    ``output_dtypes`` is ``(None,)``.

    :param equation: A pre-compiled equation.
    :param args: A list of input tensors.
    :param block_size: To keep memory usage in check, the einsum summation is
        done over multiple "windows" or "blocks" of bounded size. This
        parameter sets the maximum size of these windows. More precisely, it
        defines the maximum size of the range of values of each summed
        variable that is included in a single window. If there are ``n``
        summed variables, the size of the window tensor is proportional to
        ``block_size ** n``.
    :param func: A callback of the form described above.
    """
    equation.validate_sizes(args)
    equation.prepare_for_forward()

    def compute_sum(add_in_place, sum_block, multiply_in_place,
            include_indexes=False, output_dtypes=(None,)):
        return semiring_einsum_forward_impl(
            equation,
            args,
            block_size,
            args,
            add_in_place,
            sum_block,
            multiply_in_place,
            equation.reduce_input_to_output,
            include_indexes,
            output_dtypes)

    return func(compute_sum)

def semiring_einsum_forward_impl(equation, args, block_size, inputs,
        add_in_place, sum_block, multiply_in_place, reduce_info,
        include_indexes, output_dtypes=(None,)):

    def generate_terms():
        summed_variable_indexes = reduce_info.get_summed_variable_indexes(
            equation,
            args,
            block_size,
            output_dtypes)
        for var_values in summed_variable_indexes:
            # var_values is a tuple of slices.

            def generate_factors():
                for arg, arg_info in zip(inputs, reduce_info.lookup_info):
                    # Get a slice of arg based on the current values of the
                    # reduced variables. The result has a shape of
                    # output_vars x reduced_vars.
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
            if include_indexes:
                reduced_term = sum_block(term, reduce_info.reduced_dims, var_values)
            else:
                reduced_term = sum_block(term, reduce_info.reduced_dims)
            yield reduced_term

    # Add all the terms together.
    return reduce_in_place(add_in_place, generate_terms())

def adjust_size(arg, size):
    # If the input is a scalar with no dimensions to resize, just return a
    # clone of the input (the output must always be a copy of the input).
    if arg.dim() == 0:
        return arg.clone()
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
