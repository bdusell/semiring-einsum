import functools
import itertools
import typing

import torch

class ParsedEquation:

    def __init__(self, variable_locations, input_variables, output_variables,
            num_variables):
        self.variable_locations = variable_locations
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.num_variables = num_variables

    def validate_sizes(self, args):
        for loc_list in self.variable_locations:
            i, j = loc_list[0]
            size = args[i].size(j)
            for i, j in loc_list[1:]:
                if args[i].size(j) != size:
                    raise ValueError(
                        'dimension {} of argument {} does not match '
                        'dimension {} of argument {}'.format(
                            j, i, loc_list[0][1], loc_list[0][0]))

    def get_sizes(self, args, variables):
        result = []
        for variable in variables:
            i, j = self.variable_locations[variable][0]
            result.append(args[i].size(j))
        return result

    def all_variables(self):
        return range(self.num_variables)

def parse_equation(equation):
    args_str, output_vars = equation.split('->', 1)
    arg_strs = args_str.split(',')
    char_to_int = {}
    int_to_arg_dims = []
    args_dims = []
    for arg_no, arg_str in enumerate(arg_strs):
        arg_dims = []
        for dim_no, dim_char in enumerate(arg_str):
            dim_int = char_to_int.get(dim_char)
            if dim_int is None:
                dim_int = char_to_int[dim_char] = len(char_to_int)
                int_to_arg_dims.append([])
            int_to_arg_dims[dim_int].append((arg_no, dim_no))
            arg_dims.append(dim_int)
        args_dims.append(arg_dims)
    output_dims = [char_to_int[c] for c in output_vars]
    num_variables = len(char_to_int)
    return ParsedEquation(
        int_to_arg_dims,
        args_dims,
        output_dims,
        num_variables)

class EquationForForward(ParsedEquation):

    def __init__(self, parsed_equation, reduce_input_to_output):
        super().__init__(
            parsed_equation.variable_locations,
            parsed_equation.input_variables,
            parsed_equation.output_variables,
            parsed_equation.num_variables)
        self.reduce_input_to_output = reduce_input_to_output

def compile_equation_for_forward(parsed_equation):
    if not isinstance(parsed_equation, ParsedEquation):
        raise TypeError
    summed_vars = get_variables_not_in(
        (v for arg_vars in parsed_equation.input_variables for v in arg_vars),
        set(parsed_equation.output_variables))
    reduce_input_to_output = create_reduce_info(
        parsed_equation,
        parsed_equation.input_variables,
        summed_vars,
        parsed_equation.output_variables)
    return EquationForForward(
        parsed_equation,
        reduce_input_to_output)

def get_variables_not_in(variables, excluded):
    added = set()
    result = []
    for variable in variables:
        if variable not in excluded and variable not in added:
            added.add(variable)
            result.append(variable)
    return result

class ReduceInfo:

    def __init__(self, reduced_variables, lookup_info):
        self.reduced_variables = reduced_variables
        self.lookup_info = lookup_info

    def get_ranges(self, parsed_equation, args):
        sizes = parsed_equation.get_sizes(args, self.reduced_variables)
        return [range(s) for s in sizes]

_COLON = slice(None)

class LookupInfo:

    def __init__(self, index_map, num_extra_vars, permutation):
        self.index_map = index_map
        self.num_extra_vars = num_extra_vars
        self.permutation = permutation

    def lookup(self, arg, var_values):
        index = [_COLON] * arg.dim()
        for source_index, dest_index in self.index_map:
            index[dest_index] = var_values[source_index]
        for i in range(self.num_extra_vars):
            index.append(None)
        return arg[index].permute(self.permutation)

def create_reduce_info(parsed_equation, input_var_lists, reduced_vars,
        output_vars):
    lookup_info = []
    reduced_vars_dict = { v : i for i, v in enumerate(reduced_vars) }
    for input_vars in input_var_lists:
        index_map = []
        input_var_dict = {}
        counter = 0
        for dest_index, input_var in enumerate(input_vars):
            source_index = reduced_vars_dict.get(input_var)
            if source_index is not None:
                index_map.append((source_index, dest_index))
            else:
                input_var_dict[input_var] = counter
                counter += 1
        num_extra_vars = 0
        permutation = []
        for output_var in output_vars:
            perm_index = input_var_dict.get(output_var)
            if perm_index is None:
                perm_index = len(input_var_dict) + num_extra_vars
                num_extra_vars += 1
            permutation.append(perm_index)
        lookup_info.append(LookupInfo(index_map, num_extra_vars, permutation))
    return ReduceInfo(reduced_vars, lookup_info)

def semiring_einsum_forward(
        equation: ParsedEquation,
        args: typing.Sequence[torch.Tensor],
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
    if not isinstance(equation, EquationForForward):
        raise TypeError
    equation.validate_sizes(args)
    var_ranges = equation.reduce_input_to_output.get_ranges(equation, args)
    output_size = equation.get_sizes(args, equation.output_variables)
    lookup_info = equation.reduce_input_to_output.lookup_info

    def compute_sum(add_in_place, multiply_in_place, initialize_sum=None,
            include_indexes=False):
        return semiring_einsum_forward_impl(args, initialize_sum, add_in_place,
            multiply_in_place, var_ranges, output_size, lookup_info,
            include_indexes)

    return func(compute_sum)

def semiring_einsum_forward_impl(args, initialize_sum, add_in_place,
        multiply_in_place, variable_ranges, output_size, lookup_info,
        include_indexes):

    def generate_terms():
        for var_values in itertools.product(*variable_ranges):

            def generate_factors():
                for arg, arg_info in zip(args, lookup_info):
                    yield arg_info.lookup(arg, var_values)

            term = reduce_in_place(
                multiply_in_place,
                generate_factors(),
                lambda x: adjust_size(x, output_size))
            if include_indexes:
                yield term, var_values
            else:
                yield term

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
