import itertools

class Equation:
    r"""An einsum equation that has been pre-compiled into some useful data
    structures."""

    def __init__(self, source, variable_locations, input_variables,
            output_variables, num_variables):
        self.source = source
        self.variable_locations = variable_locations
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.num_variables = num_variables
        self.reduce_input_to_output = None
        self.reduce_others_to_input = None
        self.reduce_all_to_input = None

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

    def prepare_for_forward(self):
        if self.reduce_input_to_output is None:
            self.reduce_input_to_output = create_reduce_info(
                self.input_variables,
                self.output_variables)

    def prepare_for_backward(self):
        if self.reduce_others_to_input is None:
            self.reduce_others_to_input = []
            for i, input_var_list in enumerate(self.input_variables):
                other_vars = list(self.input_variables)
                other_vars[i] = self.output_variables
                self.reduce_others_to_input.append(create_reduce_info(
                    other_vars, input_var_list))

    def prepare_for_log_backward(self):
        if self.reduce_all_to_input is None:
            self.reduce_all_to_input = []
            inputs = self.input_variables + [self.output_variables]
            for input_var_list in self.input_variables:
                reduce_info = create_reduce_info(inputs, input_var_list)
                output_lookup_info = reduce_info.lookup_info.pop()
                self.reduce_all_to_input.append((reduce_info, output_lookup_info))

def compile_equation(equation: str) -> Equation:
    r"""Pre-compile an einsum equation for use with the einsum functions in
    this package.

    :param equation: An equation in einsum syntax.
    :return: A pre-compiled equation.
    """
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
    return Equation(
        equation,
        int_to_arg_dims,
        args_dims,
        output_dims,
        num_variables)

def get_variables_not_in(variables, excluded):
    added = set()
    result = []
    for variable in variables:
        if variable not in excluded and variable not in added:
            added.add(variable)
            result.append(variable)
    return result

class ReduceInfo:
    r"""Holds data structures that facilitate the basic einsum operation of
    multiplying terms together while summing over multiple dimensions."""

    def __init__(self, lookup_info, output_variables, reduced_variables,
            reduced_dims):
        self.lookup_info = lookup_info
        self.output_variables = output_variables
        self.reduced_variables = reduced_variables
        self.reduced_dims = reduced_dims

    def get_ranges(self, equation, args, block_size):
        return get_ranges(equation, args, self.reduced_variables,
            block_size)

    def get_term_size(self, equation, args, var_values):
        # Compute the size of each of the terms in an einsum. Each term is a
        # slice over ranges of the summed variables.
        # Note that var_values is a list of slices, not a list of ints.
        return (
            # The output dimensions come first, and their sizes match the
            # inputs.
            equation.get_sizes(args, self.output_variables) +
            # The summed dimensions come last, and their sizes correspond to
            # the sizes of the slices.
            [s.stop - s.start for s in var_values]
        )

def get_ranges(equation, args, variables, block_size):
    sizes = equation.get_sizes(args, variables)
    return [list(generate_slices(s, block_size)) for s in sizes]

def generate_slices(total_size, block_size):
    lo = 0
    while lo < total_size:
        hi = min(lo + block_size, total_size)
        yield slice(lo, hi)
        lo = hi

_COLON = slice(None)

class LookupInfo:
    r"""Holds data structures for slicing, unsqueezing, and permuting a tensor
    so that all of its dimensions line up in a predictable way."""

    def __init__(self, index_map, num_extra_vars, permutation):
        self.index_map = index_map
        self.num_extra_vars = num_extra_vars
        self.permutation = permutation

        n = max((1 + source_index for source_index, dest_index in index_map),
                default=0)
        self.source_to_dest = [None] * n
        for source_index, dest_index in index_map:
            self.source_to_dest[source_index] = dest_index

    def lookup(self, arg, var_values):
        index = [_COLON] * arg.dim()
        for source_index, dest_index in self.index_map:
            assert(dest_index == self.permutation[arg.ndim + self.num_extra_vars - len(var_values) + source_index])
            index[dest_index] = var_values[source_index]
        for i in range(self.num_extra_vars):
            index.append(None)
        return arg[tuple(index)].permute(self.permutation)

    def view(self, arg):
        for i in range(self.num_extra_vars):
            arg = arg.unsqueeze(-1)
        return arg.permute(self.permutation)

    def view_lookup(self, argv, var_values):
        # TODO: generate this code in __init__ using ast
        return argv[tuple(itertools.chain(
            (Ellipsis,),
            (_COLON if dest_index is None else var_value
             for dest_index, var_value in itertools.zip_longest(self.source_to_dest, var_values))))]

def create_reduce_info(input_vars, output_vars):
    r"""Pre-compile a data structure that will help reduce the variables
    given in ``input_vars`` to the variables in ``output_vars``."""
    reduced_vars = get_variables_not_in(
        (v for var_list in input_vars for v in var_list),
        set(output_vars))
    # The shape of the final, reshaped tensor will correspond to
    # output_vars + reduced_vars.
    lookup_info = []
    reduced_vars_dict = { v : i for i, v in enumerate(reduced_vars) }
    for input_var_list in input_vars:
        index_map = []
        # Loop over all variables in this input tensor.
        for dest_index, input_var in enumerate(input_var_list):
            source_index = reduced_vars_dict.get(input_var)
            if source_index is not None:
                # The variable is a reduced variable.
                # This maps an index of a list of reduced variable values
                # (source_index) to an index of a tuple used to slice the
                # input tensor (dest_index).
                index_map.append((source_index, dest_index))
        input_var_dict = { v : i for i, v in enumerate(input_var_list) }
        num_extra_vars = 0
        permutation = []
        # Loop over all variables in the output.
        for output_var in itertools.chain(output_vars, reduced_vars):
            perm_index = input_var_dict.get(output_var)
            if perm_index is None:
                # The variable does not appear in the input after it has been
                # indexed.
                perm_index = len(input_var_dict) + num_extra_vars
                num_extra_vars += 1
            permutation.append(perm_index)
        lookup_info.append(LookupInfo(index_map, num_extra_vars, permutation))
    reduced_dims = tuple(range(
        len(output_vars),
        len(output_vars) + len(reduced_vars)
    ))
    return ReduceInfo(lookup_info, output_vars, reduced_vars, reduced_dims)
