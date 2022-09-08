import functools
import itertools
import typing

import torch

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

class AutomaticBlockSize:
    r"""Indicates that the amount of memory used to sum elements in an einsum
    operation should be determined automatically based on the amount of
    available memory.

    When the device is ``cuda``, this automatically calculates the amount of
    free GPU memory on the current device and makes the block size as big as
    possible without exceeding it. When the device is ``cpu``, this uses the
    value of ``max_cpu_bytes`` to determine how much memory it can use.
    """

    def __init__(self,
            mock_available_bytes: typing.Optional[int]=None,
            max_cpu_bytes=(1 << 30)):
        """
        :param mock_available_bytes: If not ``None``, ignores the amount of
            available memory and uses this value as the number of available
            bytes in memory instead. Mainly useful for testing.
        :param max_cpu_bytes: The maximum amount of memory (in bytes) to use
            when the device is ``cpu``. By default, this is set to 1 GiB.
        """
        super().__init__()
        self.mock_available_bytes = mock_available_bytes
        self.max_cpu_bytes = max_cpu_bytes

AUTOMATIC_BLOCK_SIZE = AutomaticBlockSize()
r"""Use this as ``block_size`` to determine block size automatically based on
available memory."""

class ReduceInfo:
    r"""Holds data structures that facilitate the basic einsum operation of
    multiplying terms together while summing over multiple dimensions."""

    def __init__(self, lookup_info, output_variables, reduced_variables,
            reduced_dims):
        self.lookup_info = lookup_info
        self.output_variables = output_variables
        self.reduced_variables = reduced_variables
        self.reduced_dims = reduced_dims

    def get_summed_variable_indexes(self, equation, args, block_size):
        return get_summed_variable_indexes(
            equation,
            args,
            self.reduced_variables,
            block_size
        )

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

def get_summed_variable_indexes(equation, args, variables, block_size):
    sizes = equation.get_sizes(args, variables)
    if isinstance(block_size, int):
        return get_fixed_block_size_indexes(sizes, block_size)
    elif isinstance(block_size, AutomaticBlockSize):
        return get_automatic_block_size_indexes(equation, args, sizes, block_size)
    else:
        raise ValueError(f'unrecognized block_size: {block_size!r}')

def get_bits_per_element(dtype):
    try:
        return torch.finfo(dtype).bits
    except TypeError:
        return torch.iinfo(dtype).bits

def get_fixed_block_size_indexes(sizes, block_size):
    return block_sizes_to_indexes(sizes, (block_size for size in sizes))

def block_sizes_to_indexes(sizes, block_sizes):
    range_lists = [
        list(generate_slices(size, block_size))
        for size, block_size in zip(sizes, block_sizes)
    ]
    return itertools.product(*range_lists)

def generate_slices(total_size, block_size):
    lo = 0
    while lo < total_size:
        hi = min(lo + block_size, total_size)
        yield slice(lo, hi)
        lo = hi

def get_automatic_block_size_indexes(equation, args, sizes, auto_block_size):
    if not args:
        return []
    device = args[0].device
    dtype = args[0].dtype
    if auto_block_size.mock_available_bytes is not None:
        available_bytes = auto_block_size.mock_available_bytes
    else:
        available_bytes = get_available_bytes(device, auto_block_size)
    bytes_per_element = get_bits_per_element(dtype) // 8
    # Figure out the number of tensor elements that will be taken up by the
    # output tensor. This will be subtracted from the total available elements.
    output_elements = get_output_elements(equation, args)
    # Figure out the total number of tensor elements that can fit in memory.
    available_elements = available_bytes // bytes_per_element - output_elements
    block_sizes = get_automatic_block_sizes(sizes, available_elements)
    return block_sizes_to_indexes(sizes, block_sizes)

def get_available_bytes(device, auto_block_size):
    if device.type == 'cuda':
        return get_available_bytes_cuda(device)
    elif device.type == 'cpu':
        return auto_block_size.max_cpu_bytes
    else:
        raise ValueError(f'unrecognized device type: {device!r}')

def get_available_bytes_cuda(device):
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    reserved_bytes = torch.cuda.memory_reserved(device)
    allocated_bytes = torch.cuda.memory_allocated(device)
    return (reserved_bytes - allocated_bytes) + free_bytes

def get_output_elements(equation, args):
    # Take the product of the sizes of the output dimensions.
    sizes = equation.get_sizes(args, equation.output_variables)
    return functools.reduce(lambda a, b: a * b, sizes, 1)

def get_automatic_block_sizes(sizes, available_elements):
    if available_elements <= 0:
        raise ValueError('no memory available to create any blocks')
    # This is a very naive and certainly non-optimal way of finding a set of
    # block sizes where their product is close to but does not exceed the
    # number of available elements. It works by sorting dimensions from
    # smallest to largest and multiplying them together until the product gets
    # too big.
    sorted_sizes = sorted(enumerate(sizes), key=lambda x: x[1])
    block_sizes = [1] * len(sizes)
    total_size = 1
    for index, size in sorted_sizes:
        new_total_size = total_size * size
        if new_total_size <= available_elements:
            # If multiplying the next dimension into the total product does
            # not exceed the limit, multiply it in.
            block_sizes[index] = size
            total_size = new_total_size
        else:
            # Otherwise, figure out how big we can make the block size for this
            # dimension without exceeding the total by dividing the number of
            # available elements by the total size so far before adding this
            # dimension.
            new_block_size = available_elements // total_size
            block_sizes[index] = new_block_size
            # Since the running product can only increase, there's no reason
            # to continue for further iterations; they would set all the
            # remaining block sizes to 1 anyway.
            break
    return block_sizes

_COLON = slice(None)

class LookupInfo:
    r"""Holds data structures for slicing, unsqueezing, and permuting a tensor
    so that all of its dimensions line up in a predictable way."""

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
        return arg[tuple(index)].permute(self.permutation)

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
