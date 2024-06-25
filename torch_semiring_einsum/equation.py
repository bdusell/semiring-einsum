import functools
import itertools
import typing

import pynvml
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
            max_cpu_bytes: int=(1 << 30),
            max_cuda_bytes: typing.Optional[int]=None,
            cache_available_cuda_memory: bool=True,
            cuda_memory_proportion: float=0.8,
            repr_string=None):
        """
        :param max_cpu_bytes: The maximum amount of memory (in bytes) to use
            when the device is ``cpu``. By default, this is set to 1 GiB.
        :param max_cuda_bytes: The maximum amount of memory (in bytes) to use
            when the device is ``cuda``. If ``None``, then the amount of memory
            used will be determined based on the amount of free CUDA memory.
            Note that specifying an explicit memory limit is much faster than
            querying the amount of free CUDA memory.
        :param cache_available_cuda_memory: Only applies when
            ``max_cuda_bytes`` is ``None``. When true, the amount of available
            CUDA memory is only queried the first time einsum is called with
            this object as ``block_size``, and it is reused on subsequent
            calls. This is significantly faster than querying the amount of
            available memory every time. To account for future decreases in the
            amount of available memory, only a portion of the available memory
            is used, as determined by ``cuda_memory_proportion``.
        :param cuda_memory_proportion: Determines the proportion of available
            memory used when ``cache_available_cuda_memory`` is true. This
            should be a number between 0 and 1.
        """
        super().__init__()
        self.max_cpu_bytes = max_cpu_bytes
        self.max_cuda_bytes = max_cuda_bytes
        self.cache_available_cuda_memory = cache_available_cuda_memory
        self.cuda_memory_proportion = cuda_memory_proportion
        self.available_cuda_memory = {}
        self.repr_string = repr_string

    def __repr__(self):
        if self.repr_string is not None:
            return self.repr_string
        else:
            return super().__repr__()

AUTOMATIC_BLOCK_SIZE = AutomaticBlockSize(repr_string='AUTOMATIC_BLOCK_SIZE')

class ReduceInfo:
    r"""Holds data structures that facilitate the basic einsum operation of
    multiplying terms together while summing over multiple dimensions."""

    def __init__(self, lookup_info, output_variables, reduced_variables,
            reduced_dims):
        self.lookup_info = lookup_info
        self.output_variables = output_variables
        self.reduced_variables = reduced_variables
        self.reduced_dims = reduced_dims

    def get_summed_variable_indexes(self, equation, args, block_size,
            output_dtypes=(None,)):
        return get_summed_variable_indexes(
            equation,
            args,
            self.reduced_variables,
            block_size,
            output_dtypes)

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

def get_summed_variable_indexes(equation, args, variables, block_size, output_dtypes):
    sizes = equation.get_sizes(args, variables)
    if isinstance(block_size, int):
        return get_fixed_block_size_indexes(sizes, block_size)
    elif isinstance(block_size, AutomaticBlockSize):
        return get_automatic_block_size_indexes(
            equation,
            args,
            sizes,
            block_size,
            output_dtypes)
    else:
        raise ValueError(f'unrecognized block_size: {block_size!r}')

def get_bits_per_element(dtype):
    try:
        try:
            return torch.finfo(dtype).bits
        except TypeError:
            return torch.iinfo(dtype).bits
    except TypeError:
        if dtype == torch.bool:
            # torch.bool is supported by neither torch.finfo nor torch.iinfo.
            return 8
        else:
            raise TypeError(f'cannot determine number of bits in dtype {dtype}')

def get_bytes_per_element(dtype):
    return get_bits_per_element(dtype) // 8

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

def get_automatic_block_size_indexes(equation, args, sizes, auto_block_size,
        output_dtypes):
    if not args:
        return []
    device = args[0].device
    dtype = args[0].dtype
    # Get the number of bytes available in memory.
    available_bytes = get_available_bytes(device, auto_block_size)
    # Get the number of bytes per element in the block.
    bytes_per_element = get_bytes_per_element(dtype)
    # Figure out the number of tensor elements that will be taken up by the
    # output tensor. This will be subtracted from the total available memory.
    output_elements = get_output_elements(equation, args)
    # Get the number of bytes per element in the output.
    actual_output_dtypes = (dtype if x is None else x for x in output_dtypes)
    bytes_per_output_element = sum(map(get_bytes_per_element, actual_output_dtypes))
    # Count the size of the output tensor twice: once for the final output
    # tensor, and again for the temporary output tensor that is added to it.
    output_bytes = 2 * bytes_per_output_element * output_elements
    # Figure out the total number of tensor elements that can fit in memory.
    available_elements = (available_bytes - output_bytes) // bytes_per_element
    block_sizes = get_automatic_block_sizes(sizes, available_elements)
    return block_sizes_to_indexes(sizes, block_sizes)

def get_available_bytes(device, auto_block_size):
    if device.type == 'cuda':
        return get_available_bytes_cuda(device, auto_block_size)
    elif device.type == 'cpu':
        return auto_block_size.max_cpu_bytes
    else:
        raise ValueError(f'unrecognized device type: {device!r}')

def get_available_bytes_cuda(device, auto_block_size):
    if auto_block_size.max_cuda_bytes is not None:
        return auto_block_size.max_cuda_bytes
    else:
        if auto_block_size.cache_available_cuda_memory:
            # Cache the amount of CUDA memory available, since querying this
            # is very slow.
            if device.index not in auto_block_size.available_cuda_memory:
                auto_block_size.available_cuda_memory[device.index] = round(
                    auto_block_size.cuda_memory_proportion *
                    get_real_available_cuda_bytes(device))
            return auto_block_size.available_cuda_memory[device.index]
        else:
            return get_real_available_cuda_bytes(device)

def get_real_available_cuda_bytes(device):
    free_bytes = get_cuda_free_bytes(device)
    reserved_bytes = get_cuda_memory_reserved(device)
    allocated_bytes = torch.cuda.memory_allocated(device)
    return (reserved_bytes - allocated_bytes) + free_bytes

# torch.cuda.memory_cached was renamed to torch.cuda.memory_reserved in
# PyTorch 1.4.0.
if hasattr(torch.cuda, 'memory_reserved'):
    get_cuda_memory_reserved = torch.cuda.memory_reserved
else:
    get_cuda_memory_reserved = torch.cuda.memory_cached

# torch.cuda.mem_get_info was introduced in PyTorch 1.11.0.
if hasattr(torch.cuda, 'mem_get_info'):
    def get_cuda_free_bytes(device):
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        return free_bytes
else:
    def get_cuda_free_bytes(device):
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device.index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.free

def get_output_elements(equation, args):
    # Take the product of the sizes of the output dimensions.
    sizes = equation.get_sizes(args, equation.output_variables)
    return functools.reduce(lambda a, b: a * b, sizes, 1)

def get_automatic_block_sizes(sizes, available_elements):
    # This is a very naive and certainly non-optimal way of finding a set of
    # block sizes where their product is close to but does not exceed the
    # number of available elements. It works by sorting dimensions from
    # smallest to largest and multiplying them together until the product gets
    # too big.
    block_sizes = [1] * len(sizes)
    if available_elements <= 0:
        # There isn't any memory left according to our estimate. In this case,
        # just use 1 for all dimensions and hope for the best. Maybe it won't
        # actually fail.
        return block_sizes
    sorted_sizes = sorted(enumerate(sizes), key=lambda x: x[1])
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
