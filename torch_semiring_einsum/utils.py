import math

import torch

def add_in_place(a, b):
    a.add_(b)

def multiply_in_place(a, b):
    a.mul_(b)

def max_in_place(a, b):
    # `torch.max` has been available since PyTorch 0.1.12.
    torch.max(a, b, out=a)

def min_in_place(a, b):
    torch.min(a, b, out=a)

def sum_block(a, dims):
    if dims:
        # `dim` was first allowed to be a tuple in PyTorch 0.4.1.
        return torch.sum(a, dim=dims)
    else:
        # This is an edge case where PyTorch returns a bad result.
        # See https://github.com/pytorch/pytorch/issues/29137
        # On the plus side, this avoids creating an unnecessary copy of `a`.
        return a

# Define the max_block function differently depending on the version of
# PyTorch.
if hasattr(torch, 'amax'):
    def max_block(a, dims):
        # `amax` was introduced in PyTorch 1.7.0.
        # Unlike `max`, `amax` supports reducing multiple dimensions at once.
        # But `amax(dim=())` reduces all dimensions, which we override to
        # reduce no dimensions. The same issue occurs with `torch.sum` above.
        if dims:
            return torch.amax(a, dim=dims)
        else:
            return a
else:
    def max_block(a, dims):
        # Fall back to reducing each dimension one at a time using `max`.
        result = a
        for dim in reversed(dims):
            result = torch.max(result, dim=dim)[0]
        return result

# Define the min_block function, similarly to above.
if hasattr(torch, 'amin'):
    def min_block(a, dims):
        if dims:
            return torch.amin(a, dim=dims)
        else:
            return a
else:
    def min_block(a, dims):
        result = a
        for dim in reversed(dims):
            result = torch.min(result, dim=dim)[0]
        return result

# Define the clip_inf_in_place function differently depending on the version
# of PyTorch.
if hasattr(torch.Tensor, 'nan_to_num_'):
    def clip_inf_in_place(a):
        # `.nan_to_num_` was introduced in PyTorch 1.8.0.
        a.nan_to_num_(nan=math.nan)
else:
    def clip_inf_in_place(a):
        # `torch.max` and `torch.min` have been available since PyTorch 0.1.12.
        # `.dtype` was introduced in PyTorch 0.4.0.
        # `.new_tensor` was introduced in PyTorch 0.4.0.
        # `torch.finfo().min` and `torch.finfo().max` were introduced in
        # PyTorch 1.1.0.
        finfo = torch.finfo(a.dtype)
        min_float = a.new_tensor(finfo.min)
        torch.max(a, min_float, out=a)
        max_float = a.new_tensor(finfo.max)
        torch.min(a, max_float, out=a)

# Define the Boolean semiring functions differently depending on the version of
# PyTorch.

if hasattr(torch.Tensor, 'logical_or_'):

    # `.logical_or_()` and `.logical_and_()` were introduced in PyTorch 1.5.0.
    def logical_or_in_place(a, b):
        a.logical_or_(b)

    def logical_and_in_place(a, b):
        a.logical_and_(b)

else:

    def logical_or_in_place(a, b):
        a |= b

    def logical_and_in_place(a, b):
        a &= b

if hasattr(torch, 'amax'):
    # For bool tensors, `amax()` does the same thing as logical or.
    logical_or_block = max_block
else:
    # Note that PyTorch <1.2.0 uses the dtype uint8 instead of bool.
    _bool_dtype = (torch.tensor(1, device='cpu') < torch.tensor(2, device='cpu')).dtype
    if _bool_dtype == torch.bool:
        def logical_or_block(a, dims):
            return sum_block(a, dims)
    else:
        def logical_or_block(a, dims):
            # Using > 0 clamps values to 1.
            return sum_block(a, dims) > 0
