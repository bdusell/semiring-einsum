import math

import torch

def add_in_place(a, b):
    a.add_(b)

def multiply_in_place(a, b):
    a.mul_(b)

def max_in_place(a, b):
    # `torch.max` has been available since PyTorch 0.1.12.
    torch.max(a, b, out=a)

def sum_block(a, dims):
    if dims:
        # `dim` was first allowed to be a tuple in PyTorch 0.4.1.
        return torch.sum(a, dim=dims)
    else:
        # This is an edge case where PyTorch returns a bad result.
        # See https://github.com/pytorch/pytorch/issues/29137
        # On the plus side, this avoids creating an unnecessary copy of `a`.
        return a

def max_block(a, dims):
    result = a
    for dim in reversed(dims):
        # `torch.max` has been available since PyTorch 0.1.12.
        result = torch.max(result, dim=dim)[0]
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
        # `torch.finfo` was introduced in PyTorch 1.0.0.
        finfo = torch.finfo(a.dtype)
        min_float = a.new_tensor(finfo.min)
        torch.max(a, min_float, out=a)
        max_float = a.new_tensor(finfo.max)
        torch.min(a, max_float, out=a)

def amax(a, dim, keepdim=False):
    """Find the maximum of a tensor over zero or more dimensions."""
    try:
        # PyTorch >= 1.10
        result = torch.amax(a, dim=dim, keepdim=keepdim)
    except AttributeError:
        # PyTorch < 1.10
        result = a
        for dim in reversed(dims):
            result = torch.max(result, dim=dim, keepdim=keepdim).values
    return result

