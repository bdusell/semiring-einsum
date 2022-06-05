import torch

def add_in_place(a, b):
    a.add_(b)

def multiply_in_place(a, b):
    a.mul_(b)

def max_in_place(a, b):
    torch.max(a, b, out=a)

def sum_block(a, dims):
    if dims:
        return torch.sum(a, dim=dims)
    else:
        # This is an edge case where PyTorch returns a bad result.
        # See https://github.com/pytorch/pytorch/issues/29137
        # On the plus side, this avoids creating an unnecessary copy of `a`.
        return a

def max_block(a, dims):
    result = a
    for dim in reversed(dims):
        result = torch.max(result, dim=dim).values
    return result
