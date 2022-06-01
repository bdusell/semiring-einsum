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

def clip_max_values(max_values):
    # Clipping to the min/max float fixes an edge case where all terms are -inf/+inf
    # (the problem is that (-inf - -inf) or (+inf - +inf) produces nan).
    max_values.copy_(torch.nan_to_num(max_values))

def resize_max_values(max_values, num_reduced_vars):
    # Resize max_values so it can broadcast with the shape
    # output_vars + reduced_vars.
    return max_values.view(list(max_values.size()) + [1] * num_reduced_vars)
