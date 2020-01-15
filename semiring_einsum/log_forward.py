import torch

from .equation import Equation
from .extend import semiring_einsum_forward

def log_einsum_forward(
        equation: Equation,
        *args: torch.Tensor,
        block_size : int) -> torch.Tensor:
    r"""Log-space einsum, where addition :math:`a + b` is replaced with
    :math:`\log(\exp a + \exp b)`, and multiplication :math:`a \times b` is
    replaced with addition :math:`a + b`.

    :param equation: A pre-compiled equation.
    :param args: Input tensors. The number of input tensors must be compatible
        with ``equation``.
    :param block_size: Block size used to control memory usage.
    :return: Output of einsum.
    """
    def callback(compute_sum):
        num_reduced_vars = len(equation.reduce_input_to_output.reduced_variables)
        # Make an initial pass to compute the maximum terms.
        # max_values has the same size as the reduced variables.
        max_values = compute_sum(_max_in_place, _max_block, _add_in_place)
        # Resize max_values so it can broadcast with the shape
        # output_vars + reduced_vars.
        resized_max_values = max_values.view(
            list(max_values.size()) + [1] * num_reduced_vars)

        # Clipping to `min_float` fixes an edge case where all terms are -inf
        # (the problem is that (-inf - -inf) produces nan).
        min_float = max_values.new_tensor(torch.finfo(max_values.dtype).min)
        _max_in_place(max_values, min_float)

        def sumexpsub_block(a, dims):
            a.sub_(resized_max_values)
            a.exp_()
            return torch.sum(a, dim=dims)

        # Now compute the logsumexp.
        # This implements y = max(x) + log \sum_i exp(x_i - max(x))
        result = compute_sum(_add_in_place, sumexpsub_block, _add_in_place)
        result.log_()
        result.add_(max_values)
        return result

    return semiring_einsum_forward(equation, args, block_size, callback)

def _max_in_place(a, b):
    torch.max(a, b, out=a)

def _max_block(a, dims):
    result = a
    for dim in reversed(dims):
        result = torch.max(result, dim=dim).values
    return result

def _add_in_place(a, b):
    a.add_(b)
