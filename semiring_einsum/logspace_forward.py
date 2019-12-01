import torch

from .extend import semiring_einsum_forward, EquationForForward

def logspace_einsum_forward(
        equation: EquationForForward,
        *args: torch.Tensor) -> torch.Tensor:
    r"""Einsum where addition :math:`a + b` is replaced with
    :math:`\log(\exp a + \exp b)`, and multiplication :math:`a \times b` is
    replaced with addition :math:`a + b`.

    :param equation: A pre-compiled equation.
    :param args: Input tensors. The number of input tensors must be compatible
        with ``equation``.
    :return: Output of einsum.
    """
    return semiring_einsum_forward(equation, args, _callback)

def _callback(compute_sum):

    # Make an initial pass to compute the maximum terms.
    max_values = compute_sum(_max_in_place, _add_in_place)

    # Clipping to `min_float` fixes an edge case where all terms are -inf
    # (the problem is that (-inf - -inf) produces nan).
    min_float = max_values.new_tensor(torch.finfo(max_values.dtype).min)
    _max_in_place(max_values, min_float)

    def addexpsub_init(a):
        a.sub_(max_values)
        a.exp_()
        return a

    def addexpsub_in_place(a, b):
        b.sub_(max_values)
        b.exp_()
        a.add_(b)

    # Now compute the logsumexp.
    result = compute_sum(addexpsub_in_place, _add_in_place, addexpsub_init)
    result.log_()
    result.add_(max_values)
    return result

def _max_in_place(a, b):
    torch.max(a, b, out=a)

def _add_in_place(a, b):
    a.add_(b)
