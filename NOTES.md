## Minimum Python and PyTorch versions

As of writing, `torch_semiring_einsum` requires at least PyTorch 1.1.0 because
of its use of `torch.finfo().min` and `torch.finfo().max`. Although
`torch.finfo()` was introduced in PyTorch 1.0.0, `.min` and `.max` were not
made available until 1.1.0.

Also, `torch_semiring_einsum` requires at least Python 3.6 because of its use
of f-string syntax. Note that the earliest version of Python that PyTorch
1.1.0 supports is Python 3.5. For Python 3.6, PyTorch requires at least 3.6.1
because of an [ABI bug](https://github.com/pytorch/pytorch/issues/14931)
between 3.6.0 and later.
