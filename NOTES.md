## Minimum Python and PyTorch versions

The function signatures in `torch_semiring_einsum` depend on some newer
features of the `typing` module (e.g. `Literal`) that were introduced in
Python 3.8. The earliest PyTorch version that is compatible with Python 3.8 is
PyTorch 1.4.0. As of writing, the actual code in `torch_semiring_einsum` can
work with versions as early as PyTorch 1.0.0 (`Tensor.finfo` was introduced in
1.0.0).
