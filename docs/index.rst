Semiring Einsum
===============

This is a Python package for PyTorch that implements einsum for alternative
semirings besides the usual "add-multiply" semiring, namely logspace and Viterbi.
It can be extended to support additional semirings relatively easily.

The einsum implementation in this package was also specifically designed
to be memory-efficient. Whereas a naive implementation of einsum could easily
consume huge amounts of memory, this implementation uses no more memory than
necessary by performing summations in-place, at the cost of some parallelism.
In some cases, it uses even less memory than the built-in :py:func:`torch.einsum`
function.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api

Installation
------------

.. code-block:: sh

   pip install git+git://github.com/bdusell/semiring-einsum.git

Basic Usage
-----------

.. code-block:: python

   import semiring_einsum

   EQUATION = semiring_einsum.compile_equation('ik,kj->ij')

   A = torch.log(torch.rand(3, 5))
   B = torch.log(torch.rand(5, 7))
   C = semiring_einsum.logspace_einsum(EQUATION, A, B)

Note that unlike in NumPy or PyTorch, equations are pre-compiled using
:py:func:`~semiring_einsum.compile_equation` rather than re-parsed from
scratch every time einsum is called.

API Documentation
-----------------

See :doc:`api`.

What is Einsum?
---------------

The so-called "einsum" function, offered in tensor math libraries such as
`NumPy <https://docs.scipy.org/doc/numpy/reference/generated/numpy.einsum.html>`_,
`TensorFlow <https://www.tensorflow.org/api_docs/python/tf/einsum>`_,
and `PyTorch <https://pytorch.org/docs/stable/torch.html#torch.einsum>`_,
is a function that can be used to express multi-dimensional, linear
algebraic tensor operations with a simple, concise syntax inspired by
`Einstein summation <https://en.wikipedia.org/wiki/Einstein_notation>`_.
It is a very useful kernel that can be used to implement other tensor
operations; for example, the matrix-matrix product of ``A`` and ``B`` can
be implemented as ::

    C = einsum('ik,kj->ij', A, B)

In this example, the first argument to the function is the "equation," and the
lower-case letters ``i``, ``j``, and ``k`` all serve as labels for dimensions
of the tensors ``A``, ``B``, and ``C``. The left side of the equation, ``ik,kj``,
describes the dimensions of the inputs, ``A`` and ``B``; the right side of the
equation, ``ij``, describes the desired shape of the output tensor ``C``. This
means that for each ``i`` and ``j``, entry ``C[i, j]`` will be formed by
multiplying elements from ``A[i, :]`` and ``B[:, j]``. Since the variable
``k`` does not appear in the output, it is "summed out," meaning that each
``C[i, j]`` is the result of computing ``A[i, k] * B[k, j]`` for each
``k``, then summing over the resulting terms.

.. math::

   C_{ij} = \sum_k A_{ik} \times B_{kj}

Einsum can also be used with three or more tensor arguments.

Semirings
---------

It is often useful to swap out addition and multiplication for different
operators that have the same algebraic properties as addition and
multiplication do on real numbers. We can express this using
`semirings <https://en.wikipedia.org/wiki/Semiring>`_. Changing the semiring
used by a piece of code can result in new, useful algorithms. For example,
the `Viterbi Algorithm <https://en.wikipedia.org/wiki/Viterbi_algorithm>`_
and the `Forward Algorithm <https://en.wikipedia.org/wiki/Forward_algorithm>`_
on Hidden Markov Models can be viewed as instances of the same algorithm
instantiated with different semirings.

For a formal definition of semirings and an introduction to semirings in the
context of context-free grammar parsing, see :cite:`goodman1999`.

Einsum Syntax
-------------

This package supports the same einsum equation syntax as
:py:func:`torch.einsum`, except it does not support ellipses (``...``) syntax.

Space Complexity
----------------

Consider the einsum equation ``'ak,ak,ak->a'``, where :math:`A` is the size of
the ``a`` dimension and :math:`K` is the size of the ``k`` dimension.
Implementations of einsum in NumPy and PyTorch contract two tensors at time,
which means that they must create an intermediate tensor of size
:math:`A \times K`. There is even a routine in NumPy,
:py:func:`numpy.einsum_path`, which figures out the best contraction order.
However, it should, in principle, be possible to avoid this by summing over
all tensors at the same time. This is exactly what ``semiring_einsum`` does,
and as a result the amount of scratch space the forward pass of einsum requires
remains fixed as a function of :math:`K`:

.. image:: space-complexity.png

It does, however, come at a cost in time:

.. image:: time-complexity.png

Indexes
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Bibliography
------------

.. bibliography:: references.bib
