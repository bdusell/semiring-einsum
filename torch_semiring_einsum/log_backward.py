import itertools
import math
import typing

import torch
import typing_extensions

from .equation import (
    Equation,
    AutomaticBlockSize,
    AUTOMATIC_BLOCK_SIZE,
    get_summed_variable_indexes
)
from .extend import (
    semiring_einsum_forward_impl,
    reduce_in_place,
    adjust_size
)
from .utils import (
    add_in_place,
    sum_block
)
from .log_forward import (
    compute_max,
    compute_sumexpsub
)

def log_einsum_backward(
        equation: Equation,
        args: typing.Sequence[torch.Tensor],
        needs_grad: typing.Sequence[bool],
        grad: torch.Tensor,
        block_size: typing.Union[int, AutomaticBlockSize]=AUTOMATIC_BLOCK_SIZE,
        grad_of_neg_inf: typing.Union[float, typing_extensions.Literal['uniform']]=math.nan,
        saved_max: typing.Optional[torch.Tensor]=None,
        saved_sumexpsub: typing.Optional[torch.Tensor]=None
    ) -> typing.List[typing.Optional[torch.Tensor]]:
    r"""Compute the derivative of
    :py:func:`~torch_semiring_einsum.log_einsum_forward`.

    Like the forward pass, the backward pass is done in memory-efficient
    fashion by doing summations in fixed-size chunks.

    :param equation: Pre-compiled einsum equation. The derivative of the
        log-space einsum operation specified by this equation will be computed.
    :param args: The inputs to the log-space einsum operation whose derivative
        is being computed.
    :param needs_grad: Indicates which inputs in ``args`` require gradient.
    :param grad: The gradient of the loss function with respect to the output
        of the log-space einsum operation.
    :param block_size: Block size used to control memory usage.
    :param grad_of_neg_inf: How to handle the gradient of cases where all
        inputs to a logsumexp are :math:`-\infty`, which results in an output
        of :math:`-\infty`. The default behavior is to output NaN, which
        matches the behavior of PyTorch's :py:func:`~torch.logsumexp`, but
        sometimes this is not desired. If a :py:class:`float` is provided, all
        gradients will be set to that value. A value of ``0``, which causes the
        inputs not to change, may be appropriate. For example, if one input is
        a parameter and another is a constant :math:`-\infty`, it may not make
        sense to try to change the parameter. This is what the equivalent real
        space operation would do (the derivative of :math:`0x` with respect to
        :math:`x` is :math:`0`). On the other hand, if the string ``'uniform'``
        is provided, the gradient will be set to a uniform distribution that
        sums to 1. This makes sense because the gradient of logsumexp is
        softmax, and in this case it will attempt to increase the inputs to the
        logsumexp above :math:`-\infty`. NOTE: Only NaN and 0 are currently
        implemented.
    :param saved_max: See ``return_max`` in
        :py:func:`~torch_semiring_einsum.log_einsum_forward`.
    :param saved_sumexpsub: See ``return_sumexpsub`` in
        :py:func:`~torch_semiring_einsum.log_einsum_forward`.
    :return: The gradients with respect to each of the inputs to the log-space
        einsum operation. Returns ``None`` for inputs that do not require
        gradient.
    """
    # grad : same size as output of equation
    if len(args) != len(needs_grad):
        raise ValueError('length of args is not equal to length of needs_grad')
    equation.validate_sizes(args)
    equation.prepare_for_forward()
    equation.prepare_for_log_backward()
    output_size = tuple(equation.get_sizes(args, equation.output_variables))
    grad_size = tuple(grad.size())
    if grad_size != output_size:
        raise ValueError(
            'size of gradient {} does not match expected size {}'.format(
                grad_size, output_size))

    # The gradient of logsumexp is softmax (logsumexp is a soft version of max,
    # and softmax is a soft version of argmax). So essentially we're computing
    # a softmax here. In order to avoid overflow in the exp() function, we need
    # to exploit the identity
    #     \frac{ \exp(x) }{ \sum_{x'} \exp(x') } =
    #         \frac{ \exp(x-c) }{ \sum_{x'} \exp(x'-c) }
    # where c = \max_{x'} x'.
    # Z is the denominator of the softmax. We first do a separate pass through
    # the inputs to compute the maximums, then we use those to compute Z and
    # later the numerators.

    # max_values : same size as output of equation
    if saved_max is None:
        max_values = compute_max(equation, args, block_size)
    else:
        max_values = saved_max

    # Z : same size as output of equation
    if saved_sumexpsub is None:
        Z = compute_sumexpsub(equation, args, block_size, max_values)
    else:
        Z = saved_sumexpsub

    # C : same size as output of equation
    if isinstance(grad_of_neg_inf, float):
        if math.isnan(grad_of_neg_inf):
            # Wherever Z is 0, let the gradient be nan.
            # Dividing grad by 0 here actually does not result in nan, but
            # +inf. However, the +inf will later result in nan when C is
            # multiplied with `term`, because 0 * inf is nan.
            C = grad / Z
        else:
            # Whenever Z is 0, use the value of grad_of_neg_inf as the gradient
            # instead. Remember to multiply it by the incoming gradient.
            if grad_of_neg_inf == 0.0:
                # For 0 we can get away with just setting C to 0 wherever Z is
                # 0, but it's more complicated for other values.
                C = grad / Z
                C[Z == 0.0] = 0.0
            else:
                raise NotImplementedError(
                    'setting grad_of_neg_inf to a constant other than 0 is not '
                    'implemented')
    elif grad_of_neg_inf == 'uniform':
        # Whenever Z is 0, set the gradient to a uniform distribution that sums
        # to 1. Each input can have a different value for the gradient. The
        # value of the gradient is equal to the product of the sizes of all the
        # summed variables not in that input, divided by the product of the
        # sizes of all summed variables.
        # Remember to multiply the result by the incoming gradient.
        raise NotImplementedError('grad_of_neg_inf=\'uniform\' is not implemented')
    else:
        raise ValueError(f'invalid choice for grad_of_neg_inf: {grad_of_neg_inf}')
    del Z

    arg_grads = []
    for i, arg in enumerate(args):
        if needs_grad[i]:
            reduce_info, output_lookup_info = equation.reduce_all_to_input[i]

            # In this outer loop, we need to sum over all dimensions that
            # appear in the output but not in arg i. This is due to a basic
            # rule of multivariable calculus. If the summed variable is k, then
            # the gradient of the loss function L wrt arg i is the partial
            # derivative of L wrt the output at index k (which is found in
            # `grad`), times the partial derivative of the output at index k
            # wrt arg i, summed over all values of k.
            # This outer loop is simulataneously summing over all variables
            # that appear in other inputs but not in the output or in arg i.
            # This is necessary because if the summed variable is l, then arg i
            # appears in a term of the logsumexp for each value of l.
            # It would be possible to split this outer loop into two nested
            # loops, one that sums over variables that are in the output but
            # not arg i, and an inner one that sums over variables not in the
            # output or arg i. Currently they are done in the same loop.
            # This loop is *not* computing the denominator of the softmax; that
            # was already done above in Z.
            def generate_terms():
                for var_values in reduce_info.get_summed_variable_indexes(equation, args, block_size):

                    # This inner loop adds tensor slices together to get a
                    # term to be used in the outer loop.
                    def generate_factors():
                        for arg, arg_info in zip(args, reduce_info.lookup_info):
                            yield arg_info.lookup(arg, var_values)

                    term_size = reduce_info.get_term_size(equation, args, var_values)
                    term = reduce_in_place(
                        add_in_place,
                        generate_factors(),
                        lambda x: adjust_size(x, term_size))
                    # Subtract the maximum values to avoid overflow in exp().
                    term.sub_(output_lookup_info.lookup(max_values, var_values))
                    term.exp_()
                    # TODO An advantage of splitting the outer loop into two
                    # nested loops is that this multiplication could be moved
                    # outside the inner loop.
                    # As it is, this multiplication cannot be moved outside
                    # this loop, because var_values might range over a
                    # dimension in the output.
                    # If C is +inf here (because Z was 0), then this will
                    # result in nan, because term will be 0 and 0 * inf is nan.
                    term.mul_(output_lookup_info.lookup(C, var_values))
                    yield sum_block(term, reduce_info.reduced_dims)

            arg_grad = reduce_in_place(add_in_place, generate_terms())
        else:
            arg_grad = None
        arg_grads.append(arg_grad)
    return arg_grads
