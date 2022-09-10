import math
import unittest

import numpy
import torch

from torch_semiring_einsum import (
    compile_equation,
    real_einsum_forward,
    real_einsum_backward,
    einsum,
    log_einsum_forward,
    log_einsum_backward,
    log_einsum,
    log_viterbi_einsum_forward,
    AutomaticBlockSize)

EQUATION_STR = 'abce,abde,abdf->acd'
A, B, C, D, E, F = 2, 3, 5, 7, 11, 13
SIZES = [(A, B, C, E), (A, B, D, E), (A, B, D, F)]
OUTPUT_SIZE = (A, C, D)

class TestCompileEquation(unittest.TestCase):

    def test_compile_equation(self):
        equation = compile_equation(EQUATION_STR)
        equation.prepare_for_forward()
        equation.prepare_for_backward()

class TestSemiringEinsum(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cpu')
        self.generator = torch.manual_seed(123)

    def assert_is_finite(self, tensor, message=None):
        self.assertTrue(torch.all(torch.isfinite(tensor)).item(), message)

    def test_real_einsum_forward(self):
        args = [
            torch.rand(size, device=self.device, generator=self.generator)
            for size in SIZES
        ]
        expected_result = torch.einsum(EQUATION_STR, *args)
        self.assertEqual(expected_result.size(), OUTPUT_SIZE)
        result = real_einsum_forward(
            compile_equation(EQUATION_STR),
            *args,
            block_size=3)
        self.assertEqual(result.size(), OUTPUT_SIZE)
        numpy.testing.assert_allclose(result, expected_result, rtol=1e-6)

    def test_real_einsum_backward(self):
        args = [
            torch.nn.Parameter(torch.rand(
                size, device=self.device, generator=self.generator))
            for size in SIZES
        ]
        grad = torch.empty(OUTPUT_SIZE, device=self.device)
        grad.uniform_(-5.0, 5.0, generator=self.generator)
        expected_output = torch.einsum(EQUATION_STR, *args)
        expected_output.backward(grad)
        expected_grads = [arg.grad.clone() for arg in args]
        arg_grads = real_einsum_backward(
            compile_equation(EQUATION_STR),
            [arg.detach() for arg in args],
            [True for arg in args],
            grad,
            block_size=3)
        for arg_grad, arg_size in zip(arg_grads, SIZES):
            self.assertEqual(arg_grad.size(), arg_size)
        for arg_grad, expected_grad in zip(arg_grads, expected_grads):
            numpy.testing.assert_allclose(arg_grad, expected_grad, rtol=1e-3)

    def test_einsum(self):
        args = [
            torch.nn.Parameter(torch.rand(
                size, device=self.device, generator=self.generator))
            for size in SIZES
        ]
        expected_output = torch.einsum(EQUATION_STR, *args)
        expected_loss = expected_output.sum()
        expected_loss.backward()
        expected_grads = [arg.grad.clone() for arg in args]
        for arg in args:
            arg.grad.zero_()
        output = einsum(
            compile_equation(EQUATION_STR),
            *args,
            block_size=3)
        numpy.testing.assert_allclose(output.detach(), expected_output.detach(), rtol=1e-6)
        loss = output.sum()
        loss.backward()
        grads = [arg.grad.clone() for arg in args]
        for grad, expected_grad in zip(grads, expected_grads):
            numpy.testing.assert_allclose(grad, expected_grad, rtol=1e-6)

    def test_log_einsum_forward(self):
        args = [
            torch.empty(size, device=self.device)
            for size in SIZES
        ]
        for arg in args:
            arg.uniform_(-10.0, 10.0, generator=self.generator)
        exp_args = [torch.exp(arg) for arg in args]
        exp_result = torch.einsum(EQUATION_STR, *exp_args)
        expected_result = torch.log(exp_result)
        self.assertEqual(expected_result.size(), OUTPUT_SIZE)
        result = log_einsum_forward(
            compile_equation(EQUATION_STR),
            *args,
            block_size=3)
        self.assertEqual(result.size(), OUTPUT_SIZE)
        numpy.testing.assert_allclose(result, expected_result)

    def test_log_einsum_backward(self):
        args = [
            torch.nn.Parameter(torch.empty(size, device=self.device))
            for size in SIZES
        ]
        for arg in args:
            arg.data.uniform_(-10.0, 10.0, generator=self.generator)
        grad = torch.empty(OUTPUT_SIZE, device=self.device)
        grad.uniform_(-5.0, 5.0, generator=self.generator)
        exp_args = [torch.exp(arg) for arg in args]
        exp_result = torch.einsum(EQUATION_STR, *exp_args)
        expected_output = torch.log(exp_result)
        expected_output.backward(grad)
        expected_grads = [arg.grad.clone() for arg in args]
        arg_grads = log_einsum_backward(
            compile_equation(EQUATION_STR),
            [arg.detach() for arg in args],
            [True for arg in args],
            grad,
            block_size=3)
        for arg_grad, arg_size in zip(arg_grads, SIZES):
            self.assertEqual(arg_grad.size(), arg_size)
        for arg_grad, expected_grad in zip(arg_grads, expected_grads):
            numpy.testing.assert_allclose(arg_grad, expected_grad, rtol=1e-3)

    def test_log_einsum(self):
        args = [
            torch.nn.Parameter(torch.rand(
                size, device=self.device, generator=self.generator))
            for size in SIZES
        ]
        expected_output = torch.einsum(EQUATION_STR, *args)
        expected_loss = expected_output.sum()
        expected_loss.backward()
        expected_grads = [arg.grad.clone() for arg in args]
        for arg in args:
            arg.grad.zero_()
        log_output = log_einsum(
            compile_equation(EQUATION_STR),
            *[torch.log(arg) for arg in args],
            block_size=3)
        output = torch.exp(log_output)
        numpy.testing.assert_allclose(output.detach(), expected_output.detach(), rtol=1e-6)
        loss = output.sum()
        loss.backward()
        grads = [arg.grad.clone() for arg in args]
        for grad, expected_grad in zip(grads, expected_grads):
            numpy.testing.assert_allclose(grad, expected_grad, rtol=1e-6)

    def test_log_einsum_overflow(self):
        # Test that log einsum does not overflow when dealing with large
        # values.
        args = [
            torch.nn.Parameter(torch.empty(size, device=self.device))
            for size in SIZES
        ]
        for arg in args:
            arg.data.uniform_(0.0, 100.0, generator=self.generator)
        # Make sure the arguments would cause exp() to overflow.
        for arg in args:
            self.assertTrue(torch.any(torch.isinf(torch.exp(arg))).item())
        output = log_einsum(
            compile_equation(EQUATION_STR),
            *args,
            block_size=3)
        # The output should not have inf or nan.
        self.assert_is_finite(output)
        loss = output.sum()
        loss.backward()
        for arg in args:
            # The gradients should not have inf or nan.
            self.assert_is_finite(arg.grad)

    def test_log_einsum_overflow_inf(self):
        # Test that log einsum does return inf (not nan) when dealing
        # with extremely large values.
        eq = compile_equation(',->')
        out = log_einsum(eq, torch.tensor(2e38), torch.tensor(2e38), block_size=1)
        self.assertEqual(out.item(), math.inf)

    def test_log_einsum_forward_all_neg_inf(self):
        # Test the behavior of the forward pass of log einsum when all of the
        # inputs are -inf. The output should be -inf.
        args = [
            torch.full(size, -math.inf, device=self.device)
            for size in SIZES
        ]
        output = log_einsum_forward(
            compile_equation(EQUATION_STR),
            *args,
            block_size=3)
        self.assertTrue(torch.equal(output, torch.full(OUTPUT_SIZE, -math.inf, device=self.device)))

    def test_log_einsum_backward_all_neg_inf(self):
        # Test the behavior of the backward pass of log einsum when all of the
        # inputs are -inf. The gradient of logsumexp is softmax. The behavior
        # of PyTorch's builtin logsumexp is to return NaN gradients (because
        # the denominator of the softmax is 0).
        args = [
            torch.full(size, -math.inf, device=self.device)
            for size in SIZES
        ]
        grad = torch.ones(OUTPUT_SIZE, device=self.device)
        arg_grads = log_einsum_backward(
            compile_equation(EQUATION_STR),
            args,
            [True for arg in args],
            grad,
            block_size=3)
        for arg_grad, size in zip(arg_grads, SIZES):
            self.assertEqual(arg_grad.size(), size)
            self.assertTrue(torch.all(torch.isnan(arg_grad)).item(), 'gradient should be nan')
        # Test the option that sets the gradient to 0.
        arg_grads = log_einsum_backward(
            compile_equation(EQUATION_STR),
            args,
            [True for arg in args],
            grad,
            block_size=3,
            grad_of_neg_inf=0.0)
        for arg_grad, size in zip(arg_grads, SIZES):
            numpy.testing.assert_allclose(arg_grad, torch.zeros(size, device=self.device))

    def test_grad_of_neg_inf_option(self):
        # Test that the grad_of_neg_inf option works with log_einsnum.
        args = [
            torch.nn.Parameter(torch.full(size, -math.inf, device=self.device))
            for size in SIZES
        ]
        output = log_einsum(
            compile_equation(EQUATION_STR),
            *args,
            block_size=3,
            grad_of_neg_inf=0.0)
        self.assertTrue(torch.equal(output, torch.full(OUTPUT_SIZE, -math.inf, device=self.device)))
        loss = output.sum()
        loss.backward()
        for arg, size in zip(args, SIZES):
            numpy.testing.assert_allclose(arg.grad, torch.zeros(size, device=self.device))

    def test_log_einsum_edge_cases(self):
        # Test the behavior of log_einsum when inputs are inf, -inf, nan, etc.
        EQUATION_STR = 'ab,ab->a'
        A, B = 7, 7
        SIZES = [(A, B), (A, B)]
        OUTPUT_SIZE = (A,)
        args = [
            torch.zeros(size, device=self.device)
            for size in SIZES
        ]
        expected_output = torch.zeros(OUTPUT_SIZE, device=self.device)
        expected_grads = [
            torch.zeros(size, device=self.device)
            for size in SIZES
        ]

        neg_inf_grad = 0.0

        # 0. Set all inputs to 5.
        for arg in args:
            arg[0] = 5.0
        expected_output[0] = math.log(B * math.exp(10.0))
        for grad in expected_grads:
            grad[0] = 1.0 / B

        # 1. Set all inputs to -inf.
        for arg in args:
            arg[1] = -math.inf
        expected_output[1] = -math.inf
        for grad in expected_grads:
            grad[1] = neg_inf_grad

        # 2. Set one input to -inf, and the other to 5.
        args[0][2] = -math.inf
        args[1][2] = 5.0
        expected_output[2] = -math.inf
        for grad in expected_grads:
            grad[2] = neg_inf_grad

        # 3. All terms are -inf, but neither input is all -inf.
        args[0][3, (0, 2, 3, 5)] = 5.0
        args[1][3, (1, 6)] = 5.0
        for arg in args:
            arg[3][arg[3] != 5.0] = -math.inf
        expected_output[3] = -math.inf
        for grad in expected_grads:
            grad[3] = neg_inf_grad

        # 4. Only one input is -inf.
        for arg in args:
            arg[4] = 5.0
        args[0][4, 3] = -math.inf
        expected_output[4] = math.log((B-1) * math.exp(10.0))
        for grad in expected_grads:
            grad[4] = 1.0 / (B-1)
            grad[4, 3] = 0.0

        # 5. One input is nan.
        for arg in args:
            arg[5] = 5.0
        args[0][5, 2] = math.nan
        expected_output[5] = math.nan
        for grad in expected_grads:
            grad[5] = math.nan

        # 6. One input is +inf.
        for arg in args:
            arg[6] = 5.0
        args[1][6, 5] = math.inf
        expected_output[6] = math.inf
        for grad in expected_grads:
            grad[6] = 0.0
            # This is like inf/inf, which is nan. torch.logsumexp treats sets
            # the gradient to nan here.
            grad[6, 5] = math.nan

        args = [torch.nn.Parameter(arg) for arg in args]
        output = log_einsum(
            compile_equation(EQUATION_STR),
            *args,
            block_size=3,
            grad_of_neg_inf=0.0
        )
        numpy.testing.assert_allclose(output.detach(), expected_output)
        output.sum().backward()
        for arg, expected_grad in zip(args, expected_grads):
            numpy.testing.assert_allclose(arg.grad, expected_grad)

    def test_log_einsum_save(self):
        # Test that log_einsum produces the same results when the save options
        # are used and when they are not used.
        args = [
            torch.nn.Parameter(torch.rand(
                size, device=self.device, generator=self.generator))
            for size in SIZES
        ]
        save_output = log_einsum(
            compile_equation(EQUATION_STR),
            *args,
            block_size=10,
            save_max=False,
            save_sumexpsub=False
        )
        save_output.sum().backward()
        save_grads = [arg.grad.clone() for arg in args]
        for arg in args:
            arg.grad.zero_()
        no_save_output = log_einsum(
            compile_equation(EQUATION_STR),
            *args,
            block_size=10,
            save_max=True,
            save_sumexpsub=True
        )
        no_save_output.sum().backward()
        numpy.testing.assert_allclose(save_output.detach(), no_save_output.detach())
        no_save_grads = [arg.grad.clone() for arg in args]
        for save_grad, no_save_grad in zip(save_grads, no_save_grads):
            numpy.testing.assert_allclose(save_grad, no_save_grad)

    def test_log_viterbi_einsum_forward(self):
        args = [
            torch.empty(size, device=self.device)
            for size in SIZES
        ]
        for arg in args:
            arg.uniform_(-10.0, 10.0, generator=self.generator)
        expected_maxval, expected_argmax = reference_log_viterbi_einsum(
            *args, self.device)
        self.assertEqual(expected_maxval.size(), OUTPUT_SIZE)
        self.assertEqual(expected_argmax.size(), (*OUTPUT_SIZE, 3))
        maxval, argmax = log_viterbi_einsum_forward(
            compile_equation(EQUATION_STR),
            *args,
            block_size=3)
        self.assertEqual(expected_maxval.size(), OUTPUT_SIZE)
        self.assertEqual(expected_argmax.size(), (*OUTPUT_SIZE, 3))
        numpy.testing.assert_allclose(maxval, expected_maxval)
        self.assertTrue(torch.equal(argmax, expected_argmax))

    def test_log_viterbi_einsum_forward_auto_block_size(self):
        args = [
            torch.empty(size, device=self.device)
            for size in SIZES
        ]
        for arg in args:
            arg.uniform_(-10.0, 10.0, generator=self.generator)
        expected_maxval, expected_argmax = reference_log_viterbi_einsum(
            *args, self.device)
        self.assertEqual(expected_maxval.size(), OUTPUT_SIZE)
        self.assertEqual(expected_argmax.size(), (*OUTPUT_SIZE, 3))
        maxval, argmax = log_viterbi_einsum_forward(
            compile_equation(EQUATION_STR),
            *args,
            block_size=AutomaticBlockSize(mock_available_bytes=2000))
        self.assertEqual(expected_maxval.size(), OUTPUT_SIZE)
        self.assertEqual(expected_argmax.size(), (*OUTPUT_SIZE, 3))
        numpy.testing.assert_allclose(maxval, expected_maxval)
        self.assertTrue(torch.equal(argmax, expected_argmax))

    def test_log_viterbi_einsum_forward_no_summed_vars(self):
        # When there are no summed-out variables, the returned tensor
        # of argmaxes should have have a last dim with size zero.
        eq = compile_equation('a,a->a')
        x = torch.arange(5, dtype=torch.float32)
        y = torch.arange(5, dtype=torch.float32)
        m, am = log_viterbi_einsum_forward(eq, x, y, block_size=1)
        self.assertEqual(m.size(), (5,))
        self.assertEqual(am.size(), (5, 0))

    def test_zero_dim(self):
        eq = compile_equation('->')
        ans = einsum(eq, torch.tensor(1.0), block_size=1)
        self.assertAlmostEqual(ans.item(), 1.0)
        ans = log_einsum(eq, torch.tensor(2.0), block_size=1)
        self.assertAlmostEqual(ans.item(), 2.0)

    def test_zero_dim_result(self):
        eq = compile_equation('i,i->')
        ans, _ = log_viterbi_einsum_forward(eq, torch.tensor([0.,0.]), torch.tensor([0.,0.]), block_size=1)
        self.assertAlmostEqual(ans.item(), 0.0)

    def test_automatic_block_size_cuda(self):
        device = torch.device('cuda')
        args = [
            torch.rand(size, device=self.device, generator=self.generator).to(device)
            for size in SIZES
        ]
        expected_result = torch.einsum(EQUATION_STR, *args)
        self.assertEqual(expected_result.size(), OUTPUT_SIZE)
        # Allocate a big honkin' tensor to take up some GPU memory.
        gigabytes = 1.5
        num_floats = int((gigabytes * (1 << 30)) // 4)
        big_tensor = torch.empty(num_floats, device=device)
        result = real_einsum_forward(
            compile_equation(EQUATION_STR),
            *args,
            block_size=AutomaticBlockSize())
        self.assertEqual(result.size(), OUTPUT_SIZE)
        numpy.testing.assert_allclose(result.cpu(), expected_result.cpu(), rtol=1e-6)

    def test_automatic_block_size_mock(self):
        device = self.device
        args = [
            torch.rand(size, device=device, generator=self.generator)
            for size in SIZES
        ]
        expected_result = torch.einsum(EQUATION_STR, *args)
        self.assertEqual(expected_result.size(), OUTPUT_SIZE)
        for mock_available_bytes in range(700, 2000+1, 100):
            with self.subTest(mock_available_bytes):
                result = real_einsum_forward(
                    compile_equation(EQUATION_STR),
                    *args,
                    block_size=AutomaticBlockSize(mock_available_bytes=mock_available_bytes))
                self.assertEqual(result.size(), OUTPUT_SIZE)
                numpy.testing.assert_allclose(result, expected_result, rtol=1e-6)

def reference_log_viterbi_einsum(X1, X2, X3, device):
    Y_max = []
    Y_argmax = []
    for a in range(A):
        Y_max_a = []
        Y_argmax_a = []
        for c in range(C):
            Y_max_ac = []
            Y_argmax_ac = []
            for d in range(D):
                maxval = torch.tensor(-math.inf, device=device)
                argmax = None
                for b in range(B):
                    for e in range(E):
                        for f in range(F):
                            val = (
                                X1[a, b, c, e]
                                + X2[a, b, d, e]
                                + X3[a, b, d, f]
                            )
                            if val > maxval:
                                maxval = val
                                argmax = torch.tensor([b, e, f], device=device)
                Y_max_ac.append(maxval)
                Y_argmax_ac.append(argmax)
            Y_max_a.append(Y_max_ac)
            Y_argmax_a.append(Y_argmax_ac)
        Y_max.append(Y_max_a)
        Y_argmax.append(Y_argmax_a)
    return recursively_stack(Y_max), recursively_stack(Y_argmax)

def recursively_stack(tensors):
    if isinstance(tensors, torch.Tensor):
        return tensors
    else:
        return torch.stack([recursively_stack(x) for x in tensors])

if __name__ == '__main__':
    unittest.main()
