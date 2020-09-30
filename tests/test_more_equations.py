import unittest

import torch
import numpy

from torch_semiring_einsum import (
    compile_equation,
    log_einsum_backward)

class TestMoreEquations(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cpu')
        self.generator = torch.manual_seed(123)

    def test_abc_b_a(self):
        A, B, C = 3, 5, 7
        EQUATION_STR = 'abc,b->a'
        SIZES = [(A, B, C), (B,)]
        OUTPUT_SIZE = (A,)
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
            numpy.testing.assert_allclose(arg_grad, expected_grad, rtol=1e-5)

if __name__ == '__main__':
    unittest.main()
