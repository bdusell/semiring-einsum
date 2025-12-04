import random
import unittest

import torch
import numpy

from torch_semiring_einsum import (
    compile_equation,
    real_einsum_forward,
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

    def test_alternate_equation_api(self):
        equation_str = 'abce,abde,abdf->acd'
        equation = compile_equation(
            inputs=[[1, 'x', 'bar', 'foo'], [1, 'x', 4, 'foo'], [1, 'x', 4, 6]],
            output=[1, 'bar', 4])
        A, B, C, D, E, F = 2, 3, 5, 7, 11, 13
        sizes = [(A, B, C, E), (A, B, D, E), (A, B, D, F)]
        output_size = (A, C, D)
        args = [
            torch.rand(
                size, device=self.device, generator=self.generator)
            for size in sizes
        ]
        expected_result = torch.einsum(equation_str, *args)
        result = real_einsum_forward(
            equation,
            *args,
            block_size=3)
        self.assertEqual(result.size(), output_size)
        numpy.testing.assert_allclose(result, expected_result, rtol=1e-6)

    def test_more_than_26_indices(self):
        n = 64
        i = n // 2 - 10
        j = n // 2 + 10
        variables = list(range(n))
        python_generator = random.Random(123)
        python_generator.shuffle(variables)
        sizes = [1] * n
        sizes[n // 2 - 2] = 2
        sizes[n // 2 + 1] = 3
        sizes[n // 2 + 2] = 5
        a = torch.rand(sizes[:j], device=self.device, generator=self.generator)
        b = torch.rand(sizes[i:], device=self.device, generator=self.generator)
        equation = compile_equation(inputs=[variables[:j], variables[i:]], output=variables[::2])
        result = real_einsum_forward(equation, a, b, block_size=3)
        self.assertEqual(result.size(), tuple(sizes[::2]))

if __name__ == '__main__':
    unittest.main()
