import unittest
import torch
import torch_semiring_einsum as semiring
import numpy

class TestDerived(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')
        self.generator = torch.manual_seed(123)

    def test_tensordot(self):
        A, B, C, D, E, F = 2, 3, 5, 7, 11, 13
        for i, (x_size, y_size, inner_dims) in enumerate([
                ((A, B, C, D), (C, D, E, F), 2),
                ((A, B, C, D), (D, E, F), 1),
                ((A, B, C, D), (E, F), 0),
        ]):
            with self.subTest(i):
                x = torch.empty(x_size, device=self.device)
                x.uniform_(-10., 10., generator=self.generator)
                y = torch.empty(y_size, device=self.device)
                y.uniform_(-10., 10., generator=self.generator)
                
                semiring_out = semiring.tensordot(x, y, inner_dims, block_size=1)
                torch_out = torch.tensordot(x, y, inner_dims)
                numpy.testing.assert_allclose(semiring_out, torch_out, rtol=1e-3)

                semiring_out = semiring.tensordot(x, y, inner_dims, block_size=1, einsum=semiring.log_einsum)
                torch_out = torch.tensordot(x.exp(), y.exp(), inner_dims).log()
                numpy.testing.assert_allclose(semiring_out, torch_out, rtol=1e-2)

    def test_matmul(self):
        J, K, M, N, P = 2, 3, 5, 7, 11

        for i, (x_size, y_size) in enumerate([
                # ((J, 1, N, M), (K, M, P)), # from torch.matmul docs
                ((J, K, N, M), (K, M, P)),
                ((J, K, N, M), (M,)),
                ((M), (K, M, P)),
        ]):
            with self.subTest(i):
                x = torch.empty(x_size)
                x.uniform_(-10., 10., generator=self.generator)
                y = torch.empty(y_size)
                y.uniform_(-10., 10., generator=self.generator)
                
                semiring_out = semiring.matmul(x, y, block_size=1)
                torch_out = torch.matmul(x, y)
                numpy.testing.assert_allclose(semiring_out, torch_out, rtol=1e-3)

                semiring_out = semiring.matmul(x, y, block_size=1, einsum=semiring.log_einsum)
                torch_out = torch.matmul(x.exp(), y.exp()).log()
                numpy.testing.assert_allclose(semiring_out, torch_out, rtol=1e-2)

    def test_inner(self):
        A, B, C, D, E = 2, 3, 5, 7, 11

        for i, (x_size, y_size) in enumerate([
                ((A, B), (C, D, B)),
                ((A, B), ()),
                ((), (A, B)),
        ]):
            with self.subTest(i):
                x = torch.empty(x_size)
                x.uniform_(-10., 10., generator=self.generator)
                y = torch.empty(y_size)
                y.uniform_(-10., 10., generator=self.generator)
                
                semiring_out = semiring.inner(x, y, block_size=1)
                torch_out = torch.inner(x, y)
                numpy.testing.assert_allclose(semiring_out, torch_out, rtol=1e-3)

                semiring_out = semiring.inner(x, y, block_size=1, einsum=semiring.log_einsum)
                torch_out = torch.inner(x.exp(), y.exp()).log()
                numpy.testing.assert_allclose(semiring_out, torch_out, rtol=1e-2)

        
if __name__ == '__main__':
    unittest.main()
