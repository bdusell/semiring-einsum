import argparse
import datetime
import pathlib
import json

import torch

from torch_semiring_einsum import compile_equation, einsum

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=pathlib.Path)
    args = parser.parse_args()

    device = torch.device('cuda')
    EQUATION_STR = 'ak,ak,ak->a'
    COMPILED_EQUATION = compile_equation(EQUATION_STR)
    A = 10
    N = 10

    pytorch_results = []
    this_results = []
    for i in range(N+1):
        # Run the first iteration twice to warm up CUDA
        ignore = (i == 0)
        if ignore:
            i = 1
        K = i * 100
        if ignore:
            print('warming up')
        else:
            print(f'K = {K}')
        x1, x2, x3 = [torch.rand((A, K), device=device) for _ in range(3)]
        torch.cuda.synchronize(device)
        base_memory = torch.cuda.memory_allocated(device)

        def measure_cost(einsum, equation):
            torch.cuda.reset_max_memory_allocated(device)
            start_time = datetime.datetime.now()
            y = einsum(equation, x1, x2, x3)
            torch.cuda.synchronize(device)
            duration = (datetime.datetime.now() - start_time).total_seconds()
            memory = torch.cuda.max_memory_allocated(device) - base_memory
            return { 'K' : K, 'duration' : duration, 'memory' : memory }

        pytorch_result = measure_cost(torch.einsum, EQUATION_STR)
        this_result = measure_cost(einsum, COMPILED_EQUATION)
        if not ignore:
            pytorch_results.append(pytorch_result)
            this_results.append(this_result)
    with args.output.open('w') as fout:
        json.dump({ 'pytorch' : pytorch_results, 'this' : this_results }, fout)

if __name__ == '__main__':
    main()
