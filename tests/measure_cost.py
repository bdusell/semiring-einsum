import argparse
import datetime
import pathlib
import json

import torch

from torch_semiring_einsum import compile_equation, einsum

def main():

    parser = argparse.ArgumentParser(
        description=
        'Generate data for the time and space complexity plots included in '
        'the documentation.'
    )
    parser.add_argument('output', type=pathlib.Path)
    parser.add_argument('-A', type=int, default=10)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--step-size', type=int, default=10000)
    parser.add_argument('--block-sizes', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 50])
    args = parser.parse_args()

    device = torch.device('cuda')
    EQUATION_STR = 'ak,ak,ak->a'
    COMPILED_EQUATION = compile_equation(EQUATION_STR)

    block_sizes = args.block_sizes + ['unbounded']

    pytorch_results = []
    blocked_results = [ (block_size, []) for block_size in block_sizes ]
    for i in range(args.steps + 1):
        # Run the first iteration twice to warm things up
        ignore = (i == 0)
        if ignore:
            i = 1
        K = i * args.step_size
        if ignore:
            print('warming up')
        else:
            print(f'K = {K}')
        x1, x2, x3 = [torch.rand((args.A, K), device=device) for _ in range(3)]
        torch.cuda.synchronize(device)
        base_memory = torch.cuda.memory_allocated(device)

        def measure_cost(einsum, equation):
            torch.cuda.synchronize(device)
            torch.cuda.reset_max_memory_allocated(device)
            start_time = datetime.datetime.now()
            y = einsum(equation, x1, x2, x3)
            torch.cuda.synchronize(device)
            duration = (datetime.datetime.now() - start_time).total_seconds()
            memory = torch.cuda.max_memory_allocated(device) - base_memory
            return { 'K' : K, 'duration' : duration, 'memory' : memory }

        result = measure_cost(torch.einsum, EQUATION_STR)
        if not ignore:
            pytorch_results.append(result)
        for block_size, results in blocked_results:
            print(f'  block size = {block_size}')
            if block_size == 'unbounded':
                # Just use a big number.
                block_size = 9999999999999999
            result = measure_cost(lambda *args: einsum(*args, block_size=block_size), COMPILED_EQUATION)
            if not ignore:
                results.append(result)
    with args.output.open('w') as fout:
        json.dump({ 'pytorch' : pytorch_results, 'blocked' : blocked_results }, fout)

if __name__ == '__main__':
    main()
