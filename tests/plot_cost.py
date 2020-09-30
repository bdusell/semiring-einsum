import argparse
import json
import pathlib

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy

def force_integer_ticks(ax):
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 5]))

def parse_type(s):
    if s not in ('space', 'time'):
        raise ValueError
    return s

def read_data(data, key):
    x = numpy.array([p['K'] for p in data])
    y = numpy.array([p[key] for p in data])
    return x, y

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=pathlib.Path)
    parser.add_argument('output', type=pathlib.Path)
    parser.add_argument('--type', required=True, type=parse_type)
    parser.add_argument('--block-sizes', type=int, nargs='+', default=[])
    parser.add_argument('--no-pytorch', action='store_true', default=False)
    parser.add_argument('--no-unbounded', action='store_true', default=False)
    args = parser.parse_args()

    block_sizes = set(args.block_sizes)

    with args.input.open() as fin:
        data = json.load(fin)
    key = 'memory' if args.type == 'space' else 'duration'

    fig, ax = plt.subplots()
    ax.set_title(f'{args.type.capitalize()} Complexity')
    ax.set_xlabel('K')
    ylabel = 'GPU memory (bytes)' if args.type == 'space' else 'Time (s)'
    ax.set_ylabel(ylabel)

    if not args.no_pytorch:
        x, y = read_data(data['pytorch'], key)
        ax.plot(x, y, label='torch.einsum()', marker='x')
    for block_size, results in data['blocked']:
        if (
            block_size in block_sizes or
            (not args.no_unbounded and block_size == 'unbounded')
        ):
            x, y = read_data(results, key)
            ax.plot(x, y, label=f'torch_semiring_einsum.einsum(), block size {block_size}')

    ax.set_ylim(bottom=0)
    force_integer_ticks(ax)
    ax.legend()
    plt.tight_layout()
    plt.savefig(args.output)

if __name__ == '__main__':
    main()
