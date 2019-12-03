import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy

def parse_type(s):
    if s not in ('space', 'time'):
        raise ValueError
    return s

def read_data(data, impl, key):
    impl_data = data[impl]
    x = numpy.array([p['K'] for p in impl_data])
    y = numpy.array([p[key] for p in impl_data])
    return x, y

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=pathlib.Path)
    parser.add_argument('output', type=pathlib.Path)
    parser.add_argument('--type', required=True, type=parse_type)
    args = parser.parse_args()

    with args.input.open() as fin:
        data = json.load(fin)
    key = 'memory' if args.type == 'space' else 'duration'
    pytorch_x, pytorch_y = read_data(data, 'pytorch', key)
    this_x, this_y = read_data(data, 'this', key)

    fig, ax = plt.subplots()
    ax.set_title(f'{args.type.capitalize()} Complexity')
    ax.set_xlabel('K')
    ylabel = 'GPU memory (bytes)' if args.type == 'space' else 'Time (s)'
    ax.set_ylabel(ylabel)

    ax.plot(pytorch_x, pytorch_y, label='pytorch.einsum()')
    ax.plot(this_x, this_y, label='semiring_einsum.einsum()')
    ax.set_ylim(bottom=0)

    ax.legend()
    plt.savefig(args.output)

if __name__ == '__main__':
    main()
