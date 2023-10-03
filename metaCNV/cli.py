#! /usr/bin/env python3

import argparse
import sys
from _preprocess import entropy, metaCNV_data
from _ptr import learn_PTR_parameters
import os

def get_parser():

    def file_exists(filename):
        if not os.path.exists(filename):
            raise argparse.ArgumentTypeError(f"File {filename} does not exist")
        return filename

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest = 'command')

    entropy_parser = subparsers.add_parser('entropy')
    entropy_parser.add_argument('pileup', type = argparse.FileType('r'))
    entropy_parser.add_argument('--output', '-o', type = argparse.FileType('w'), default = sys.stdout)
    entropy_parser.set_defaults(func = entropy)


    process_parser = subparsers.add_parser('preprocess-data')
    process_parser.add_argument('bamfile', type = file_exists)
    process_parser.add_argument('--reference','-ref', type = file_exists, required=True)
    process_parser.add_argument('--output', '-o', type = str, required=True)
    process_parser.add_argument('--window-size','-w', type = int, default = 200)
    process_parser.add_argument('--min-quality', '-Q', type = int, default = 20)
    process_parser.add_argument('--subcontig-size', '-sub', type = int, default = 25000)
    process_parser.set_defaults(func = metaCNV_data)


    ptr_parser = subparsers.add_parser('learn-ptr')
    ptr_parser.add_argument('samples', type = file_exists, nargs='+')
    ptr_parser.add_argument('--contigs', '-c', type = str, nargs='+', required=True)
    ptr_parser.add_argument('--outprefix', '-o', type = str, required=True)
    ptr_parser.add_argument('--strand-bias-tol', '-sb', type = float, default = 0.1)
    ptr_parser.add_argument('--max-median-coverage-deviation', '-cov', type = float, default = 8)
    ptr_parser.set_defaults(func = learn_PTR_parameters)

    return parser


def main():

    parser = get_parser()
    args = parser.parse_args()

    try:
        args.func
    except AttributeError:
        parser.print_help()
        sys.exit(1)

    del args.command
    func = args.func; del args.func

    func(**vars(args))


if __name__ == '__main__':
    main()