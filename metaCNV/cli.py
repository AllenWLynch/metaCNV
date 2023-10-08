#! /usr/bin/env python3

import argparse
import sys
from _preprocess import entropy, metaCNV_data
from _gc_skew import main as gc_skew
from _gc_skew import get_ori_distance
import os

def get_parser():

    def file_exists(filename):
        if not os.path.exists(filename):
            raise argparse.ArgumentTypeError(f"File {filename} does not exist")
        return filename

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest = 'command')

    entropy_parser = subparsers.add_parser('pileup-entropy',
                                           help = 'From a samtools pileup file, extract entropy and coverage features. \n'
                                           'This command is used as part of the "extract-features" pipeline.')
    entropy_parser.add_argument('pileup', type = argparse.FileType('r'))
    entropy_parser.add_argument('--output', '-o', type = argparse.FileType('w'), default = sys.stdout)
    entropy_parser.set_defaults(func = entropy)


    process_parser = subparsers.add_parser('extract-features',
                                           help = 'Extract features for CNV calling from a bam file.')
    process_parser.add_argument('bamfile', type = file_exists)
    process_parser.add_argument('--reference','-ref', type = file_exists, required=True)
    process_parser.add_argument('--ori-distances', '-ori', type = file_exists, required=True)
    process_parser.add_argument('--output', '-o', type = str, required=True)
    process_parser.add_argument('--window-size','-w', type = int, default = 200)
    process_parser.add_argument('--min-quality', '-Q', type = int, default = 255)
    process_parser.set_defaults(func = metaCNV_data)


    gc_skew_parser = subparsers.add_parser('gc-skew',
                                           help = 'Calculate GC skew from a fasta file.')
    gc_skew_parser.add_argument('reference', type = file_exists)
    gc_skew_parser.add_argument('--output', '-o', type = str, required=True)
    gc_skew_parser.set_defaults(func = gc_skew)

    ori_distance_parser = subparsers.add_parser('ori-distance',
                                                help = 'Calculate distance from origin of replication from a GC skew file.')
    ori_distance_parser.add_argument('gc_skew_file', type = file_exists)
    ori_distance_parser.add_argument('--contig', '-c', type = str, required=True)
    ori_distance_parser.add_argument('--window-size','-w', type = int, default = 200)
    ori_distance_parser.add_argument('--reference','-ref', type = file_exists, required=True)
    ori_distance_parser.add_argument('--output', '-o', type = str, default = sys.stdout)
    ori_distance_parser.set_defaults(func = get_ori_distance)

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