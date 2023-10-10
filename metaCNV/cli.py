#! /usr/bin/env python3

from metaCNV import call_CNVs
from metaCNV._preprocess import entropy, metaCNV_data
from metaCNV._gc_skew import main as gc_skew
from metaCNV._gc_skew import get_ori_distance
from metaCNV._mutation_rate_estimation import get_mutation_rate
import argparse
import sys
import os

def get_parser():

    def file_exists(filename):
        if not os.path.exists(filename):
            raise argparse.ArgumentTypeError(f"File {filename} does not exist")
        return filename

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest = 'command', metavar='[command]')

    entropy_parser = subparsers.add_parser('pileup-entropy')
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
                                                help = 'For a whole genome, calculate distance from origin of replication from a GC skew file.')
    ori_distance_parser.add_argument('gc_skew_file', type = file_exists)
    ori_distance_parser.add_argument('--contig', '-c', type = str, required=True)
    ori_distance_parser.add_argument('--reference','-ref', type = file_exists, required=True)
    ori_distance_parser.add_argument('--window-size','-w', type = int, default = 200)
    ori_distance_parser.add_argument('--output', '-o', type = argparse.FileType('w'), default = sys.stdout)
    ori_distance_parser.set_defaults(func = get_ori_distance)

    mut_parser = subparsers.add_parser('mutation-rate',
                                        help = 'Calculate mutation rate from a collection of *ALIGNED* fasta files.')
    mut_parser.add_argument('fastas', type = file_exists, nargs = '+')
    mut_parser.add_argument('--output', '-o', type = argparse.FileType('w'), default = sys.stdout)
    mut_parser.add_argument('--window-size','-w', type = int, default = 200)
    mut_parser.add_argument('--n-jobs', '-j', type = int, default = 1)
    mut_parser.add_argument('--region', '-r', type = str, default = None)
    mut_parser.set_defaults(func = get_mutation_rate)


    cnv_parser = subparsers.add_parser('call', help = 'Call CNVs from a metaCNV file.')
    req = cnv_parser.add_argument_group('Required arguments')
    req.add_argument('metaCNV_file', type = file_exists)
    req.add_argument('--mutation_rates', '-mut', type = file_exists, required=True)
    req.add_argument('--outprefix', '-o', type = str, required=True)
    req.add_argument('--contigs', '-c', type = str, nargs = '+', required=True)

    tmat_group = cnv_parser.add_argument_group('Transition matrix arguments')
    tmat = tmat_group.add_mutually_exclusive_group()
    tmat.add_argument('--transition-matrix', '-tm', type = file_exists)
    tmat.add_argument('--alpha', '-a', type = float, default = 5e7, 
                      help = 'Regularization parameter for basic sparse transition matrix.\nHigher --> fewer ploidy transitions.')

    model_params = cnv_parser.add_argument_group('Model parameters')
    model_params.add_argument('--max-fdr', '-fdr', type = float, default = 0.05)
    model_params.add_argument('--coverage-weight', '-cw', type = float, default = 0.8)
    model_params.add_argument('--entropy-df', '-df', type = int, default = 20)
    model_params.add_argument('--max-strains', '-ms', type = int, default = 3)

    data_params = cnv_parser.add_argument_group('Data loading parameters')
    data_params.add_argument('--strand-bias-tol', '-sb', type = float, default = 0.1)
    data_params.add_argument('--min-entropy-positions', '-ep', type = float, default = 0.9)
    cnv_parser.set_defaults(func = call_CNVs)

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