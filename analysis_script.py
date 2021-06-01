import argparse
from analysis.analysis_funcs import *


def build_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resultsdir', required=True)
    parser.add_argument('--naivedir', help='Example: --naivedir enivronment/configs/naive/1622214885.172365')
    parser.add_argument('--modeldir', help='Example: --environment/fully_connected/1622048185.470065')
    parser.add_argument('--naivecompare', action='store_true')
    parser.add_argument('--savedir', default='results_analysis', required=False)
    return parser


def execute_actions(args):
    if args.naivecompare:
        assert args.naivedir is not None, 'Compare with naive is true but no path for naive model was given'
        assert args.modeldir is not None, 'Compare with naive is true but no path for the compared model was given'
        naive_compare(args.naivedir, args.modeldir, args.savedir)


if __name__ == '__main__':
    arg_parser = build_arguments()
    args = arg_parser.parse_args()
    execute_actions(args)
    print()
