# -*- coding: utf-8 -*-

import argparse
from tools.cmds.cmd import parse

def main():
    parser = argparse.ArgumentParser(description='Tool for preprocessing conll format.')
    subparsers = parser.add_subparsers(title='Commands', dest='mode')

    # convert PoS tagset to another PoS tagset
    subparser = subparsers.add_parser('convert', help='Convert PoS tagset to another PoS tagset.')
    subparser.add_argument('--input-file', '-fi', help='Input file to convert PoS tagset.')
    subparser.add_argument('--output-file', '-fo', default='./output.conllx', help='Output file.')
    subparser.add_argument('--mapping-file', '-mp', help='Mapping old PoS tagset to new PoS tagset.')

    # concatenate all files into one file
    subparser = subparsers.add_parser('concat', help='Concatenate all files into one.')
    subparser.add_argument('--folder', '-fol', help='Input folder.')

    # statistic treebanks
    subparser = subparsers.add_parser('statistics', help='Statistic a treebanks.')
    subparser.add_argument('--input-file', '-fi', help='Input file to convert PoS tagset.')
    subparser.add_argument('--print-word', action='store_true', help='whether to print list of word.')
    subparser.add_argument('--print-pos', action='store_true', help='whether to print list of pos tags.')
    subparser.add_argument('--print-rel', action='store_true', help='whether to print list of relations.')

    parse(parser)

if __name__ == "__main__":
    main()