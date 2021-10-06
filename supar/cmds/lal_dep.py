# -*- coding: utf-8 -*-

import argparse

from supar import LabelAttentionDependencyParser
from supar.cmds.cmd import parse


def main():
    parser = argparse.ArgumentParser(description='Create Biaffine Dependency Parser.')
    parser.add_argument('--tree', action='store_true', help='whether to ensure well-formedness')
    parser.add_argument('--proj', action='store_true', help='whether to projectivise the data')
    parser.add_argument('--partial', action='store_true', help='whether partial annotation is included')
    parser.set_defaults(Parser=LabelAttentionDependencyParser)
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    # train
    subparser = subparsers.add_parser('train', help='Train a parser.')
    subparser.add_argument('--feat', '-f', choices=['tag', 'char', 'elmo', 'bert'], nargs='+', help='features to use')
    subparser.add_argument('--build', '-b', action='store_true', help='whether to build the model first')
    subparser.add_argument('--checkpoint', action='store_true', help='whether to load a checkpoint to restore training')
    subparser.add_argument('--encoder', choices=['lstm', 'bert', 'lal'], default='lstm', help='encoder to use')
    subparser.add_argument('--punct', action='store_true', help='whether to include punctuation')
    subparser.add_argument('--max-len', type=int, help='max length of the sentences')
    subparser.add_argument('--buckets', default=32, type=int, help='max num of buckets to use')
    subparser.add_argument('--train', default='data/ptb/train.conllx', help='path to train file')
    subparser.add_argument('--dev', default='data/ptb/dev.conllx', help='path to dev file')
    subparser.add_argument('--test', default='data/ptb/test.conllx', help='path to test file')
    subparser.add_argument('--embed', help='path to pretrained embeddings')
    subparser.add_argument('--unk', default='unk', help='unk token in pretrained embeddings')
    subparser.add_argument('--n-embed', default=100, type=int, help='dimension of embeddings')
    subparser.add_argument('--bert', default='bert-base-cased', help='which BERT model to use')
    subparser.add_argument('--num-layer', default=3, help='Number of label attention layer')
    subparser.add_argument('--lal-d-kv', default=64, help='Dimension of Key and Query Vectors in the LAL')
    subparser.add_argument('--lal-d-proj', default=64, help='Dimension of the output vector from each label attention head')
    subparser.add_argument('--lal-resdrop', action='store_true', help='True means the LAL uses Residual Dropout')
    subparser.add_argument('--lal-pwff', action='store_true', help='True means the LAL has a Position-wise Feed-forward Layer')
    subparser.add_argument('--lal-q-as-matrix', action='store_true', help='False means the LAL uses learned query vectors')
    subparser.add_argument('--lal-partitioned', action='store_true', help='Partitioned as per the Berkeley Self-Attentive Parser')
    subparser.add_argument('--lal-combine-as-self', action='store_true', help='False means the LAL uses concatenation')
    # evaluate
    subparser = subparsers.add_parser('evaluate', help='Evaluate the specified parser and dataset.')
    subparser.add_argument('--punct', action='store_true', help='whether to include punctuation')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', default='code/data/ptb/test.conllx', help='path to dataset')
    # predict
    subparser = subparsers.add_parser('predict', help='Use a trained parser to make predictions.')
    subparser.add_argument('--prob', action='store_true', help='whether to output probs')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', default='code/data/ptb/test.conllx', help='path to dataset')
    subparser.add_argument('--pred', default='pred.conllx', help='path to predicted result')
    parse(parser)


if __name__ == "__main__":
    main()
