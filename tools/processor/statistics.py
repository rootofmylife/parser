# -*- coding: utf-8 -*-

from tools.utils import Config

def stats(**kwargs):
    args = Config(**locals())

    sentences = 0
    words = []
    pos = []

    with open(args.input_file) as fin:
        for line in fin:
            if not line.strip().startswith("#"):
                if len(line.strip()) > 0:
                    tokens = line.strip().split('\t')
                    w = tokens[1] # word
                    p = tokens[4] # pos

                    if w not in words:
                        words.append(w)
                    if p not in pos:
                        pos.append(p)
                else:
                    sentences += 1
    
    print('Number of sentences: {}'.format(sentences))
    print('Number of words: {}'.format(len(words)))
    print('Number of pos: {}'.format(len(pos)))

    if args.print_pos:
        print('List of pos: {}'.format(pos))
