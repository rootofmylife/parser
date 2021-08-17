# -*- coding: utf-8 -*-

from tools.utils import Config

def stats(**kwargs):
    args = Config(**locals())

    sentences = 0
    words = []
    pos = []
    rel = []

    with open(args.input_file) as fin:
        for line in fin:
            if not line.strip().startswith("#"):
                if len(line.strip()) > 0:
                    tokens = line.strip().split('\t')
                    w = tokens[1] # word
                    p = tokens[4] # pos
                    r = tokens[7] # relation

                    if w.lower() not in words:
                        words.append(w.lower())
                    if p not in pos:
                        pos.append(p)
                    if r not in rel:
                        rel.append(r)
                else:
                    sentences += 1
    
    print('Number of sentences: {}'.format(sentences))
    print('Number of words: {}'.format(len(words)))
    print('Number of pos tags: {}'.format(len(pos)))
    print('Number of relations: {}'.format(len(rel)))

    if args.print_pos:
        print('List of pos tags: {}'.format(pos))

    if args.print_rel:
        print('List of relations: {}'.format(rel))
