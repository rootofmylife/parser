# -*- coding: utf-8 -*-

from tools.utils import Config

def permute(lines):
    lines = '\n'.join(lines)
    return lines + '\n'

def wordorder(**kwargs):
    args = Config(**locals())

    with open(args.output_file, 'w') as fout:
        with open(args.input_file) as fin:
            sentences = []
            for line in fin:
                if not line.strip().startswith("#"):
                    if len(line.strip()) > 0:
                        sentences.append(line.strip())
                    else:
                        new_lines = permute(sentences)
                        fout.write(new_lines)
                        fout.write('\n')
                        sentences = []
