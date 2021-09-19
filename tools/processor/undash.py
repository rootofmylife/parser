# -*- coding: utf-8 -*-

from tools.utils import Config

def permute(lines):
    pre_lines = []
    for line in lines:
        l = line.split('\t')
        l[1] = l[1].replace('_', ' ')
        if len(l[1].strip()) == 0:
            print(line)
        nl = '\t'.join(l)
        pre_lines.append(nl)

    lines = '\n'.join(pre_lines)
    return lines + '\n'

def undash(**kwargs):
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
