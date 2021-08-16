# -*- coding: utf-8 -*-

from tools.utils import Config

def converter(**kwargs):
    args = Config(**locals())

    # Read mapping file
    maps = {}
    with open(args.mapping_file) as file_mapping:
        for line in file_mapping:
            old, new = line.split('\t')[0].strip(), line.split('\t')[1].strip()
            maps[old] = new

    # Write new PoS tagset treebank to file
    with open(args.output_file, 'w') as outf:
        with open(args.input_file) as inf:
            for inl in inf:
                if not inl.strip().startswith("#"):
                    if len(inl.strip()) > 0:
                        olds = inl.strip().split('\t')

                        olds[4] = maps[olds[4]] # replace old pos by new pos
                        olds[3] = olds[4] # replace gold pos by new pos
                        news = '\t'.join(olds) # merge to new line

                        outf.write(news) # write to file
                        outf.write('\n')
                    else:
                        outf.write('\n')
