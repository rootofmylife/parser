# -*- coding: utf-8 -*-

from tools.utils import Config
import glob

def merge(**kwargs):
    args = Config(**locals())

    with open('output.conll', 'wb') as outfile:
        for filename in glob.glob(args.folder + '*.*'):
            with open(filename, 'rb') as readfile:
                outfile.write(readfile.read())
