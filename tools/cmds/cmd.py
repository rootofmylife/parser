# -*- coding: utf-8 -*-

import logging
from tools.utils import Config

from tools.processor.convert import converter
from tools.processor.statistics import stats
from tools.processor.concat import merge
from tools.processor.reorder import wordorder

logger = logging.getLogger('tools')

def parse(parser):
    args, unknown = parser.parse_known_args()
    args, unknown = parser.parse_known_args(unknown, args)
    args = Config.load(**vars(args), unknown=unknown)

    logger.info(str(args))

    if args.mode == 'convert':
        converter(**args)
    elif args.mode == 'statistics':
        stats(**args)
    elif args.mode == 'concat':
        merge(**args)
    elif args.mode == 'reorder':
        wordorder(**args)
