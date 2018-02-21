#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import logging.config
import warnings

import resq.config as config

log_config = {
    'version': 1,
    'formatters': {
        'colored': {
            '()': 'colorlog.ColoredFormatter',
            'format':
            '%(asctime)s.%(msecs)03d %(log_color)s %(levelname)-8s [%(name)-10s] %(message)s',
            'datefmt': '%m-%d %H:%M:%S',
            'reset': True,
            'log_colors': {
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red',
            }
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'colored',
            'level': 'DEBUG',
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'colored',
            'filename': config.log_file,
            'level': 'DEBUG'
        },
    },
    'loggers': {
        'asyncio': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG'
        },
        'melvingen': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG'
        },
        'netqos': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG'
        },
    }
}

logging.config.dictConfig(log_config)
try:
    import numpy as np
    warnings.simplefilter('ignore', np.RankWarning)
except:
    pass


def logger():
    #import inspect
    #curframe = inspect.currentframe()
    #calframe = inspect.getouterframes(curframe, 2)
    #caller_name = inspect.getmodule(calframe).__name__
    return logging.getLogger('netqos')


log_debug = logger().debug
log_error = logger().error
log_info = logger().info
log_warn = logger().warn
