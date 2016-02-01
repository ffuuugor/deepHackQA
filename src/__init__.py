# coding: utf-8
"""
created by artemkorkhov at 2016/01/16
"""

import os
import logging.config


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

logging_conf = {
    'version': 1,
    'formatters': {
        'simple': {
            'format': '%(asctime)s\t%(name)s\t%(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
            'level': 'DEBUG'
        }
    },
    'root': {
        'handlers': ['console'],
        'level': 'DEBUG'
    },
    'loggers': {
        '__main__': {
            'level':'DEBUG',
            # 'handlers':['console']
            'propagate': True
        },
    }
}


logging.config.dictConfig(logging_conf)
