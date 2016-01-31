# coding: utf-8
"""
created by artemkorkhov at 2016/01/16
"""

import argparse
import requests
from esconf.index_settings import *


def make_index(name, idx_setting="DEFAULT_INDEX"):
    """
    :param idx_setting:
    :return:
    """
    url = "http://localhost:9200/{}".format(name)
    return requests.post(url, data=globals()[idx_setting])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', default='wikitest')
    parser.add_argument('-s', '--settings', default="DEFAULT_INDEX")
    args = parser.parse_args()

    make_index(args.name, args.settings)