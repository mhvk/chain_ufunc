#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('chain_ufunc',
                           parent_package,
                           top_path)
    config.add_extension('ufunc_chain', ['chain_ufunc/ufunc_chain.c'])

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(configuration=configuration)
