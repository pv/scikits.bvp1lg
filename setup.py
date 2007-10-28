#!/usr/bin/env python

# Author: Pauli Virtanen <pav@iki.fi>, 2006.
# All rights reserved. See LICENSE.txt.

try: import setuptools
except ImportError: pass

import warnings
from numpy.distutils.misc_util import Configuration
from numpy.distutils.system_info import (get_info, AtlasNotFoundError,
                                         BlasNotFoundError)

def configuration(parent_package='', top_path=None):
    blas_info = get_info('atlas')
    if not blas_info:
        warnings.warn(AtlasNotFoundError.__doc__)
        blas_info = get_info('blas')
        if not blas_info:
            # Blas is required
            print "\nError:\n%s\n" % BlasNotFoundError.__doc__
            raise SystemExit(1)
    
    info = __import__('bvp/info')

    config = Configuration('bvp', parent_package, top_path,
                           package_path='bvp',
                           version=info.__version__,
                           author=info.__author__,
                           author_email=info.EMAIL,
                           url=info.URL,
                           license=info.LICENSE,
                           description=info.DESCRIPTION,
                           long_description=info.LONG_DESCRIPTION,
                           install_requires=["numpy >= 1.0", "scipy >= 0.5"],
                           )

    config.add_extension('_colnew',
                         sources=['lib/colnew.pyf',
                                  'lib/colnew.f',
                                  'lib/dgesl.f',
                                  'lib/dgefa.f'],
                         libraries=blas_info['libraries'],
                         library_dirs=blas_info['library_dirs'])

    config.add_extension('_mus',
                         sources=['lib/mus.pyf',
                                  'lib/mus1.f',
                                  'lib/mus2.f',
                                  'lib/mus3.f',
                                  'lib/dgesl.f',
                                  'lib/dgefa.f'],
                         libraries=blas_info['libraries'],
                         library_dirs=blas_info['library_dirs'])

    config.add_data_files('LICENSE.txt', 'TODO')
    config.add_subpackage('tests', 'bvp/tests')
    
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
