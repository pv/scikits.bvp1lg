#!/usr/bin/env python

# Author: Pauli Virtanen <pav@iki.fi>, 2006.
# All rights reserved. See LICENSE.txt for the BSD-style license.

from numpy.distutils.misc_util import Configuration

from numpy.distutils.system_info import get_info,dict_append,\
     AtlasNotFoundError,LapackNotFoundError,BlasNotFoundError,\
     LapackSrcNotFoundError,BlasSrcNotFoundError

import sys

def configuration(parent_package='', top_path=None):
    atlas_info = get_info('atlas')
    blas_libs = []
    if not atlas_info:
        warnings.warn(AtlasNotFoundError.__doc__)
        blas_info = get_info('blas')
        if blas_info:
            blas_libs.extend(blas_info['libraries'])
        else:
            warnings.warn(BlasNotFoundError.__doc__)
    else:
        blas_libs.extend(atlas_info['libraries'])

    sys.path.append('bvp')
    info = __import__('info', {}, {}, [])
    sys.path.pop()

    config = Configuration('bvp', parent_package, top_path,
                           package_path='bvp',
                           version=info.__version__,
                           author=info.__author__,
                           author_email=info.EMAIL,
                           url=info.URL,
                           license="BSD",
                           description=info.DESCRIPTION,
                           )

    config.add_extension('_colnew',
                         sources=['lib/colnew.pyf',
                                  'lib/colnew.f',
                                  'lib/dgesl.f',
                                  'lib/dgefa.f'],
                         libraries=blas_libs,
                         library_dirs=atlas_info['library_dirs'])

    config.add_extension('_mus',
                         sources=['lib/mus.pyf',
                                  'lib/mus1.f',
                                  'lib/mus2.f',
                                  'lib/mus3.f',
                                  'lib/dgesl.f',
                                  'lib/dgefa.f'],
                         libraries=blas_libs,
                         library_dirs=atlas_info['library_dirs'])

    config.add_data_files('LICENSE.txt', 'TODO')
    config.add_subpackage('tests', 'bvp/tests')
    
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
