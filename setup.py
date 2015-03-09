#!/usr/bin/env python

# Author: Pauli Virtanen <pav@iki.fi>, 2006.
# All rights reserved. See LICENSE.txt.

import os
import sys

import os
import sys

DISTNAME            = 'scikits.bvp1lg'
DESCRIPTION         = 'Boundary value problem (legacy) solvers for ODEs'
LONG_DESCRIPTION = """\
Python-wrapped legacy solvers for boundary value problems for ODEs.

These are implemented by wrapping the COLNEW and MUS Fortran
codes from netlib.org.
"""
MAINTAINER          = 'Pauli Virtanen',
MAINTAINER_EMAIL    = 'pav@iki.fi',
URL                 = 'http://www.iki.fi/pav/software/bvp'
LICENSE             = 'Noncommercial, see LICENSE.txt'
DOWNLOAD_URL        = URL
VERSION             = '0.2.5'

def configuration(parent_package='', top_path=None, package_name=DISTNAME):
    if os.path.exists('MANIFEST'): os.remove('MANIFEST')
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path,
                           namespace_packages=['scikits'],
                           version=VERSION,
                           maintainer =MAINTAINER,
                           maintainer_email=MAINTAINER_EMAIL,
                           description=DESCRIPTION,
                           license=LICENSE,
                           url=URL,
                           download_url=DOWNLOAD_URL,
                           long_description=LONG_DESCRIPTION)

    config.set_options(
            ignore_setup_xxx_py=True,
            assume_default_configuration=True,
            delegate_options_to_subpackages=True,
            quiet=True,
            )

    config.add_subpackage('scikits')
    config.add_data_files('scikits/__init__.py')
    config.add_subpackage(DISTNAME)

    f = open(os.path.join('scikits', 'bvp1lg', 'version.py'), 'w')
    f.write("__version__ = \"%s\"\n" % VERSION)
    f.close()

    return config

if __name__ == "__main__":
    try: import setuptools
    except ImportError: pass

    from numpy.distutils.core import setup
    setup(configuration=configuration,
          name=DISTNAME,
          install_requires=["numpy >= 1.0", "scipy >= 0.5"],
          namespace_packages=['scikits'],
          packages=setuptools.find_packages(),
          include_package_data=True,
          zip_safe=False,
          test_suite="nose.collector",
          tests_require = ['nose >= 0.10.3'],
          classifiers=
          ['Development Status :: 4 - Beta',
           'Environment :: Console',
           'Intended Audience :: Developers',
           'Intended Audience :: Science/Research',
           'License :: Free for non-commercial use',
           'Topic :: Scientific/Engineering'])
