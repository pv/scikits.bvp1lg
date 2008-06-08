#!/usr/bin/env python

# Author: Pauli Virtanen <pav@iki.fi>, 2006.
# All rights reserved. See LICENSE.txt.

try: import setuptools
except ImportError: pass

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.add_subpackage('bvp')
    config.add_data_files(('bvp', '*.txt'),
                          ('bvp', 'TODO'))
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(
        name='bvp',
        author="Pauli Virtanen <pav@iki.fi>",
        author_email="pav@iki.fi",
        version="0.2.4.dev",
        url="http://www.iki.fi/pav/software/bvp",
        license="Noncommercial, see LICENSE.txt",
        description="Boundary value problem solvers for ODEs",
        long_description="""\
Solvers for boundary value problems for ODEs.

These are implemented by wrapping the COLNEW and MUS Fortran
codes from netlib.org.""",
        install_requires=["numpy >= 1.0", "scipy >= 0.5"],
        configuration=configuration,
        )        
