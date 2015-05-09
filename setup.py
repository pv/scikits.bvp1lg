#!/usr/bin/env python

# Author: Pauli Virtanen <pav@iki.fi>, 2006.
# All rights reserved. See LICENSE.txt.

import os
import sys
import shutil
from hashlib import sha256
from distutils.dep_util import newer

try:
    from urllib2 import urlopen
except ImportError:
    from urllib.request import urlopen

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))
import patchit

DISTNAME            = 'scikits.bvp1lg'
DESCRIPTION         = 'Boundary value problem (legacy) solvers for ODEs'
LONG_DESCRIPTION = """\
Python-wrapped legacy solvers for boundary value problems for ODEs.

These are implemented by wrapping the COLNEW and MUS Fortran
codes from netlib.org.
"""
MAINTAINER          = 'Pauli Virtanen',
MAINTAINER_EMAIL    = 'pav@iki.fi',
URL                 = 'https://pv.github.io/scikits.bvp1lg'
LICENSE             = 'Noncommercial, see LICENSE.txt'
DOWNLOAD_URL        = 'https://pypi.python.org/pypi/scikits.bvp1lg'
VERSION             = '0.2.8'


DOWNLOADABLE_FILES = [
    ('http://netlib.org/ode/colnew.f', 'lib/colnew.f.patch', 'lib/colnew.f',
     'e868b78f41c60bc38ac0f56981730c0dba8e592097f24af2537452f3d283e79e'),
    ('http://netlib.org/ode/mus1.f', 'lib/mus1.f.patch', 'lib/mus1.f',
     '3e159aa446dc1aa06695999d8b6deea5b5ba41a347fdf7aa6a33849d5de5f3a6'),
    ('http://netlib.org/ode/mus2.f', None, 'lib/mus2.f',
     'a17132108507d9c9b62fce8225fc3f134334ef1ce2abe61281307cf7313bc3c7'),
    ('http://netlib.org/ode/mus3.f', 'lib/mus3.f.patch', 'lib/mus3.f',
     '3aa1e25e79b8af2123036092bb4d5f1b957c60401987ba00ddcd921b45d0c5fe'),
]


def sha256sum(fn):
    with open(fn, 'rb') as f:
        return sha256(f.read()).hexdigest()


def download_and_patch(url, patch, dest, checksum):
    orig_dst = dest + '.orig'

    if (patch is None or not newer(patch, dest)) and os.path.isfile(dest) and os.path.isfile(orig_dst):
        return

    if not os.path.isfile(orig_dst):
        print("Downloading %s..." % (url,))
        with open(orig_dst, 'wb') as dst:
            src = urlopen(url)
            try:
                shutil.copyfileobj(src, dst)
            finally:
                src.close()

    downloaded_hash = sha256sum(orig_dst)
    if downloaded_hash != checksum:
        os.remove(orig_dst)
        raise RuntimeError("Downloaded file fails checksum!")

    if patch is None:
        shutil.copyfile(orig_dst, dest)
    else:
        print("Patching %s..." % (os.path.basename(dest),))

        with open(patch, 'r') as f:
            patch_set = patchit.PatchSet.from_stream(f)

        with open(orig_dst, 'r') as f:
            lines = [x.rstrip("\r\n") for x in f.readlines()]

        lines = list(patch_set[0].merge(lines))
        with open(dest, 'w') as f:
            f.writelines("\n".join(lines))


def configuration(parent_package='', top_path=None, package_name=DISTNAME):
    if os.path.exists('MANIFEST'): os.remove('MANIFEST')
    from numpy.distutils.misc_util import Configuration

    for url, patch, fn, checksum in DOWNLOADABLE_FILES:
        download_and_patch(url,
                           os.path.join(os.path.dirname(__file__), patch) if patch else None,
                           os.path.join(os.path.dirname(__file__), fn),
                           checksum)

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
          install_requires=["numpy >= 1.5", "scipy >= 0.9"],
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
