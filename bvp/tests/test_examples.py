# Author: Pauli Virtanen <pav@iki.fi>, 2006.
# All rights reserved. See LICENSE.txt for the BSD-style license.
"""
Tests examples
"""
from numpy.testing import *
import sys, os

set_package_path()
import bvp.examples
restore_path()

sys.path.append(os.path.dirname(__file__))
from testutils import *
import test_problems

test_doc = get_doctest_checker([bvp.examples])

###############################################################################

if __name__ == "__main__":
    ScipyTest().run()
