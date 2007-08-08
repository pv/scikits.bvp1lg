# Author: Pauli Virtanen <pav@iki.fi>, 2006.
# All rights reserved. See LICENSE.txt.
"""
Tests examples
"""
from numpy.testing import *

set_package_path()
import bvp.examples
restore_path()

set_local_path()
from testutils import *
restore_path()

test_doc = get_doctest_checker([bvp.examples])

###############################################################################

if __name__ == "__main__":
    NumpyTest().run()
