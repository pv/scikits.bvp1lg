# Author: Pauli Virtanen <pav@iki.fi>, 2006.
# All rights reserved. See LICENSE.txt.
"""
Tests examples
"""
from numpy.testing import *

import scikits.bvp1lg.examples as examples

from testutils import *

class test_doc(TestCase):
    def test_all(self):
        assert doctest.testmod(examples, verbose=0)[0] == 0

###############################################################################

if __name__ == '__main__':
    import unittest
    unittest.main()
