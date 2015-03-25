# Author: Pauli Virtanen <pav@iki.fi>, 2006.
# All rights reserved. See LICENSE.txt.
"""
Tests examples
"""
from __future__ import division, absolute_import, print_function

from numpy.testing import *

import scikits.bvp1lg.examples as examples

from testutils import *

def test_doctests():
    import matplotlib
    matplotlib.use('Agg')
    assert doctest.testmod(examples, verbose=0, optionflags=doctest.ELLIPSIS)[0] == 0
