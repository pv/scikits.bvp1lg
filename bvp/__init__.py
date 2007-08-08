# Author: Pauli Virtanen <pav@iki.fi>, 2006.
# All rights reserved. See LICENSE.txt.
from info import __doc__, __version__, __author__, __date__

from error import *
import colnew
import mus
import jacobian
import examples

__all__ = filter(lambda s: not s.startswith('_'), dir())

def test(level=1, verbosity=1):
    from numpy.testing import ScipyTest as _ScipyTest
    return _ScipyTest().test(level, verbosity)
