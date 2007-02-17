# Author: Pauli Virtanen <pav@iki.fi>, 2006.
# All rights reserved. See LICENSE.txt for the BSD-style license.
from info import __doc__, __version__, __author__, __date__

from error import *
import colnew
import mus
import jacobian
import examples

__all__ = filter(lambda s: not s.startswith('_'), dir())
from numpy.testing import ScipyTest as _ScipyTest
test = _ScipyTest().test
