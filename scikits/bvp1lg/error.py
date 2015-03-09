# Author: Pauli Virtanen <pav@iki.fi>, 2006.
# All rights reserved. See LICENSE.txt.
"""
Exceptions for the bvp package
"""
from __future__ import absolute_import, division, print_function


class NoConvergence(RuntimeError):
    """No numerical convergence obtained"""
    pass

class SingularCollocationMatrix(NoConvergence):
    """The collocation matrix was singular (double-check your equations or try a different initial guess)"""
    pass

class TooManySubintervals(NoConvergence):
    """COLNEW couldn't find a solution within the given storage limits"""
    pass

class SingularityError(NoConvergence):
    """A solution element became singular"""
    pass
