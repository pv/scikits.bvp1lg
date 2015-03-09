# Author: Pauli Virtanen <pav@iki.fi>, 2006.
# All rights reserved. See LICENSE.txt.

from __future__ import division, absolute_import, print_function

from numpy.testing import *
import doctest
import numpy as np

__all__ = ['curves_close', 'doctest']


def curves_close(x, y1, xtol1, ytol1,
                    y2, xtol2, ytol2):
    """Returns true if curves coincide up to given absolute tolerances.

    :Parameters:
      - `x`: x-coordinates
      - `y1`: points of curve 1 at `x`
      - `y2`: points of curve 2 at `x`
      - `xtol1`: absolute x-tolerance for curve 1
      - `xtol2`: absolute x-tolerance for curve 2
      - `ytol1`: absolute y-tolerance for curve 1
      - `ytol2`: absolute y-tolerance for curve 2
    """
    
    def ddiff(x):
        dx = np.diff(x)
        return np.r_[dx, dx[-1]]

    dx = ddiff(x)
    dy1 = np.absolute(ddiff(y1))
    dy2 = np.absolute(ddiff(y2))

    tol = ytol1 + ytol2 + xtol1*dy1/dx + xtol2*dy2/dx
    d = np.less(np.absolute(y1 - y2), tol)
    return d.ravel().all()
