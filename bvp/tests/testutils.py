# Author: Pauli Virtanen <pav@iki.fi>, 2006.
# All rights reserved. See LICENSE.txt for the BSD-style license.

from numpy.testing import *
import doctest
import scipy as N

def _get_doctests(m):
    """Get doctests from module(s) m"""
    if hasattr(doctest, 'DocTestFinder'):
        if not hasattr(m, '__len__'): m = [m]
        tests = []
        for x in m:
            tests += doctest.DocTestFinder().find(x)
        return tests
    else:
        return m

class _DocTestChecker(NumpyTestCase):
    def __init__(self, *args, **kw):
        NumpyTestCase.__init__(self, *args, **kw)
        
        if hasattr(doctest, 'DocTestFinder'):
            self.runner = doctest.DocTestRunner(verbose=False)
        else:
            self.runner = None
            
    def _run_test(self, test):
        if self.runner:
            s = []
            self.runner.run(test, out=lambda x: s.append(x))
            if s: raise AssertionError('\n' + ''.join(s))
        else:
            doctest.testmod(test, verbose=False)

def _get_test_func(test):
    func = lambda self: self._run_test(test)
    if hasattr(test, 'name'):
        func.__doc__ = "Doctests for %s" % test.name
    return func

def get_doctest_checker(m, clsname='test_doc'):
    methods = {}
    for test in _get_doctests(m):
        if hasattr(test, 'name'):
            name = str(test.name).replace('.', '_')
            methods['check_doc_' + name] = _get_test_func(test)
        else:
            methods['check_doc'] = _get_test_func(test)
    return type(clsname, (_DocTestChecker,), methods)

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
        dx = N.diff(x)
        return N.r_[dx, dx[-1]]

    dx = ddiff(x)
    dy1 = N.absolute(ddiff(y1))
    dy2 = N.absolute(ddiff(y2))

    tol = ytol1 + ytol2 + xtol1*dy1/dx + xtol2*dy2/dx
    d = N.less(N.absolute(y1 - y2), tol)
    return d.ravel().all()
