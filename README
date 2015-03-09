==============
scikits.bvp1lg
==============

Solve boundary value problems for ODEs, using legacy solvers.

Contents
========

``colnew``
    Solver for both linear and non-linear multi-point boundary-value
    problems, with separated boundary conditions. Uses a collocation
    method: the COLNEW solver.

``mus``
    Solves both linear and non-linear two-point boundary-value problems,
    also with unseparated boundary conditions. Uses a multiple shooting
    method: the MUS solver.

``jacobian``
    Utility routines, for checking functions that calculate Jacobians,
    or just calculating them.

``examples``

    Examples (in docstrings).

Installation
============

The usual ``python setup.py install`` instructions apply.  You need to have
Numpy and a supported Fortran compiler installed.  You also need Scipy if you
want to run the test suite, or use the ``mus`` solver.

To run tests, you also need the Nose testing framework. You can run the tests
with::

    nosetests -v --exe scikits.bvp1lg

All tests should pass without failures.
