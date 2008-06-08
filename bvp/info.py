# Author: Pauli Virtanen <pav@iki.fi>, 2006.
# All rights reserved. See LICENSE.txt.
"""
Solve boundary value problems for ODEs

Contents
========

- `colnew`:
  Solver for both linear and non-linear multi-point boundary-value
  problems, with separated boundary conditions. Uses a collocation
  method.

- `mus`:
  Solves both linear and non-linear two-point boundary-value problems,
  also with unseparated boundary conditions. Uses a multiple shooting
  method.

- `jacobian`:
  Utility routines, for checking functions that calculate Jacobians,
  or just calculating them.

- `examples`:
  Examples (in docstrings).
"""

__version__ = "0.2.4"
