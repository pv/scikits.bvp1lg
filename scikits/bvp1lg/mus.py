# Author: Pauli Virtanen <pav@iki.fi>, 2006.
# All rights reserved. See LICENSE.txt.
"""
mus
===

Solve two-point boundary value problems for ODEs

- `solve_linear`: Solve linear problems
- `solve_nonlinear`: Solve non-linear problems

Description
-----------

This module uses the multiple shooting code MUS [MS]_ for solving
boundary-value problems.

.. warning::

    For some reason, this code can be slow for certain problems.
    This may be due to that
    
        1. This wrapper uses MUS somehow incorrectly
        2. Speed of MUS is sensitive to tweaking parameters
        3. Python call overhead is significant for MUS as it is
           not vectorized over mesh points
    
    I have not yet figured out what the problem is, so you may be
    better off using the `colnew` package.

References
----------

.. [MS] R. M. M. Mattheij, G. W. M. Staarink.
        EUT Report 92-WSK-01 (1992).

Module contents
---------------
"""
from __future__ import absolute_import, division, print_function

import numpy as np
from . import _mus
import warnings as _warnings

from .error import *

###############################################################################

_musl_errors = {
    100: ValueError('either N < 1 or IHOM < 0 or NRTI < 0 or NTI < 5 or '
                    'NU < N * (N+1) / 2 or A=B'),
    101: ValueError('either ER(1) or ER(2) or ER(3) is negative'),
    103: ValueError('either LW < 8*N + 2*N*N or LIW < 3*N'),
    120: ValueError('the routine was called with NRTI = 1, but the given '
                    'output points in the array TI are not in strict '
                    'monotone order.'),
    121: ValueError('the routine was called with NRTI = 1, but the first '
                    'given output point or the last output point is not '
                    'equal to A or B.'),
    122: ValueError('the value of NTI is too small; the number of output '
                    'points is greater than NTI-3.'),
    215: NoConvergence('during integration the particular solution or a '
                       'homogeneous solution has vanished, making a pure '
                       'relative error test impossible. Must use non-zero '
                       'absolute tolerance to continue.'),
    216: NoConvergence('during integration the requested accuracy could not '
                       'be achieved. User must increase error tolerance.'),
    218: ValueError('the input parameter N <= 0, or that either the '
                    'either the relative tolerance or the absolute tolerance '
                    'is negative'),
    240: NoConvergence('global error is probably larger than the error '
                       'tolerance due to instabilities in the system. Most '
                       'likely the problem is ill-conditioned. Output value '
                       'is the estimated error amplification factor ER(5)'),
    250: SingularityError('one of the U(k) is singular'),
    260: NoConvergence('the problem is probably too ill-conditioned with '
                       'respect to the BC'),
}
_musl_warnings = {
    200: ('there is a minor shooting interval on which the '
          'incremental growth is greater than the AMP. This is to be '
          'attributed to the used method for computing the fundamental '
          'solution, and may jeopardize the global accuracy if '
          'ER(3) * AMP > max(ER(1),ER(2)).'),
    213: ('the relative tolerance was too small. The subroutine '
          'has changed it into a suitable value.'),
}

_musn_errors = dict(_musl_errors)
_musn_errors.update({
    105: ValueError('either N < 1 or NRTI < 0 or NTI < 3 or NU < N*(N+1)/2 or '
                    'A=B'),
    106: ValueError('either LW < 7*N + 3*N*NTI + 4*N*N  or LIW < 3*N + NTI'),
    122: ValueError('the value of NTI is too small; the number of necessary '
                    'output points is greater than NTI-1.'),
    123: ValueError('the value of LWG is less than the number of output '
                    'points. Increase the dimension of the array WG and '
                    'the value of LWG.'),
    219: TooManySubintervals('the routine needs more space to store the '
                             'integration grid point. An estimate for the '
                             'required workspace (i.e. the value for LWG) '
                             'is given.'),
    230: NoConvergence('Newton failed to converge'),
    231: NoConvergence('number of iterations has become greater than ITLIM'),
})

def __check_errors(ierror, errors, warnings, unknown_msg):
    if ierror != 0:
        err = errors.get(ierror, None)
        if err == None:
            err = warnings.get(ierror, None)
            if err:
                _warnings.warn(str(err))
                err = False

        if err == False:
            pass # no error after all
        elif err != None:
            raise err
        else:
            raise RuntimeError(unknown_msg)

###############################################################################

def __get_output_points(output_points):
    nrti = 0
    try:
        nrti = int(output_points)
        output_points = None
    except (TypeError, ValueError):
        pass
    
    if output_points == None:
        if nrti <= 1:
            ti = np.zeros([200], np.float64)
            nrti = 0
        else:
            ti = np.zeros([nrti + 4], np.float64)
    else:
        nrti = len(output_points)
        ti = np.zeros([nrti + 4], np.float64)
        ti[:nrti] = output_points
    return nrti, ti

def __get_tolerance(rtol, atol):
    if rtol == None:
        rtol = np.finfo(np.float64).eps * 10
    if atol == None:
        atol = 0

    er = np.zeros([5], np.float64)
    er[0] = rtol
    er[1] = atol
    er[2] = np.finfo(np.float64).eps
    
    return er

###############################################################################

def solve_linear(f_homogenous, f_nonhomogenous, a, b, m_a, m_b, bcv,
                 max_amplification=None, rtol=None, atol=None,
                 output_points=None, verbosity=0):
    """Solve a linear two-point boundary value problem.

    The problem is assumed to be::

        u'(t) = L(t) u(t) + r(t),        a <= t <= b
        M_A u(a) + M_B u(b) = BCV

    where u is the solution vector.

    :Parameters:
    
      - `f_homogenous`:
        Function ``f_h(t, u)`` that evaluates the homogenous part ``L(t) u``.
        It should return an array of shape (n,).
        
      - `f_nonhomogenous`:
        Function ``f(t, u)`` that evaluates ``L(t) u + r(t)``.
        For homogenous problems, you can set f_nonhomogenous to None.
        It should return an array of shape (n,).
        
      - `a`:
        Left boundary point.
        
      - `b`:
        Right boundary point.
        
      - `m_a`:
        The (n, n) M_A matrix in the boundary condition.
        
      - `m_b`:
        The (n, n) M_B matrix in the boundary condition.
        
      - `bcv`:
        The (n,) BCV vector in the boundary condition.
        
      - `max_amplification`:
        The allowed incremental factor of the homogeneous solutions between
        two succesive output points. If the increment of a homogeneous
        solution between two succesive output points becomes greater
        than 2*AMP, a new output point is inserted.
        Set to None to use a sensible default.
        
      - `rtol`:
        A relative tolerance for solving the differential equation.
        
      - `atol`:
        An absolute tolerance for solving the differential equation.
        
      - `output_points`:
        If integer: desired number of output points, based on ``amplification``
        If array: desired output points
        If None: let the algorithm decide the output points
        
      - `verbosity`:
        0 silent, 1 some output, 2 more output

    :returns:
        A tuple ``(t, y)`` where ``t`` is a (m,) array of mesh points, and
        ``y`` is (m, n) array of solution values at the mesh points.

    :raise ValueError: Invalid input
    :raise NoConvergence: Numerical convergence problems
    :raise SingularityError: Infinities occurred
    :raise SystemError:
        Invalid output from user routines. (FIXME: these should be fixed)
    """

    ## Homogenity

    ihom = int(f_nonhomogenous != None)
    if f_nonhomogenous == None:
        f_nonhomogenous = f_homogenous

    ## Output points

    nrti, ti = __get_output_points(output_points)

    ## Tolerance

    er = __get_tolerance(rtol, atol)

    ## Call

    if max_amplification == None:
        max_amplification = max(er[0], er[1]) / er[2]

    ierror = verbosity - 1
    if ierror < -1: ierror = -1
    if ierror >  1: ierror =  1

    er, nrti, ti, y, ierror = _mus.musl(f_homogenous, f_nonhomogenous,
                                        ihom, a, b, m_a, m_b, bcv,
                                        er, nrti, ti, ierror,
                                        amp=max_amplification)

    __check_errors(ierror, _musl_errors, _musl_warnings,
                   "Unknown error from MUSL")

    ## Finish

    return ti[:nrti].copy(), np.transpose(y[:,:nrti]).copy()

###############################################################################

def solve_nonlinear(func, gsub, initial_guess, a, b,
                    max_amplification=0, rtol=1e-5, atol=None,
                    output_points=None, verbosity=0,
                    iteration_limit=100):
    """Solve a non-linear two-point boundary value problem.

    The problem is assumed to be::

        u'(t) = f(t, u(t)),                   a <= t <= b
        g_j(u(a), u(b)) = 0                   j = 0, 1, ..., n

    where u is the solution vector with shape (n,).

    :Parameters:
    
      - `func`:
        Function ``f = f(t, u)`` that evaluates derivatives.
        It should return an array of shape (n,).
        
      - `gsub`:
        Function ``g, dga, dgb = g(u_a, u_b)`` that evaluates the boundary
        condition function and its Jacobian. The output should be::
        
            g[i] = g_i(u_a, u_b),                          i = 0, ..., n-1
            dga[i,j] = d g_i(u_a, u_b) / d u_a[j],         j = 0, ..., n-1
            dgb[i,j] = d g_i(u_a, u_b) / d u_b[j],

      - `initial_guess`:
        Either callable ``u = initial_guess(t)`` that provides an initial
        guess for the solution vector, or a tuple ``(t, u)`` providing
        mesh and values for the guess --- these are interpolated
        linearly to form the guess at all points.
        (Previous output from ``musn`` can be used as ``initial_guess``.)
        
      - `a`:
        Left boundary point.
      
      - `b`:
        Right boundary point.
      
      - `max_amplification`:
        The allowed incremental factor of the homogeneous solutions between
        two succesive output points. If the increment of a homogeneous
        solution between two succesive output points becomes greater
        than 2*AMP, a new output point is inserted.
        Set to None to use a sensible default.
        
      - `rtol`:
        A relative tolerance for solving the differential equation.
      
      - `atol`:
        An absolute tolerance for solving the differential equation.
      
      - `output_points`:
        If integer: desired number of output points, based on ``amplification``
        If array: desired output points
        If None: let the algorithm decide the output points
        
      - `verbosity`:
        0 silent, 1 some output, 2 more output
      
      - `iteration_limit`:
        Maximum allowed number of Newton iterations
      
    :returns:
        A tuple ``(t, u)`` where ``t`` is a (m,) array of mesh points, and
        ``u`` is (m, n) array of solution values at the mesh points.

    :raise ValueError:
        Invalid input
    :raise NoConvergence:
        Numerical convergence problems
    :raise SingularityError:
        Infinities occurred
    :raise SystemError:
        Invalid output from user routines. (FIXME: these should be fixed)

    """
    ## Postponed input -- soft dependency on Scipy only
    import scipy.interpolate as _interpolate

    ## Determine size

    f0 = func(a, initial_guess(a))
    n = len(f0)

    ## Output points

    nrti, ti = __get_output_points(output_points)

    ## Tolerance

    er = __get_tolerance(rtol, atol)

    ## Estimate workspace

    # FIXME: this value for ``lwg`` is usually enough, but how to determine
    #        the needed storage space reliably?
    lwg = 10 * (len(ti) + 20)

    ## Initial guess

    if not callable(initial_guess) and isinstance(initial_guess, tuple):
        x, y = initial_guess
        interp = _interpolate.interp1d(x, y)
        def simple_guess(x):
            return interp(x)
        initial_guess = simple_guess

    ## Call

    if max_amplification == None:
        max_amplification = max(er[0], er[1]) / er[2]

    ierror = verbosity - 1
    if ierror < -1: ierror = -1
    if ierror >  1: ierror =  1

    er, ti, nrti, y, ierror = _mus.musn(func, initial_guess, gsub,
                                        n, a, b,
                                        er, ti, nrti, iteration_limit, lwg,
                                        ierror, amp=max_amplification)

    __check_errors(ierror, _musn_errors, {},
                   "Unknown error from MUSN")

    ## Finish

    return ti[:nrti].copy(), np.transpose(y[:,:nrti]).copy()
