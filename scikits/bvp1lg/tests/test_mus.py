# Author: Pauli Virtanen <pav@iki.fi>, 2006.
# All rights reserved. See LICENSE.txt.
"""
Tests for the MUS wrappers.
"""
from __future__ import division, absolute_import, print_function

from numpy.testing import *
import numpy as np

import scikits.bvp1lg.mus as mus

from testutils import *
import test_problems

def solve_with_mus(problem, **kw):

    if max(problem.m) > 1:
        problem = test_problems.FirstOrderConverter(problem)

    def gsub(ya, yb):
        fg = problem.g(ya, yb)
        dga, dgb = problem.dg(ya, yb)
        return fg, dga, dgb

    def guess(x):
        z, dm = problem.guess(x)
        return z

    if not problem.linear:
        x, y = mus.solve_nonlinear(problem.f, gsub, guess,
                                       problem.a, problem.b,
                                       **kw)
    else:
        def f_get_homogenous(x, u):
            return problem.f(x, u) - problem.f(x, 0*u)

        if hasattr(problem, 'f_homog'):
            f_homogenous = problem.f_homog
        else:
            f_homogenous = f_get_homogenous

        if problem.homogenous:
            f_nonhomogenous = None
        else:
            f_nonhomogenous = problem.f
        
        u0 = np.zeros([sum(problem.m)])
        m_a, m_b = problem.dg(u0, u0)
        bcv = -problem.g(u0, u0)
        
        x, y = mus.solve_linear(f_homogenous, f_nonhomogenous,
                                    problem.a, problem.b,
                                    m_a, m_b, bcv,
                                    **kw)
    return x, y

class test_mus(object):
    def test_problem_1(self):
        # Solve problem #1 and compare to exact solution
        problem = test_problems.Problem1()
        x, y = solve_with_mus(problem, output_points=51, rtol=1e-3, atol=1e-6)
        assert np.allclose(problem.exact_solution(x), y[:,0],
                          rtol=1e-5)

    def test_problem_2(self):
        # Solve problem #2 and compare to exact solution
        problem = test_problems.Problem2()
        x, y = solve_with_mus(problem, output_points=51, rtol=1e-3, atol=1e-6)
        assert np.allclose(problem.exact_solution(x), y,
                          rtol=1e-3, atol=1e-6)

    def test_problem_3(self):
        # Solve problem #3 and compare to exact solution
        problem = test_problems.Problem3()
        x, y = solve_with_mus(problem, output_points=51, rtol=1e-3, atol=1e-6)
        assert np.allclose(problem.exact_solution(x), y[:,0],
                          rtol=1e-5)

    def test_problem_4(self):
        # Solve problem #4 and compare to exact solution
        problem = test_problems.Problem4()
        x, y = solve_with_mus(problem, output_points=51, rtol=1e-3, atol=1e-6)
        assert np.allclose(problem.exact_solution(x), y,
                          rtol=1e-5)

    def test_problem_5(self):
        # Solve problem #5 and compare to exact solution
        problem = test_problems.Problem5()
        x, y = solve_with_mus(problem, output_points=51, rtol=1e-3, atol=1e-6)
        assert np.allclose(problem.exact_solution(x), y,
                          rtol=1e-2)
        # Needed to reduce tolerance, since "exact" solution data is
        # interpolated

    def test_problem_6(self):
        # Solve problem #6 and compare to exact solution
        problem = test_problems.Problem6()
        x, y = solve_with_mus(problem, output_points=51, rtol=1e-3, atol=1e-6)
        assert np.allclose(problem.exact_solution(x), y,
                          rtol=1e-5)

def test_doctests():
    assert doctest.testmod(mus, verbose=0)[0] == 0
