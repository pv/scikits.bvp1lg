# Author: Pauli Virtanen <pav@iki.fi>, 2006.
# All rights reserved. See LICENSE.txt.
"""
Tests for the partial derivative routines
"""
from __future__ import division, absolute_import, print_function

from numpy.testing import *
import numpy as np

import scikits.bvp1lg.jacobian as jacobian
import scikits.bvp1lg.colnew as colnew

from testutils import *
import test_problems

class TestJacobian(object):
    def test_problems(self):
        for problem in [test_problems.Problem1(),
                        test_problems.Problem2(),
                        test_problems.Problem3(),
                        test_problems.Problem4(),
                        test_problems.Problem5(),
                        test_problems.Problem6(),
                        test_problems.Problem7(),
                        test_problems.Problem8()]:

            if problem.vectorized:
                assert jacobian.check_jacobian(
                    sum(problem.m),
                    lambda u: np.squeeze(problem.f(np.array([0.5]), u)),
                    lambda u: np.squeeze(problem.df(np.array([0.5]), u)))
            else:
                assert jacobian.check_jacobian(
                    sum(problem.m),
                    lambda u: np.squeeze(problem.f(0.5, u)),
                    lambda u: problem.df(0.5, u))
            assert jacobian.check_jacobian(
                sum(problem.m),
                lambda u: np.squeeze(problem.g(u, 0*u)),
                lambda u: problem.dg(u, 0*u)[0])
            assert jacobian.check_jacobian(
                sum(problem.m),
                lambda u: np.squeeze(problem.g(0*u, u)),
                lambda u: problem.dg(0*u, u)[1])

    def test_solving(self):
        # Solve problem #3 with numerical partial derivatives
        problem = test_problems.Problem3()

        def dfsub(x, z):
            df = np.empty([1, 2, len(x)])
            for k in range(z.shape[1]):
                df[:,:,k] = jacobian.jacobian(
                    lambda u: problem.f(x, u), z[:,k], eps=1e-6)
            return df
        def gsub(z):
            return problem.g(z[:,0], z[:,1])
        def dgsub(z):
            za = z[:,0]
            zb = z[:,1]
            xa = jacobian.jacobian(lambda u: problem.g(u, za), za,eps=1e-6)
            xb = jacobian.jacobian(lambda u: problem.g(zb, u), zb,eps=1e-6)
            return np.r_[xa[None,0,:], xb[None,1,:]]

        solution = colnew.solve(
            [problem.a, problem.b],
            problem.m, problem.f, gsub,
            dfsub=dfsub, dgsub=dgsub,
            initial_guess=problem.guess,
            tolerances=[1e-5, 1e-5])

        x = np.linspace(problem.a, problem.b, 100)
        assert np.allclose(problem.exact_solution(x), solution(x)[:,0],
                          rtol=1e-5)

    def test_solving_2(self):
        # Solve problem #3 with numerical partial derivatives
        problem = test_problems.Problem3()

        def gsub(z):
            return problem.g(z[:,0], z[:,1])

        solution = colnew.solve(
            [problem.a, problem.b],
            problem.m, problem.f, gsub,
            initial_guess=problem.guess,
            tolerances=[1e-5, 1e-5])

        x = np.linspace(problem.a, problem.b, 100)
        assert np.allclose(problem.exact_solution(x), solution(x)[:,0],
                          rtol=1e-5)
        
def test_doctest():
    assert doctest.testmod(jacobian, verbose=0)[0] == 0
