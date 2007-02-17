# Author: Pauli Virtanen <pav@iki.fi>, 2006.
# All rights reserved. See LICENSE.txt for the BSD-style license.
"""
Tests for the partial derivative routines
"""

from numpy.testing import *
import scipy as N

set_package_path()
import bvp.jacobian, bvp.colnew
restore_path()

set_local_path()
from testutils import *
import test_problems
restore_path()

class test_jacobian(ScipyTestCase):
    def check_problems(self):
        for problem in [test_problems.Problem1(),
                        test_problems.Problem2(),
                        test_problems.Problem3(),
                        test_problems.Problem4(),
                        test_problems.Problem5(),
                        test_problems.Problem6(),
                        test_problems.Problem7(),
                        test_problems.Problem8()]:

            if problem.vectorized:
                assert bvp.jacobian.check_jacobian(
                    sum(problem.m),
                    lambda u: N.squeeze(problem.f(N.array([0.5]), u)),
                    lambda u: N.squeeze(problem.df(N.array([0.5]), u)))
            else:
                assert bvp.jacobian.check_jacobian(
                    sum(problem.m),
                    lambda u: N.squeeze(problem.f(0.5, u)),
                    lambda u: problem.df(0.5, u))
            assert bvp.jacobian.check_jacobian(
                sum(problem.m),
                lambda u: N.squeeze(problem.g(u, 0*u)),
                lambda u: problem.dg(u, 0*u)[0])
            assert bvp.jacobian.check_jacobian(
                sum(problem.m),
                lambda u: N.squeeze(problem.g(0*u, u)),
                lambda u: problem.dg(0*u, u)[1])

    def check_solving(self):
        """Solve problem #3 with numerical partial derivatives"""
        problem = test_problems.Problem3()

        def dfsub(x, z):
            df = N.empty([1, 2, len(x)])
            for k in range(z.shape[1]):
                df[:,:,k] = bvp.jacobian.jacobian(
                    lambda u: problem.f(x, u), z[:,k], eps=1e-6)
            return df
        def gsub(z):
            return problem.g(z[:,0], z[:,1])
        def dgsub(z):
            za = z[:,0]
            zb = z[:,1]
            xa = bvp.jacobian.jacobian(lambda u: problem.g(u, za), za,eps=1e-6)
            xb = bvp.jacobian.jacobian(lambda u: problem.g(zb, u), zb,eps=1e-6)
            return N.r_[xa[None,0,:], xb[None,1,:]]

        solution = bvp.colnew.solve(
            [problem.a, problem.b],
            problem.m, problem.f, gsub,
            dfsub=dfsub, dgsub=dgsub,
            initial_guess=problem.guess,
            tolerances=[1e-5, 1e-5])

        x = N.linspace(problem.a, problem.b, 100)
        assert N.allclose(problem.exact_solution(x), solution(x)[:,0],
                          rtol=1e-5)

    def check_solving_2(self):
        """Solve problem #3 with numerical partial derivatives"""
        problem = test_problems.Problem3()

        def gsub(z):
            return problem.g(z[:,0], z[:,1])

        solution = bvp.colnew.solve(
            [problem.a, problem.b],
            problem.m, problem.f, gsub,
            initial_guess=problem.guess,
            tolerances=[1e-5, 1e-5])

        x = N.linspace(problem.a, problem.b, 100)
        assert N.allclose(problem.exact_solution(x), solution(x)[:,0],
                          rtol=1e-5)
        
test_doc = get_doctest_checker([bvp.jacobian])

if __name__ == "__main__":
    ScipyTest().run()
