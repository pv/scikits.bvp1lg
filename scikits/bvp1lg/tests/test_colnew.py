# Author: Pauli Virtanen <pav@iki.fi>, 2006.
# All rights reserved. See LICENSE.txt.
"""
Tests for the COLNEW wrappers.
"""
from __future__ import division, absolute_import, print_function

from numpy.testing import *
import numpy as np
import inspect

import scikits.bvp1lg.colnew as colnew

from testutils import *
from test_problems import *

###############################################################################

def solve_with_colnew(problem, numerical_jacobians=False,
                      check_jacobian_only=False, **kw):
    assert problem.n_a != None, "COLNEW can solve only separated BCs"

    mstar = sum(problem.m)

    zeta = np.zeros([mstar], np.float_)
    zeta[:problem.n_a] = problem.a
    zeta[problem.n_a:] = problem.b

    a_map = np.where(zeta == problem.a)[0]
    b_map = np.where(zeta == problem.b)[0]

    a_idx = a_map[0]
    b_idx = b_map[0]

    def gsub(z):
        x = problem.g(z[:,a_idx], z[:,b_idx])
        return np.asarray(x)
    
    def dgsub(z):
        x = problem.dg(z[:,a_idx], z[:,b_idx])
        return np.r_[x[0][a_map,:], x[1][b_map,:]]

    if problem.dg is None:
        dgsub = None

    if 'dgsub' not in kw:
        kw['dgsub'] = dgsub

    if 'dfsub' not in kw:
        kw['dfsub'] = problem.df

    if 'tolerances' not in kw:
        kw['tolerances'] = [1e-5]*mstar

    if 'initial_guess' not in kw:
        kw['initial_guess'] = problem.guess

    if numerical_jacobians:
        kw['dfsub'] = None
        kw['dgsub'] = None

    if check_jacobian_only:
        return colnew.check_jacobians(zeta, problem.m,
                                          problem.f, gsub,
                                          dfsub=problem.df,
                                          dgsub=dgsub,
                                          vectorized=problem.vectorized)

    solution = colnew.solve(zeta, problem.m, problem.f, gsub,
                                left=problem.a, right=problem.b,
                                is_linear=problem.linear,
                                vectorized=problem.vectorized,
                                **kw)
    return solution

###############################################################################

class TestColnew(object):
    def test_problem_1(self, num_jac=False):
        # Solve problem #1 and compare to exact solution
        problem = Problem1()
        solution = solve_with_colnew(problem, numerical_jacobians=num_jac)
        x = np.linspace(problem.a, problem.b, 100)
        assert np.allclose(problem.exact_solution(x), solution(x)[:,0],
                          rtol=1e-5)

    def test_problem_2(self, num_jac=False):
        # Solve problem #2 and compare to exact solution
        problem = Problem2()
        solution = solve_with_colnew(problem, numerical_jacobians=num_jac)
        x = np.linspace(problem.a, problem.b, 100)
        assert np.allclose(problem.exact_solution(x), solution(x),
                          rtol=1e-3, atol=1e-6)

    def test_complex_problem_2(self, num_jac=False):
        # Solve problem (complex-valued) #2 and compare to exact solution
        problem = ComplexProblem2()
        solution = solve_with_colnew(problem, numerical_jacobians=num_jac,
                                     is_complex=True)
        x = np.linspace(problem.a, problem.b, 100)
        assert np.allclose(problem.exact_solution(x), solution(x),
                          rtol=1e-3, atol=1e-6)

    def test_problem_3(self, num_jac=False):
        # Solve problem #3 and compare to exact solution
        problem = Problem3()
        solution = solve_with_colnew(problem, numerical_jacobians=num_jac)
        x = np.linspace(problem.a, problem.b, 100)
        assert np.allclose(problem.exact_solution(x), solution(x)[:,0],
                          rtol=1e-5)

    def test_problem_5(self, num_jac=False):
        # Solve problem #5 and compare to exact solution
        problem = Problem5()
        solution = solve_with_colnew(problem, numerical_jacobians=num_jac)
        x = np.linspace(problem.a, problem.b, 100)
        assert np.allclose(problem.exact_solution(x), solution(x),
                          rtol=1e-2)
        # Needed to reduce tolerance, since "exact" solution data is
        # interpolated

    def test_problem_6(self, num_jac=False):
        # Solve problem #6 and compare to exact solution
        problem = Problem6()
        solution = solve_with_colnew(problem, numerical_jacobians=num_jac)
        x = np.linspace(problem.a, problem.b, 100)
        assert np.allclose(problem.exact_solution(x), solution(x),
                          rtol=1e-5)

    def test_problem_7(self, num_jac=False):
        # Solve problem #7 for two different initial guesses
        # and compare to known approximate solutions
        problem = Problem7()

        tol = [1e-4]*4
        
        solution1 = solve_with_colnew(problem, maximum_mesh_size=500,
                                      initial_guess=None,
                                      collocation_points=4,
                                      tolerances=tol,
                                      verbosity=0,
                                      numerical_jacobians=num_jac)

        solution2 = solve_with_colnew(problem, maximum_mesh_size=500,
                                      initial_guess=problem.guess,
                                      collocation_points=4,
                                      tolerances=tol,
                                      verbosity=0,
                                      numerical_jacobians=num_jac)

        x = problem.solution_1[:,0]
        assert curves_close(x, problem.solution_1[:,1],
                            problem.solution_1_xtol, problem.solution_1_ytol,
                            solution1(x)[:,0], 0, 0)

        x = problem.solution_2[:,0]
        assert curves_close(x, problem.solution_2[:,1],
                            problem.solution_2_xtol, problem.solution_2_ytol,
                            solution2(x)[:,0], 0, 0)

    def test_problem_8(self, num_jac=False):
        # Solve problem #8 with simple continuation
        # and compare to known approximate solutions
        problem = Problem8()
        solution = problem.guess
        for c in problem.continuation:
            problem.__dict__.update(c)
            solution = solve_with_colnew(problem, maximum_mesh_size=500,
                                         initial_guess=solution,
                                         numerical_jacobians=num_jac)

        xg = problem.solution_g[:,0]
        assert curves_close(xg, problem.solution_g[:,1],
                            problem.solution_g_xtol, problem.solution_g_ytol,
                            solution(xg)[:,0], 0, 0)

        xh = problem.solution_h[:,0]
        assert curves_close(xh, problem.solution_h[:,1],
                            problem.solution_h_xtol, problem.solution_h_ytol,
                            solution(xh)[:,2], 0, 0)

    def test_problem_9(self, num_jac=False):
        # Solve problem #9
        problem = Problem9()
        solution = solve_with_colnew(problem, numerical_jacobians=num_jac)
        x = np.linspace(problem.a, problem.b, 10)
        assert np.allclose(problem.exact_solution(x), solution(x),
                          rtol=1e-3, atol=1e-6)

    def test_problem_10(self, num_jac=False):
        # Solve problem #10
        problem = Problem10()
        solution = solve_with_colnew(problem, numerical_jacobians=num_jac)
        x = np.linspace(problem.a, problem.b, 10)
        assert np.allclose(problem.exact_solution(x), solution(x),
                          rtol=1e-3, atol=1e-6)

    def test_continuation(self, num_jac=False):
        # Solve problem #3 for multiple values of C using continuation
        problem = Problem3()
        x = np.linspace(problem.a, problem.b, 501)

        # Near C = 1.7 there is a "difficult point", be careful about it.
        Cvals = np.r_[np.linspace(1, 1.69, 5),
                     np.linspace(1.69, 1.72, 10),
                     np.linspace(1.8, 150, 10)]

        # Use simple continuation
        solution = problem.guess
        for C in Cvals:
            problem.C = C
            solution = solve_with_colnew(problem,
                                         initial_guess=solution,
                                         coarsen_initial_guess_mesh=True,
                                         collocation_points=3,
                                         numerical_jacobians=num_jac)
            assert np.allclose(problem.exact_solution(x), solution(x)[:,0],
                              rtol=1e-5)

        # Finding the given analytical solution without continuation
        # is not possible, probably due to a branch point
        solution = solve_with_colnew(problem, numerical_jacobians=num_jac)
        assert not np.allclose(problem.exact_solution(x), solution(x)[:,0],
                              rtol=1e-1)

    def test_initial_mesh(self, num_jac=False):
        # Solve problem #3 with a specified initial mesh
        problem = Problem3()
        x = np.linspace(0, 1, 13)
        solution = solve_with_colnew(problem, initial_mesh=x,
                                     tolerances=[1e-7, 1e-7],
                                     numerical_jacobians=num_jac)
        assert np.allclose(problem.exact_solution(x), solution(x)[:,0],
                          rtol=1e-5)

    def test_fixed_mesh(self, num_jac=False):
        # Solve problem #3 without adaptive mesh selection
        problem = Problem3()
        x = np.linspace(0, 1, 20)

        # Doesn't work without a specified initial mesh
        assert_raises(ValueError,
                      solve_with_colnew,
                      problem, adaptive_mesh_selection=False,
                      numerical_jacobians=num_jac)

        # This works
        solution = solve_with_colnew(problem, initial_mesh=x,
                                     adaptive_mesh_selection=False,
                                     tolerances=[1e-7, 1e-7])

        # Check that the mesh indeed is simple
        mesh_delta = np.diff(solution.mesh)
        assert np.allclose(mesh_delta, mesh_delta[0], rtol=1e-9, atol=1e-9)

    def test_extra_fixed_points(self, num_jac=False):
        # Solve problem #3, specifying additional fixed points
        problem = Problem3()
        solution = solve_with_colnew(problem,
                                     extra_fixed_points=[0.12345, 0.54321],
                                     numerical_jacobians=num_jac)
        assert 0.12345 in solution.mesh and 0.54321 in solution.mesh

    def test_collocation_points(self, num_jac=False):
        # Solve problem #3 with different numbers of collocation points
        problem = Problem3()


        assert_raises(ValueError,
                      solve_with_colnew, problem, collocation_points=1,
                      numerical_jacobians=num_jac)
        
        x = np.linspace(problem.a, problem.b, 100)
        solution = solve_with_colnew(problem, collocation_points=2,
                                     numerical_jacobians=num_jac)
        assert np.allclose(problem.exact_solution(x),solution(x)[:,0],rtol=1e-5)
        solution = solve_with_colnew(problem, collocation_points=3,
                                     numerical_jacobians=num_jac)
        assert np.allclose(problem.exact_solution(x),solution(x)[:,0],rtol=1e-5)
        solution = solve_with_colnew(problem, collocation_points=4,
                                     numerical_jacobians=num_jac)
        assert np.allclose(problem.exact_solution(x),solution(x)[:,0],rtol=1e-5)
        solution = solve_with_colnew(problem, collocation_points=5,
                                     numerical_jacobians=num_jac)
        assert np.allclose(problem.exact_solution(x),solution(x)[:,0],rtol=1e-5)
        solution = solve_with_colnew(problem, collocation_points=6,
                                     numerical_jacobians=num_jac)
        assert np.allclose(problem.exact_solution(x),solution(x)[:,0],rtol=1e-5)
        solution = solve_with_colnew(problem, collocation_points=7,
                                     numerical_jacobians=num_jac)
        assert np.allclose(problem.exact_solution(x),solution(x)[:,0],rtol=1e-5)

        assert_raises(ValueError,
                      solve_with_colnew, problem, collocation_points=8,
                      numerical_jacobians=num_jac)

    def test_tolerances(self, num_jac=False):
        # Solve problem #3 with different tolerances
        problem = Problem3()
        x = np.linspace(0, 1, 20)

        # Large tolerances, less exact solution
        solution = solve_with_colnew(problem, tolerances=[1e-1, 1e-1],
                                     numerical_jacobians=num_jac)
        assert np.allclose(problem.exact_solution(x), solution(x)[:,0],
                          rtol=1e-1)
        assert not np.allclose(problem.exact_solution(x), solution(x)[:,0],
                              rtol=1e-5)

        # Large tolerances, more exact solution
        solution = solve_with_colnew(problem, tolerances=[1e-5, 1e-5],
                                     numerical_jacobians=num_jac)
        assert np.allclose(problem.exact_solution(x), solution(x)[:,0],
                          rtol=1e-5)

        # Invalid number of tolerances
        assert_raises(ValueError,
                      solve_with_colnew, problem, tolerances=[1, 2, 3],
                      numerical_jacobians=num_jac)

    def test_problem_jacobians(self):
        solve_with_colnew(Problem1(), check_jacobian_only=True)
        solve_with_colnew(Problem2(), check_jacobian_only=True)
        solve_with_colnew(Problem3(), check_jacobian_only=True)
        #solve_with_colnew(Problem4(), check_jacobian_only=True)
        solve_with_colnew(Problem5(), check_jacobian_only=True)
        solve_with_colnew(Problem6(), check_jacobian_only=True)
        solve_with_colnew(Problem7(), check_jacobian_only=True)
        solve_with_colnew(Problem8(), check_jacobian_only=True)

    def test_reentrancy(self):
        count = [0]
        depth = [0]
        class RecursiveProblem(Problem1):
            def f(self, x, z):
                # work out a back-and-forth jumping recursion
                # 1 2 3 4 5 6 4 5 5 3 4 3 1 2 2 0
                if count[0] + 2*depth[0] + 3*((count[0]+2) % 3) < 20:
                    count[0] += 1
                    depth[0] += 1
                    problem = RecursiveProblem()
                    solution = solve_with_colnew(problem)
                    depth[0] -= 1
                    xx = np.linspace(problem.a, problem.b, 100)
                    assert np.allclose(problem.exact_solution(xx),
                                      solution(xx)[:,0],
                                      rtol=1e-5)
                return Problem1.f(self, x, z)

        problem1 = RecursiveProblem()
        
        solution = solve_with_colnew(problem1)
        x = np.linspace(problem1.a, problem1.b, 100)
        assert np.allclose(problem1.exact_solution(x), solution(x)[:,0],
                          rtol=1e-5)

###############################################################################

class TestColnewNumericalJacobians(TestColnew):
    # Same as test_colnew, but with numerically approximated Jacobians
    pass

def add_jac_func(cls, v):
    # Needed for making the lambda a closure
    func = lambda self: v(self, True)
    func.__name__ = v.__name__
    setattr(TestColnewNumericalJacobians, func.__name__, func)

def ignored_check(self):
    return True

for name, v in inspect.getmembers(TestColnewNumericalJacobians):
    if name.startswith('test_'):
        argspec = inspect.getargspec(v)
        if 'num_jac' in argspec[0] or 'numerical_jacobians' in argspec[0]:
            add_jac_func(TestColnewNumericalJacobians, v)
        else:
            setattr(TestColnewNumericalJacobians, name, ignored_check)
    del v

def test_doctests():
    assert doctest.testmod(colnew, verbose=0)[0] == 0
