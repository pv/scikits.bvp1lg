"""
Support for complex-analytic equations.

Both the RHS of equations and the boundary conditions must be complex
analytic, ie., complex differentiable in the unknown variables.

"""
from __future__ import absolute_import, division, print_function

import numpy as np
from . import jacobian as _jacobian

class ComplexAdapter(object):
    """
    Convert complex-analytic boundary value problem to a real boundary
    value problem.

    """

    def _unpack_z(self, z):
        """
        Unpack real variable vector to complex vector.

        The complex variables are packed as::

            [Re z1, Re z1', ..., Re z2, Re z2', ..., ...,
             Im z1, Im z1', ..., Im z2, Im z2', ..., ...]

        """
        m = z.shape[0]//2
        return z[:m] + 1j*z[m:]

    def fsub(self, x, z):
        """
        Unpack complex RHS equations to real ones.

        The equations are packed in the same way as the variables::

            [ d^n1 Re z1 = Re[ rhs1 ],
              d^n2 Re z2 = Re[ rhs2 ],
              ...,
              d^n1 Im z1 = Im[ rhs1 ],
              d^n2 Im z2 = Im[ rhs2 ],
              ... ]
        """
        x = np.atleast_1d(x)
        c_z = self._unpack_z(z)
        c_f = self.c_fsub(x, c_z)
        c_f = np.asarray(c_f)

        m = c_f.shape[0]

        r_f = np.empty((2*c_f.shape[0], x.shape[0]), dtype=np.float_)
        r_f[:m] = c_f.real
        r_f[m:] = c_f.imag

        return r_f

    def dfsub(self, x, z):
        """
        Unpack complex partial derivatives of RHS to real ones.

        This assumes that the rhs functions are complex analytic.

        """
        x = np.atleast_1d(x)
        c_z = self._unpack_z(z)
        c_df = self.c_dfsub(x, c_z)
        c_df = np.asarray(c_df)

        m = c_z.shape[0]

        r_df = np.empty((2*c_df.shape[0], 2*c_df.shape[1]) + c_df.shape[2:],
                        dtype=np.float_)

        r_df[:m,:m] = c_df.real
        r_df[:m,m:] = -c_df.imag
        r_df[m:,:m] = c_df.imag
        r_df[m:,m:] = c_df.real

        return r_df

    def gsub(self, z):
        """
        Unpack complex boundary condition equations to real ones.

        The boundary conditions are packed as::

            [ Re g_1, Im g_1, Re g_2, Im g_2, ... ]

        This order is different than for the equations or variables,
        because we need to preserve the order of the boundary points.

        """
        #
        # Note: g[j] can only depend on z[:,j]
        #
        # However, since we know that boundary points for re/im are the
        # same, we can assume all(z[:,0::2] == z[:,1::2]), to avoid
        # calling c_gsub twice.
        #
        # This requires re-implementation of automatic differentation
        # in dgsub below -- complex analyticity gives more power to
        # differentiation.
        #
        c_z = self._unpack_z(z[:,0::2])
        c_g = self.c_gsub(c_z)
        c_g = np.asarray(c_g)

        r_g = np.empty((2*c_g.shape[0],), dtype=np.float_)
        r_g[0::2] = c_g.real
        r_g[1::2] = c_g.imag

        return r_g

    def dgsub(self, z):
        """
        Unpack complex partial derivatives of the boundary conditions

        This assumes that the boundary conditions are complex analytic.

        """
        #
        # Note: dg[j] can only depend on z[:,j]
        #
        # However, here we assume that  all(z[:,0::2] == z[:,1::2]),
        # to avoid needing to call c_gsub twice.
        #
        c_z = self._unpack_z(z[:,0::2])
        c_dg = self.c_dgsub(c_z)
        c_dg = np.asarray(c_dg)

        m = c_z.shape[0]

        r_dg = np.empty((2*c_dg.shape[0], 2*c_dg.shape[1]), dtype=np.float_)

        r_dg[0::2,:m] = c_dg.real
        r_dg[0::2,m:] = -c_dg.imag
        r_dg[1::2,:m] = c_dg.imag
        r_dg[1::2,m:] = c_dg.real

        return r_dg

    def dgsub_numerical(self, z):
        """
        Compute partial derivatives of the boundary conditions numerically

        Pack result to reals -- this assumes that the rhs functions
        are complex analytic.

        """
        #
        # Reimplementation of numerical differentiation, for complex vars,
        # making use of complex analyticity
        #
        c_z = self._unpack_z(z[:,0::2])
        c_zero = np.zeros([c_z.shape[0]], dtype=c_z.dtype)

        mstar = sum(self.degrees) // 2
        c_dg = _jacobian.jacobian(
            lambda u: np.reshape(self.c_gsub(c_z + u[:,None]), [mstar]),
            c_zero)

        m = c_z.shape[0]

        r_dg = np.empty((2*c_dg.shape[0], 2*c_dg.shape[1]) + c_dg.shape[2:],
                        dtype=np.float_)
        r_dg[0::2,:m] = c_dg.real
        r_dg[0::2,m:] = -c_dg.imag
        r_dg[1::2,:m] = c_dg.imag
        r_dg[1::2,m:] = c_dg.real

        return r_dg

    def __init__(self, boundary_points, degrees, fsub, gsub,
                 dfsub=None, dgsub=None, tolerances=None):
        self.c_fsub = fsub
        self.c_gsub = gsub 
        self.c_dfsub = dfsub
        self.c_dgsub = dgsub

        if dfsub is None:
            self.dfsub = None

        if dgsub is None:
            self.dgsub = self.dgsub_numerical

        # Choose degrees according to the above variable packing scheme

        self.degrees = list(degrees) + list(degrees)

        if tolerances is not None:
            self.tolerances = list(tolerances) + list(tolerances)
        else:
            self.tolerances = None

        self.boundary_points = np.repeat(boundary_points, 2)

class ComplexSolution(object):
    """
    Convert a real solution of a complex-valued problem to complex-valued
    solution.

    """

    def __init__(self, solution):
        self.r_solution = solution

    def __call__(self, x):
        r = self.r_solution.__call__(x)
        m = r.shape[1]//2
        return r[:,:m] + 1j*r[:,m:]

    def __getattr__(self, name):
        return getattr(self.r_solution, name)
