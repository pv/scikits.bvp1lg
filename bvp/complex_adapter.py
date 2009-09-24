import numpy as np

class ComplexAdapter(object):
    def fsub(self, x, z):
        c_z = z[::2] + 1j*z[1::2]
        c_f = self.c_fsub(x, c_z)

        r_f = np.empty((2*c_f.shape[0], x.shape[0]), dtype=np.float_)
        r_f[0::2] = c_f.real
        r_f[1::2] = c_f.imag

        return r_f

    def gsub(self, z):
        c_z = z[::2,::2] + 1j*z[1::2,::2]
        c_g = self.c_gsub(c_z)

        r_g = np.empty((2*c_g.shape[0],), dtype=np.float_)
        r_g[0::2] = c_g.real
        r_g[1::2] = c_g.imag

        return r_g

    def dfsub(self, x, z):
        c_z = z[::2] + 1j*z[1::2]
        c_df = self.c_dfsub(x, c_z)

        r_df = np.empty((2*c_df.shape[0], 2*c_df.shape[1]) + c_df.shape[2:],
                        dtype=np.float_)

        r_df[0::2,0::2] = c_df.real
        r_df[0::2,1::2] = -c_df.imag
        r_df[1::2,0::2] = c_df.imag
        r_df[1::2,1::2] = c_df.real

        return r_df

    def dgsub(self, z):
        c_z = z[::2] + 1j*z[1::2]
        c_g = self.c_gsub(c_z)

        r_dg = np.empty((2*c_dg.shape[0], 2*c_dg.shape[1]) + c_dg.shape[2:],
                        dtype=np.float_)

        r_dg[0::2,0::2] = c_dg.real
        r_dg[0::2,1::2] = -c_dg.imag
        r_dg[1::2,0::2] = c_dg.imag
        r_dg[1::2,1::2] = c_dg.real

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
            self.dgsub = None

        def duplist(x):
            if x is None:
                return x
            xs = []
            for y in x:
                xs.extend([y,y])
            return xs

        self.degrees = duplist(degrees)
        self.boundary_points = duplist(boundary_points)
        self.tolerances = duplist(tolerances)

class ComplexSolution(object):
    def __init__(self, solution):
        self.r_solution = solution

    def __call__(self, x):
        r = self.r_solution.__call__(x)
        return r[:,0::2] + 1j*r[:,1::2]

    def __getattr__(self, name):
        return getattr(self.r_solution, name)
