# Author: Pauli Virtanen <pav@iki.fi>, 2006.
# All rights reserved. See LICENSE.txt.
"""
Numerical approximations to Jacobians of functions
"""
import scipy as _N

def jacobian(f, u, eps=1e-6):
    """Evaluate partial derivatives of f(u) numerically.

    :note:
        This routine is currently naive and could be improved.

    :returns:
        (*f.shape, *u.shape) array ``df``, where df[i,j] ~= (d f_i / u_j)(u)
    """
    f0 = _N.asarray(f(u)) # asarray: because of matrices

    u_shape = u.shape
    nu = _N.prod(u_shape)

    f_shape = f0.shape
    nf = _N.prod(f_shape)

    df = _N.empty([nf, nu], dtype=u.dtype)
    
    for k in range(nu):
        du = _N.zeros(nu, dtype=u.dtype)
        du[k] = max(eps*abs(u.flat[k]), eps)
        f1 = _N.asarray(f(u + _N.reshape(du, u_shape)))
        df[:,k] = _N.reshape((f1 - f0) / eps, [nf])

    df.shape = f_shape + u_shape
    return df

def check_jacobian(N, f, df, bounds=None,
                   eps=1e-6, rtol=1e-3, atol=1e-8, times=None):
    """Check that ``df`` is a partial derivative of ``f``.

    This is done by computing (f(u + eps*e_k) - f(u))/eps and checking
    its difference from ``df(u)`` in norm-2.

    :note:
        This routine is currently naive and could be improved.

    :Parameters:
      - `N` : number of variables
      - `f`: f(u) should return array(N) for u=array(N)
      - `df`: df(u) should return array(N,N) for u=array(N)
      - `bounds`: bounds for elements of u, as [(lower[0], upper[0]), ...]
      - `eps`: epsilon to use for evaluating the partial derivatives
      - `rtol`: relative tolerance to allow
      - `atol`: absolute tolerance to allow
      - `times`: how many random checks to perform

    :returns:
      True if ``df`` passes the test, False otherwise.
    """

    ## Check input & set defaults

    if N <= 0:
        raise ValueError("No variables given")

    if bounds == None:
        bounds = [(0.1, 0.9)]*N

    if times == None:
        times = N
    
    ## Check.

    bounds = _N.asarray(bounds)
    match_seen = False
    
    for k in range(times):
        u  = bounds[:,0] + _N.rand(N) * (bounds[:,1] - bounds[:,0])
        z  = _N.asarray(df(u))
        z2 = jacobian(f, u, eps)
        
        if _N.linalg.norm(z - z2) < atol + .5*rtol*(_N.linalg.norm(z) + _N.linalg.norm(z2)):
            match_seen = True
        else:
            if _N.alltrue(_N.isfinite(z)) and _N.alltrue(_N.isfinite(z2)):
                return False

    if not match_seen:
        raise ValueError("Did not find points where partial derivatives "
                         "were finite")
    return True
