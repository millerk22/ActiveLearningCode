import numpy as np
import scipy.linalg as sla
from scipy.special import erfc, erfcinv


class TruncRandNormMulticlass(object):
    """ Multiclass Truncated Random Normal sampler object.

    *** Ported to Python from MATLAB code of code/multiclassUQGibbs/trandn_multiclass.m ***


    Attributes:
        K : int, number of classes
        Qs : dict of numpy arrays, matrices for use in the sampling
        invQs : dict of numpy arrays, matrices for use in the sampling
    """
    def __init__(self, K):
        """ Instantiate the dictionaries for sampling later"""
        self.K = K
        self.Qs = {}
        self.invQs = {}
        for i in range(self.K):
            Q = np.zeros((self.K,self.K))
            Q[:,0] = np.ones(self.K)
            count = 1
            for j in range(self.K):
                if j != i:
                    Q[i, count] = 1
                    Q[j, count] = -1
                    count += 1
            Q = Q.T
            for l in range(self.K):
                Q[l,:] /= np.linalg.norm(Q[l,:])
            self.Qs[i] = Q
            self.invQs[i] = sla.inv(Q)

    def gen_samples(self, u, gamma, k):
        """ Generate samples for the given class, k"""
        N = u.shape[0]
        ut = u.T
        Q = self.Qs[k]
        Q_inv = self.invQs[k]
        u_e = Q.dot(ut)
        v_e = u_e.copy()
        v_e[0,:] = u_e[0,:] + gamma * np.random.randn(N)
        tmp = u_e[1:,:].T.flatten()
        u_inf = np.inf * np.ones(len(tmp))
        v_tmp = trandn(-tmp/gamma, u_inf)*gamma + tmp
        v_e[1:,:] = v_tmp.reshape(N, self.K-1).T
        v = Q_inv.dot(v_e)
        return v.T


"""   Functions for Multivariate Truncated Normal Samples """

def ntail(l, u):
    """ Samples a vector of length = len(l)=len(u)
    from the standard multivariate normal distribution, truncated over the
    region [l,u], where l>0 and l and u are column vectors.
        Uses acceptance-rejection from Rayleigh distribution similar to Marsaglia (1964)
    """
    if len(l) != len(u):
        raise ValueError("Lengths of input arrays must match")

    c = 0.5*l**2.
    n = len(l)
    f = np.expm1(c - 0.5*u**2.)
    x = c - np.log(np.real(1. + np.random.rand(n) * f))  # sample using Raleigh
    # keep list of rejected
    I = np.where(np.random.rand(n)**2. * x > c)[0]
    d = len(I)
    while d > 0:
        cy = c[I]
        y = cy - np.log(np.real(1. + np.random.rand(d)*f[I]))
        idx = np.random.randn(d)**2. * y < cy   # accepted
        x[I[idx]] = y[idx]  # store the accepted
        I = I[~idx] # remove accepted from list
        d = len(I)  # number of rejected
    x = np.sqrt(2*x)
    return x

def tn(l,u):
    """ Samples a column vector of length=len(l)=len(u)from the standard
    multivariate normal distribution, truncated over the region [l,u],
    where -a<l<u<a for some 'a' and l and u are column vectors """
    tol = 2. # controls switch between methods
    I = np.abs(u-l) > tol
    x = l
    if np.any(I):
        tl = l[I]
        tu = u[I]
        x[I] = trnd(tl, tu)

    # Case: np.abs(u-l)< tol, uses inverse-transform
    I = ~I
    if np.any(I):
        tl = l[I]
        tu = u[I]
        pl = 0.5*erfc(tl/np.sqrt(2))
        pu = 0.5*erfc(tu/np.sqrt(2))
        x[I] = np.sqrt(2)*erfcinv(2.*(pl - (pl-pu)*np.random.rand(*tl.shape)))
    return x

def trnd(l,u):
    """ Uses acceptance/rejection to simulate from truncated normal """
    x = np.random.randn(*l.shape) # sample normal
    I = np.where((x < l) | (x > u))[0]
    d = len(I)
    while d > 0:
        ly = l[I] # find the thresholds of rejected
        uy = u[I]
        y = np.random.randn(*ly.shape)
        idx = (y > ly) & (y < uy) # accepted
        x[I[idx]] = y[idx] # store the accepted
        I = I[~idx] # remove accepted from list
        d = len(I) # number of rejected
    return x

def trandn(l, u, seed=None):
    """ Efficient generator of a vector of length = len(l) = len(u)
    from the standard multivariate normal distribution, truncated over the
    region [l,u]. (infinite values for 'u' and 'l' are accepted)

    *** Ported to Python from MATLAB code of code/multiclassUQGibbs/trandn.m ***

    Reference: Botev, Z. I. (2016). "The normal law under linear restrictions:
    simulation and estimation via minimax tilting". Journal of the
    Royal Statistical Society: Series B (Statistical Methodology).
    doi:10.1111/rssb.12162
    """
    if len(l) != len(u):
        raise ValueError("Lengths of input arrays must match")
    n = len(l)
    x = np.empty(n)
    x[:] = np.nan
    a = .66 # threshold for switching between methods. Can be tuned for max spped

    # Case 1 : a < l < u
    I = l > a

    if np.any(I):
        tl = l[I]
        tu = u[I]
        x[I] = ntail(tl, tu)

    # Case 2 : l < u < -a
    J = u < -a
    if np.any(J):
        tl = -u[J]
        tu = -l[J]
        x[J] = -ntail(tl, tu)

    # Case 3 : otherwise use inverse transform or accept-reject
    I = ~(I | J)
    if np.any(I):
        tl = l[I]
        tu = u[I]
        x[I] = tn(tl, tu)
    return x
