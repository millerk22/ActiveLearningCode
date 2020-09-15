'''
Graph Based SSL model class
'''
from .al_util import *

VALID_MODELS = ['gr', 'probit-log', 'probit-norm']

class BinaryGraphBasedSSLModel(object):
    '''
    Implements full storage of C_tau, C.
        * Can have truncated eigenvalues and eigenvectors


    Get rid of storing eigenvectors and eigenvalues?? (Since we're already committing to storing C_tau and C fully)
    '''
    def __init__(self, modelname, gamma, tau, v=None, w=None, Ct=None):
        self.gamma = gamma
        self.tau = tau
        if v is None:
            raise ValueError("Need to provide the eigenvectors in the variable 'v'")
        if w is None:
            raise ValueError("Need to provide the eigenvalues in the variable 'w'")
        self.v = v
        if self.v.shape[0] != self.v.shape[1]:
            self.trunc = True
        else:
            self.trunc = False
        self.w = w
        self.d = (self.tau ** (-2.)) * ((self.w + self.tau**2.))
        #self.d = self.w + self.tau**2.
        if Ct is not None:
            self.Ct = Ct
        else:
            self.Ct = (self.v * (1./self.d)) @ self.v.T
        if modelname not in VALID_MODELS:
            raise ValueError("%s is not a valid modelname, must be in %s" % (modelname, str(VALID_MODELS)))
        self.modelname = modelname
        self.full_storage = True
        self.m = None
        self.C = None
        return

    def calculate_model(self, labeled, y):
        self.m = self.get_m(labeled, y)
        self.C = self.get_C(labeled, y, self.m)
        self.labeled = labeled
        self.y = y
        self.unlabeled = list(filter(lambda x: x not in self.labeled, range(self.C.shape[0])))
        return

    def update_model(self, Q, yQ, exact=False):
        if self.m is None or self.C is None:
            print("Previous model not defined, so assuming you are passing in initial labeled set and labelings...")
            self.calculate_model(Q, yQ)
            return
        if not exact:
            for k,yk in zip(Q, yQ):
                if self.modelname == 'gr':
                    self.m += (yk - self.m[k])/(self.gamma**2 + self.C[k,k])*self.C[:, k]
                    self.C -= np.outer(self.C[:,k], self.C[:,k])/(self.gamma**2. + self.C[k,k])
                elif self.modelname == 'probit-log':
                    self.m -= jac_calc2(self.m[k], yk, self.gamma) / (1. + self.C[k,k] * hess_calc2(self.m[k], yk, self.gamma))*self.C[k,:]
                    self.C -= hess_calc2(self.m[k], yk, self.gamma)/(1. + self.C[k,k] * hess_calc2(self.m[k], yk, self.gamma))*np.outer(self.C[k,:], self.C[k,:])
                elif self.modelname == 'probit-norm':
                    self.m -= jac_calc(self.m[k], yk, self.gamma) / (1. + self.C[k,k] * hess_calc(self.m[k], yk, self.gamma))*self.C[k,:]
                    self.C -= hess_calc(self.m[k], yk, self.gamma)/(1. + self.C[k,k]*hess_calc(self.m[k], yk, self.gamma))*np.outer(self.C[k,:], self.C[k,:])
            else:
                raise ValueError("model name %s not recognized or implemented" % str(self.modelname))
            self.labeled += list(Q)
            self.y += list(yQ)
        else:
            self.labeled += list(Q)
            self.y += list(yQ)
            self.calculate_model(self.labeled, self.y)

        return

    def get_m(self, Z, y):
        if self.modelname == "probit-norm":
            if len(y) <= len(self.w):
                return probit_map_dr(Z, y, self.gamma, self.Ct)
            else:
                return probit_map_st(Z, y, self.gamma, self.d, self.v)
        elif self.modelname == "probit-log":
            if len(y) <= len(self.w):
                return probit_map_dr2(Z, y, self.gamma, self.Ct)
            else:
                return probit_map_st2(Z, y, self.gamma, self.d, self.v)
        elif self.modelname == "gr":
            return gr_map(Z, y, self.gamma, self.Ct)
        else:
            raise ValueError("did not recognize modelname = %s" % self.modelname)

    def get_C(self, Z, y, m):
        if self.modelname == "probit-norm":
            if len(y) <= len(self.w) or not self.trunc:
                return Hess_inv(m, Z, y, self.gamma, self.Ct)
            else:
                return Hess_inv_st(m, Z, y, self.d, self.v, self.gamma)
        elif self.modelname == "probit-log":
            if len(y) <= len(self.w) or not self.trunc:
                return Hess2_inv(m, Z, y, self.gamma, self.Ct)
            else:
                return Hess_inv_st2(m, Z, y, self.d, self.v, self.gamma)
        elif self.modelname == "gr":
            #return gr_C(Z, self.gamma, self.Ct) # old
            return gr_C(Z, self.gamma, self.d, self.v)
        else:
            raise ValueError("did not recognize modelname = %s" % self.modelname)


class MultiGraphBasedSSLModel(object):
    '''
    Implements full storage of C_tau, C.
        * Can have truncated eigenvalues and eigenvectors


    Get rid of storing eigenvectors and eigenvalues?? (Since we're already committing to storing C_tau and C fully)
    '''
    def __init__(self, modelname, gamma, tau, v=None, w=None, Ct=None):
        self.gamma = gamma
        self.tau = tau
        if v is None:
            raise ValueError("Need to provide the eigenvectors in the variable 'v'")
        if w is None:
            raise ValueError("Need to provide the eigenvalues in the variable 'w'")
        self.v = v
        if self.v.shape[0] != self.v.shape[1]:
            self.trunc = True
        else:
            self.trunc = False
        self.w = w
        self.d = (self.tau ** (-2.)) * ((self.w + self.tau**2.))
        #self.d = self.w + self.tau**2.
        if Ct is not None:
            self.Ct = Ct
        else:
            self.Ct = (self.v * (1./self.d)) @ self.v.T
        if modelname not in VALID_MODELS:
            raise ValueError("%s is not a valid modelname, must be in %s" % (modelname, str(VALID_MODELS)))
        self.modelname = modelname
        self.full_storage = True
        self.m = None
        self.C = None
        return

    def calculate_model(self, labeled, y):
        if self.modelname == 'gr':
            self.C = gr_C(labeled, self.gamma, self.d, self.v)
            self.m = (self.C[:,labeled] @ y)/(self.gamma**2.)
        else:
            self.m = self.get_m(labeled, y)
            self.C = self.get_C(labeled, y, self.m)
        self.labeled = labeled
        self.y = y
        self.unlabeled = list(filter(lambda x: x not in self.labeled, range(self.C.shape[0])))
        return

    def update_model(self, Q, yQ, exact=False):
        if self.m is None or self.C is None:
            print("Previous model not defined, so assuming you are passing in initial labeled set and labelings...")
            self.calculate_model(Q, yQ)
            return
        if not exact:
            print("... Updating model with NA updates ...")
            if self.modelname == 'gr':
                for k,yk in zip(Q, yQ): # done
                    self.m += self.C[:, k].reshape((-1, 1)) @ (yk - self.m[k])/(self.gamma**2 + self.C[k,k])
                    self.C -= np.outer(self.C[:,k], self.C[:,k])/(self.gamma**2. + self.C[k,k])
            # Probit not implemented for Multiclass
            # elif self.modelname == 'probit-log':
            #     for k,yk in zip(Q, yQ):
            #         self.m -= jac_calc2(self.m[k], yk, self.gamma) / (1. + self.C[k,k] * hess_calc2(self.m[k], yk, self.gamma))*self.C[k,:]
            #         self.C -= hess_calc2(self.m[k], yk, self.gamma)/(1. + self.C[k,k] * hess_calc2(self.m[k], yk, self.gamma))*np.outer(self.C[k,:], self.C[k,:])
            # elif self.modelname == 'probit-norm':
            #     for k,yk in zip(Q, yQ):
            #         self.m -= jac_calc(self.m[k], yk, self.gamma) / (1. + self.C[k,k] * hess_calc(self.m[k], yk, self.gamma))*self.C[k,:]
            #         self.C -= hess_calc(self.m[k], yk, self.gamma)/(1. + self.C[k,k]*hess_calc(self.m[k], yk, self.gamma))*np.outer(self.C[k,:], self.C[k,:])
            else:
                raise ValueError("model name %s not recognized or implemented" % str(self.modelname))
            self.labeled += list(Q)
            self.y = np.concatenate((self.y, yQ))
        else:
            print("... Updating model with EXACT updates ...")
            self.labeled += list(Q)
            self.y = np.concatenate((self.y, yQ))
            self.calculate_model(self.labeled, self.y)

        return

    def get_m(self, Z, y):
        # PROBIT NOT IMPLEMENTED FOR MULTI
        # if self.modelname == "probit-norm":
        #     if len(y) <= len(self.w):
        #         return probit_map_dr(Z, y, self.gamma, self.Ct)
        #     else:
        #         return probit_map_st(Z, y, self.gamma, self.d, self.v)
        # elif self.modelname == "probit-log":
        #     if len(y) <= len(self.w):
        #         return probit_map_dr2(Z, y, self.gamma, self.Ct)
        #     else:
        #         return probit_map_st2(Z, y, self.gamma, self.d, self.v)
        if self.modelname == "gr":
            return gr_map(Z, y, self.gamma, self.d, self.v)
        else:
            raise ValueError("did not recognize modelname = %s" % self.modelname)

    def get_C(self, Z, y, m):
        # PROBIT NOT IMPLEMENTED FOR MULTI
        # if self.modelname == "probit-norm":
        #     if len(y) <= len(self.w) or not self.trunc:
        #         return Hess_inv(m, Z, y, self.gamma, self.Ct)
        #     else:
        #         return Hess_inv_st(m, Z, y, self.d, self.v, self.gamma)
        # elif self.modelname == "probit-log":
        #     if len(y) <= len(self.w) or not self.trunc:
        #         return Hess2_inv(m, Z, y, self.gamma, self.Ct)
        #     else:
        #         return Hess_inv_st2(m, Z, y, self.d, self.v, self.gamma)
        if self.modelname == "gr":
            #return gr_C(Z, self.gamma, self.Ct)
            return gr_C(Z, self.gamma, self.d, self.v)
        else:
            raise ValueError("did not recognize modelname = %s" % self.modelname)





class BinaryGraphBasedSSLModelReduced(object):
    '''
    NEED TO FINISH.. NEED TO PREPARE SO STORAGE IS IN ALPHA SPACE.
    '''
    def __init__(self, modelname, gamma, tau, v=None, w=None):
        self.gamma = gamma
        self.tau = tau
        if v is None:
            raise ValueError("Need to provide the eigenvectors in the variable 'v'")
        if w is None:
            raise ValueError("Need to provide the eigenvalues in the variable 'w'")
        self.v = v
        self.trunc = True
        if self.v.shape[0] == self.v.shape[1]:
            print("Warning : It appears that you've given the full spectrum, this class is not optimized for that case...")
        self.w = w
        self.d = (self.tau ** (-2.)) * ((self.w + self.tau**2.))
        #self.d = self.w + self.tau**2.
        if modelname not in VALID_MODELS:
            raise ValueError("%s is not a valid modelname, must be in %s" % (modelname, str(VALID_MODELS)))
        self.full_storage = False
        self.modelname = modelname
        self.m = None
        self.alpha = None
        self.C_a = None
        return

    def calculate_model(self, labeled, y):
        self.alpha = self.get_alpha(labeled, y)
        self.C_a = self.get_C_alpha(labeled, y)
        self.m = self.v @ self.alpha
        self.labeled = labeled
        self.y = y
        self.unlabeled = list(filter(lambda x: x not in self.labeled, range(self.v.shape[0])))
        return

    def update_model(self, Q, yQ, exact=False):
        if self.alpha is None or self.C_a is None:
            print("Previous model not defined, so assuming you are passing in initial labeled set and labelings...")
            self.calculate_model(Q, yQ)
            return
        if not exact:
            for k, yk in zip(Q, yQ):
                C_a_vk = self.C_a @ (self.v[k,:].T)
                ip = np.inner(self.v[k,:], C_a_vk)
                mk = np.inner(self.v[k,:], self.alpha)
                if self.modelname == 'gr':
                    self.alpha += (yk - mk)/(self.gamma**2 + ip) * C_a_vk
                    self.C_a -= np.outer(C_a_vk, C_a_vk)/(self.gamma**2. + ip)
                elif self.modelname == 'probit-log':
                    self.alpha -= jac_calc2(mk, yk, self.gamma) / (1. + ip * hess_calc2(mk, yk, self.gamma))*C_a_vk
                    mk = np.inner(self.v[k,:], self.alpha)
                    self.C_a -= hess_calc2(mk, yk, self.gamma)/(1. + ip * hess_calc2(mk, yk, self.gamma))*np.outer(C_a_vk, C_a_vk)
                elif self.modelname == 'probit-norm':
                    self.alpha -= jac_calc(mk, yk, self.gamma) / (1. + ip * hess_calc(mk, yk, self.gamma))*C_a_vk
                    mk = np.inner(self.v[k,:], self.alpha)
                    self.C_a -= hess_calc(mk, yk, self.gamma)/(1. + ip * hess_calc(mk, yk, self.gamma))*np.outer(C_a_vk, C_a_vk)
                else:
                    raise ValueError("model name %s not recognized or implemented" % str(model))
            self.m = self.v @ self.alpha
            self.labeled += list(Q)
            self.y += list(yQ)
        else:
            self.labeled += list(Q)
            self.y += list(yQ)
            self.calculate_model(self.labeled, self.y)

        return

    def get_alpha(self, Z, y):
        '''
        TODO: implement different option for when there are fewer labels than eigenvalues...
        '''
        if self.modelname == "probit-norm":
            return probit_map_st_alpha(Z, y,  self.gamma, self.d, self.v)
            # if len(y) <= len(self.w):
            #     return probit_map_dr(Z, y, self.gamma, self.Ct)
            # else:
            #     return probit_map_st(Z, y, self.gamma, self.d, self.v)
        elif self.modelname == "probit-log":
            return probit_map_st2_alpha(Z, y,  self.gamma, self.d, self.v)
            # if len(y) <= len(self.w):
            #     return probit_map_dr2(Z, y, self.gamma, self.Ct)
            # else:
            #     return probit_map_st2(Z, y, self.gamma, self.d, self.v)
        elif self.modelname == "gr":
            vZ = self.v[Z, :]
            C_a = np.diag(self.d) + vZ.T @ vZ / (self.gamma**2.)
            return (1. / self.gamma**2.)* sp.linalg.inv(C_a) @ vZ.T @ np.array(y)
        else:
            pass

    def get_C_alpha(self, Z, y):
        '''
        TODO: implement different option for when there are fewer labels than eigenvalues...
        '''
        if self.modelname == "probit-norm":
            return Hess_inv_st_alpha(self.alpha, y, 1./self.d, self.v[Z,:], self.gamma)
            # if len(y) <= len(self.w):
            #     return Hess_inv(m, Z, y, self.gamma, self.Ct)
            # else:
            #     return Hess_inv_st(m, Z, y, self.d, self.v, self.gamma)

        elif self.modelname == "probit-log":
            return Hess_inv_st2_alpha(self.alpha, y, 1./self.d, self.v[Z,:], self.gamma)
            # if len(y) <= len(self.w):
            #     return Hess2_inv(m, Z, y, self.gamma, self.Ct)
            # else:
            #     return Hess2_inv_st2(m, Z, y, self.d, self.v, self.gamma)

        elif self.modelname == "gr":
            vZ = self.v[Z, :]
            C_a = np.diag(self.d) + vZ.T @ vZ / (self.gamma**2.)
            return sp.linalg.inv(C_a)
        else:
            pass

##################################################################################
################### Helper Functions for Reduced Model ###########################
##################################################################################


def Hess_inv_st2_alpha(alpha, y, d, v_Z, gamma):
    '''
    This method keeps everything in the "alpha"-space, of dimension num_eig x num_eig
    '''
    Dprime = sp.sparse.diags([1./hess_calc2(np.inner(v_Z[i,:],alpha), yi, gamma) for i, yi in enumerate(y)], format='csr')
    A = d[:, np.newaxis] * ((v_Z.T @ sp.linalg.inv(Dprime + v_Z @ (d[:, np.newaxis] * (v_Z.T))) @ v_Z) * d)
    return np.diag(d) - A


def Hess_inv_st_alpha(alpha, y, d, v_Z, gamma):
    '''
    This method keeps everything in the "alpha"-space, of dimension num_eig x num_eig
    '''
    Dprime = sp.sparse.diags([1./hess_calc(np.inner(v_Z[i,:],alpha), yi, gamma) for i, yi in enumerate(y)], format='csr')
    A = d[:, np.newaxis] * ((v_Z.T @ sp.linalg.inv(Dprime + v_Z @ (d[:, np.newaxis] * (v_Z.T))) @ v_Z) * d)
    return np.diag(d) - A

def probit_map_st2_alpha(Z, y,  gamma, w, v):
    n,l = v.shape[1], len(y)
    def f(x):
        vec = np.zeros(l)
        for j, (i,yi) in enumerate(zip(Z,y)):
            vec[j] = - jac_calc2(np.inner(v[i,:], x), yi, gamma)
        return w * x  - v[Z,:].T @ vec
    def fprime(x):
        vec = np.zeros(l)
        for j, (i,yi) in enumerate(zip(Z, y)):
            vec[j] = -hess_calc2(np.inner(v[i,:], x), yi, gamma)

        H = (-v[Z,:].T * vec) @ v[Z,:]
        H[np.diag_indices(n)] += w
        return H
    x0 = np.random.rand(len(w))
    res = root(f, x0, jac=fprime)
    #print(f"Root Finding is successful: {res.success}")
    return res.x

def probit_map_st_alpha(Z, y,  gamma, w, v):
    n,l = v.shape[1], len(y)
    def f(x):
        vec = np.zeros(l)
        for j, (i,yi) in enumerate(zip(Z,y)):
            vec[j] = - jac_calc(np.inner(v[i,:], x), yi, gamma)
        return w * x  - v[Z,:].T @ vec
    def fprime(x):
        vec = np.zeros(l)
        for j, (i,yi) in enumerate(zip(Z, y)):
            vec[j] = -hess_calc(np.inner(v[i,:], x), yi, gamma)

        H = (-v[Z,:].T * vec) @ v[Z,:]
        H[np.diag_indices(n)] += w
        return H
    x0 = np.random.rand(len(w))
    res = root(f, x0, jac=fprime)
    #print(f"Root Finding is successful: {res.success}")
    return res.x





# def probit_map_dr2_alpha(Z_, yvals, gamma, Ct):
#     """
#     Probit MAP estimator, using dimensionality reduction via Representer Theorem.
#     *** This uses logistic cdf ***
#     """
#     Ctp = Ct[np.ix_(Z_,Z_)]
#     Jj = len(yvals)
#
#     def f(x):
#         vec = [jac_calc2(x[j], yj, gamma)
#                for j,yj in enumerate(yvals)]
#         return x + Ctp @ vec
#
#     def fprime(x):
#         H = Ctp * np.array([hess_calc2(x[j], yj, gamma)
#                             for j, yj in enumerate(yvals)])
#         if np.any(H == np.inf):
#             print('smally')
#         H[np.diag_indices(Jj)] += 1.0
#         return H
#
#     x0 = np.random.rand(Jj)
#     x0[np.array(yvals) < 0] *= -1
#     res = root(f, x0, jac=fprime)
#     tmp = sla.inv(Ctp) @ res.x
#     return Ct[:, Z_] @ tmp
