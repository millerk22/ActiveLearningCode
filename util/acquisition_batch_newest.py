# author: Kevin Miller
import numpy as np
from scipy.stats import norm
from util.al_util import *
import time

MODELS = ['gr', 'probit-log', 'probit-norm']

def sgn(x):
    if x >= 0:
        return 1.
    else:
        return -1.

def mc_full(Cand, m, C, modelname, gamma=0.1):
    if modelname not in MODELS:
        raise ValueError("%s is not a valid model name, must be in %s" % (modelname, MODELS))
    if len(m.shape) > 1: # Multiclass case
        if modelname == 'gr':
            return np.array([ np.sqrt(np.inner(m[k,:], m[k,:])[0,0] + 1. - 2.*np.max(m[k,:])) * np.linalg.norm(C[:,k])/(gamma**2. + C[k,k]) for k in Cand])
    else:
        if modelname == 'probit-log':
            return np.array([np.absolute(jac_calc2(m[k], sgn(m[k]), gamma))/(1. + C[k,k]*hess_calc2(m[k], sgn(m[k]), gamma)) \
                           * np.linalg.norm(C[:,k]) for k in Cand])
        elif modelname == 'probit-norm':
            return np.array([np.absolute(jac_calc(m[k], sgn(m[k]), gamma))/(1. + C[k,k]*hess_calc(m[k], sgn(m[k]), gamma)) \
                           * np.linalg.norm(C[:,k]) for k in Cand])
        else:
            return np.array([np.absolute(m[k] - sgn(m[k]))/(gamma**2. + C[k,k]) * np.linalg.norm(C[:,k]) for k in Cand])

def mc_reduced(C_a, alpha, v_Cand, modelname, uks=None, gamma=0.1):
    if modelname not in MODELS:
        raise ValueError("%s is not a valid model name, must be in %s" % (modelname, MODELS))

    if uks is None: # if have not already calculated MAP estimator on full (as we are doing in our Reduced model), then do this calculation
        uks = v_Cand @ alpha
    C_a_vk = C_a @ (v_Cand.T)
    if modelname == 'probit-log':
        return np.array([np.absolute(jac_calc2(uks[i], sgn(uks[i]),gamma))/(1. + \
            np.inner(v_Cand[i,:], C_a_vk[:,i])*hess_calc2(uks[i], sgn(uks[i]),gamma))* np.linalg.norm(C_a_vk[:,i]) \
                        for i in range(v_Cand.shape[0])])
    elif modelname == 'probit-norm':
        return np.array([np.absolute(jac_calc(uks[i], sgn(uks[i]),gamma))/(1. + \
            np.inner(v_Cand[i,:], C_a_vk[:,i])*hess_calc(uks[i], sgn(uks[i]),gamma))* np.linalg.norm(C_a_vk[:,i]) \
                        for i in range(v_Cand.shape[0])])
    else:
        return np.array([np.absolute(uks[i] - sgn(uks[i]))/(gamma**2. + np.inner(v_Cand[i,:], C_a_vk[:,i]))
                           * np.linalg.norm(C_a_vk[:,i]) for i in range(v_Cand.shape[0])])
