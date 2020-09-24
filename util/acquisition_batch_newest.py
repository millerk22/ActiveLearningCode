# author: Kevin Miller
import numpy as np
from scipy.stats import norm
from util.al_util import *
import time

MODELS = ['gr', 'probit-log', 'probit-norm', 'softmax']

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
            raise NotImplementedError("Have not implemented full storage model change calculation for multiclass besides Gaussian Regression ('gr')")
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

    if modelname == 'softmax':
        num_cand, M = v_Cand.shape
        nc = alpha.shape[0]//M
        print(nc)

        if uks is None:
            uks = v_Cand @ (alpha.reshape(nc, M).T)

        v_Cand /= gamma
        piks = np.exp(uks/gamma)
        piks /= np.sum(piks, axis=1)[:,np.newaxis]

        C_aV_candT = np.empty((M*nc, num_cand*nc))
        for c in range(nc):
            C_aV_candT[:,c*num_cand:(c+1)*num_cand] = C_a[:,c*M:(c+1)*M] @ v_Cand.T

        mc_vals = []
        for k in range(num_cand):
            inds_k_in_cand = [k + c*num_cand for c in range(nc)]
            B_k = np.diag(piks[k,:]) - np.outer(piks[k,:], piks[k,:])
            CVT_k = C_aV_candT[:, inds_k_in_cand]
            VCVT_k = np.empty((nc, nc))
            for c in range(nc):
                VCVT_k[c,:] = v_Cand[k,:][np.newaxis,:] @ C_aV_candT[c*M:(c+1)*M, inds_k_in_cand]

            Mk = np.linalg.inv(VCVT_k) + B_k
            Mk = (B_k @ np.linalg.inv(Mk)) - np.eye(nc)
            Mk = (Mk @ (B_k @ VCVT_k)) + np.eye(nc)
            Mk = CVT_k @ Mk
            Mkpi_k_yk_mat = Mk @ (np.tile(piks[k,:][:,np.newaxis], (1,nc))  - np.eye(nc))
            mc_vals_for_k = [np.linalg.norm(Mkpi_k_yk_mat[:,c]) for c in range(nc)]

            argmin_mcvals = np.argmin(mc_vals_for_k)
            argmax_piks = np.argmax(piks[k,:])

            if argmin_mcvals != argmax_piks:
                print("%d (index in Cand) did Not choose choice that we thought" % k)


            mc_vals.append(np.min(mc_vals_for_k))

        return np.array(mc_vals)


    else:
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
            # Multiclass GR
            if len(alpha.shape) > 1:
                raise NotImplementedError("Multiclass GR not implemented yet in the reduced case.")
            # Binary GR
            else:
                return np.array([np.absolute(uks[i] - sgn(uks[i]))/(gamma**2. + np.inner(v_Cand[i,:], C_a_vk[:,i]))
                               * np.linalg.norm(C_a_vk[:,i]) for i in range(v_Cand.shape[0])])
