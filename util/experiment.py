import scipy as sp
import scipy.sparse
import scipy.linalg
import numpy as np
import copy
from .al_util import *
from .acquisition import *

def run_experiment(w, v, labels, tau = 0.1, gamma = 0.1, n_eig = None,
                   num_to_query = 10,
                   n_start  = 10, seed = 42,
                   exact_update = True,
                   classifier="probit", acquisition="modelchange_p"):
    
    """
    labels should be binary {-1, 1}
    Classifier:
        probit-st:      probit with spectral truncation
        probit:         probit without spectral truncation
        probit2-st:     probit with logistic likelihood with spectral truncation
                        (not implemented)
        probit2:        probit with logistic likelihood without spectral        
                        truncation
    """

    if n_eig:
        w = w[:n_eig]
        v = v[:, :n_eig]

    d = (tau ** (2.)) * ((w + tau**2.) ** (-1.))
    Ct = v @ sp.sparse.diags(d, format='csr') @ v.T
    Ct_inv = v @ sp.sparse.diags(1./d, format='csr') @ v.T

    np.random.seed(seed)
    labeled_orig =  list(np.random.choice(np.where(labels == -1)[0],
                        size=n_start//2, replace=False))
    labeled_orig += list(np.random.choice(np.where(labels ==  1)[0],
                        size=n_start//2, replace=False))

    def get_m(Z, y):
        if classifier == "probit":
            return probit_map_dr(Z, y, gamma, Ct)
        elif classifier == "probit-st":
            return probit_map_st(Z, y, gamma, 1./d, v)
        elif classifier == "probit2":
            return probit_map_dr2(Z, y, gamma, Ct)
        else:
            pass

    def get_C(Z, y, m):
        if classifier in ["probit", "probit-st"]:
            if len(y) > n_eig:
                return Hess_inv_st(m, y, Z, 1./d, v, gamma)
            else:
                return Hess_inv(m, y, Z, gamma, Ct)
        elif classifier in ["probit2", "probit2-st"]:
            return Hess2_inv(m, y, Z, gamma, Ct)
        else: 
            pass
        
        

    u_map = get_m(labeled_orig, labels[labeled_orig])
    C_orig = get_C(labeled_orig, labels[labeled_orig], u_map)

    C = copy.deepcopy(C_orig)    # get the original data, before AL choices
    labeled = copy.deepcopy(labeled_orig)
    m = copy.deepcopy(u_map)

    acc = []
    acc.append(get_acc(m, labels)[1])

    for i in range(num_to_query):
        print("{}/{}".format(i+1, num_to_query))
        # Calculate V-Opt criterion for unlabeled points
        unlabeled = list(filter(lambda x: x not in labeled, range(len(labels))))
        if acquisition == "modelchange_p":
            k = modelchange_p(C, unlabeled, gamma, m, probit_norm=True)
        elif acquisition == "vopt_p":
            k = vopt_p(C, unlabeled, gamma, m, probit_norm=True)
        elif acquisition == "mbr_p":
            k = mbr_p(C, unlabeled, gamma, m, probit_norm=True)
        else:
            pass

        labeled += [k]
        if exact_update == True:
            m = get_m(labeled, labels[labeled])
            C = get_C(labeled, labels[labeled], m)
        else:
            a = labels[k]
            m -= jac_calc2(m[k], a, gamma) / (1. + C[k,k] * hess_calc2(m[k], a, 
                 gamma ))*C[k,:] 
            C = get_C(labeled, labels[labeled], m)
            #ck = C[k,:]
            #ckk = ck[k]
            #C -= (1./(gamma2 + ckk)) * np.outer(ck,ck)
        acc.append(get_acc(m, labels)[1])
    return acc
