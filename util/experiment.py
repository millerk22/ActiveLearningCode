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
                   acc_classifier_name="probit",
                   model_classifier_name = "probit",
                   acquisition="modelchange_p"):

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

    #d = (tau ** (2.)) * ((w + tau**2.) ** (-1.))
    #Ct = v @ sp.sparse.diags(d, format='csr') @ v.T
    #Ct_inv = v @ sp.sparse.diags(1./d, format='csr') @ v.T
    acc_classifier = Classifier(acc_classifier_name, gamma, tau, v=v, w=w)
    model_classifier = Classifier(model_classifier_name, gamma, tau, v=v, w=w)
    np.random.seed(seed)
    labeled_orig =  list(np.random.choice(np.where(labels == -1)[0],
                        size=n_start//2, replace=False))
    labeled_orig += list(np.random.choice(np.where(labels ==  1)[0],
                        size=n_start//2, replace=False))

    m = model_classifier.get_m(labeled_orig, labels[labeled_orig])
    C_orig = model_classifier.get_C(labeled_orig, labels[labeled_orig], m)
    acc_m = acc_classifier.get_m(labeled_orig, labels[labeled_orig])

    C = copy.deepcopy(C_orig)    # get the original data, before AL choices
    labeled = copy.deepcopy(labeled_orig)

    acc = []
    acc.append(get_acc(acc_m, labels)[1])

    for i in range(num_to_query):
        print("{}/{}".format(i+1, num_to_query))
        # Calculate V-Opt criterion for unlabeled points
        unlabeled = list(filter(lambda x: x not in labeled, range(len(labels))))
        k = get_k(C, unlabeled, gamma, acquisition, m = m, y = labels[labeled])
        labeled += [k]
        if exact_update == True:
            m = model_classifier.get_m(labeled, labels[labeled])
            C = model_classifier.get_C(labeled, labels[labeled], m)
        else:
            a = labels[k]
            m -= jac_calc2(m[k], a, gamma) / (1. + C[k,k] * hess_calc2(m[k], a,
                 gamma ))*C[k,:]
            C = model_classifier.get_C(labeled, labels[labeled], m)
            #ck = C[k,:]
            #ckk = ck[k]
            #C -= (1./(gamma2 + ckk)) * np.outer(ck,ck)
        acc_m = acc_classifier.get_m(labeled, labels[labeled])
        acc.append(get_acc(acc_m, labels)[1])
    return acc
