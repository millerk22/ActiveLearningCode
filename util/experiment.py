import scipy as sp
import scipy.sparse
import scipy.linalg
import numpy as np
import copy
from .al_util import *
from .acquisition import *
import time
import os
import os.path

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

    unlabeled = list(filter(lambda x: x not in labeled, range(len(labels))))
    acc = []
    acc.append(get_acc(acc_m, labels, unlabeled = unlabeled)[1])

    num_batch = num_to_query // 4
    for i in range(num_to_query):
        if (i+1) % num_batch == 0:
            print("\t{}/{}".format(i+1, num_to_query))
        # Calculate V-Opt criterion for unlabeled points
        #k = get_k(C, unlabeled, acquisition, gamma=gamma, m = m, y = labels[labeled])
        k = get_k(C, unlabeled, acquisition, gamma=gamma, m = m) # we don't need y for any acquisition function now
        labeled += [k]
        unlabeled = list(filter(lambda x: x not in labeled, range(len(labels))))
        if exact_update == True:
            m = model_classifier.get_m(labeled, labels[labeled])
            C = model_classifier.get_C(labeled, labels[labeled], m)
        else:
            a = labels[k]
            m -= jac_calc2(m[k], a, gamma) / (1. + C[k,k] * hess_calc2(m[k], a,
                 gamma ))*C[k,:]
            C = C - hess_calc2(m[k], a, gamma)/(1. + C[k,k]*hess_calc2(m[k], a, gamma))*np.outer(C[k,:], C[k,:])

        acc_m = acc_classifier.get_m(labeled, labels[labeled])
        acc.append(get_acc(acc_m, labels, unlabeled = unlabeled)[1])
    return acc, labeled



def run_experiment_hf(w, v, L, labels, tau =0.1, gamma=0.1, num_to_query = 10,
                   n_start  = 10, seed = 42, acc_classifier_name = "probit2",
                   acquisition="modelchange_hf"):

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


    acc_classifier = Classifier(acc_classifier_name, gamma, tau, v=v, w=w)
    model_classifier = Classifier_HF(tau, L)
    np.random.seed(seed)
    labeled =  list(np.random.choice(np.where(labels == -1)[0],
                        size=n_start//2, replace=False))
    labeled += list(np.random.choice(np.where(labels ==  1)[0],
                        size=n_start//2, replace=False))
    unlabeled = list(filter(lambda x: x not in labeled, range(len(labels))))

    # Accuracy Model MAP estimator
    acc_m = acc_classifier.get_m(labeled, labels[labeled])

    # Harmonic Function labels in {0,1}
    labels_hf = copy.deepcopy(labels)
    labels_hf[labels_hf == -1.] = 0.
    n_hf = len(labels) - n_start

    # Harmonic Functions
    C = model_classifier.get_C(labeled)
    m = model_classifier.get_m(labeled, labels_hf[labeled])

    # Instantiate accuracy
    acc = []
    acc.append(get_acc(acc_m, labels, unlabeled = unlabeled)[1])

    num_batch = num_to_query // 4
    for i in range(num_to_query):
        if (i+1) % num_batch == 0:
            print("\t{}/{}".format(i+1, num_to_query))

        # Calculate AL criterion for unlabeled points
        #k = get_k(C, unlabeled, acquisition=acquisition, m=m, y=labels[labeled])
        k = get_k(C, unlabeled, acquisition=acquisition, m=m) # we don't need y for any of the acquisition functions now
        labeled += [unlabeled[k]]


        # Harmonic Function updates
        unl_hf = list(filter(lambda x: x != k, range(n_hf)))
        m = m + (labels_hf[unlabeled[k]] - m[k])*C[k,:]/C[k,k]
        m = m[unl_hf]
        C -= np.outer(C[k,:], C[k,:])/C[k,k]
        C = C[np.ix_(unl_hf, unl_hf)]

        unlabeled = list(filter(lambda x: x not in labeled, range(len(labels))))
        acc_m = acc_classifier.get_m(labeled, labels[labeled]) # accuracy classifier is in +1, -1 classification
        acc.append(get_acc(acc_m, labels, unlabeled = unlabeled)[1])

        n_hf -= 1
    return acc, labeled


    

def test(w, v, labels, gamma, tau, n_eig,
         num_to_query, n_start, seed, filename, acqs):
    print(filename)
    if not os.path.exists(filename):
        os.mkdir(filename)
    try:
        acc = dict(np.load(filename + "acc.npz"))
        labeled = dict(np.load(filename + "labeled.npz"))
    except:
        acc = {}
        labeled = {}

    for acqs in acqs:
        print(acqs)
        if acqs in acc and acqs in labeled:
            print("Found exiting results")
            continue
        if acqs in ("modelchange_gr", "vopt_gr", "mbr_gr"):
            acc[acqs], labeled[acqs] = run_experiment(w, v, labels,
                        tau = tau, gamma = gamma,
                        n_eig = n_eig,
                        num_to_query = num_to_query,
                        n_start  = n_start, seed = seed,
                        exact_update = True,
                        acc_classifier_name="probit2", model_classifier_name="gr", acquisition=acqs)
        elif acqs in ("modelchange_p2", "vopt_p2", "mbr_p2", "vopt_new_p2"):
            acc[acqs], labeled[acqs] = run_experiment(w, v, labels,
                        tau = tau, gamma = gamma, n_eig = n_eig,
                        num_to_query = num_to_query,
                        n_start  = n_start, seed = seed,
                        exact_update = True,
                        acc_classifier_name="probit2", model_classifier_name="probit2",
                        acquisition=acqs)
        elif acqs in ("modelchange_p", "vopt_p", "mbr_p", "vopt_new_p"):
            acc[acqs], labeled[acqs] = run_experiment(w, v, labels,
                        tau = tau, gamma = gamma, n_eig = n_eig,
                        num_to_query = num_to_query,
                        n_start  = n_start, seed = seed,
                        exact_update = True,
                        acc_classifier_name="probit2", model_classifier_name="probit",
                        acquisition=acqs)
        elif acqs in ("modelchange_pNA", "vopt_pNA"):
            acc[acqs], labeled[acqs] = run_experiment(w, v, labels,
                        tau = tau, gamma = gamma, n_eig = n_eig,
                        num_to_query = num_to_query,
                        n_start  = n_start, seed = seed,
                        exact_update = False,
                        acc_classifier_name="probit2", model_classifier_name="probit",
                        acquisition=acqs)
        elif acqs in ("modelchange_p2NA", "vopt_p2NA"):
            acc[acqs], labeled[acqs] = run_experiment(w, v, labels,
                        tau = tau, gamma = gamma, n_eig = n_eig,
                        num_to_query = num_to_query,
                        n_start  = n_start, seed = seed,
                        exact_update = False,
                        acc_classifier_name="probit2", model_classifier_name="probit2",
                        acquisition=acqs)
        np.savez(filename + "acc.npz", **acc)
        np.savez(filename + "labeled.npz", **labeled)
    return acc, labeled
