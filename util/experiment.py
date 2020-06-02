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



def run_experiment_hf(w, v, L_un, labels, tau =0.1, gamma=0.1, num_to_query = 10,
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

    acc_class = False
    if acc_classifier_name is not "hf":
        acc_class = True
        acc_classifier = Classifier(acc_classifier_name, gamma, tau, v=v, w=w)

    model_classifier = Classifier_HF(tau, L_un)
    np.random.seed(seed)
    labeled =  list(np.random.choice(np.where(labels == -1)[0],
                        size=n_start//2, replace=False))
    labeled += list(np.random.choice(np.where(labels ==  1)[0],
                        size=n_start//2, replace=False))
    unlabeled = list(filter(lambda x: x not in labeled, range(len(labels))))



    # Harmonic Function labels in {0,1}
    labels_hf = copy.deepcopy(labels)
    labels_hf[labels_hf == -1.] = 0.
    n_hf = len(labels) - n_start

    # Harmonic Functions
    C = model_classifier.get_C(labeled)
    m = model_classifier.get_m(labeled, labels_hf[labeled])

    # Instantiate accuracy
    acc = []
    # Accuracy Model MAP estimator
    if acc_class:
        acc_m = acc_classifier.get_m(labeled, labels[labeled])
        acc.append(get_acc(acc_m, labels, unlabeled = unlabeled)[1])
    else:
        acc.append(get_acc(2*m - 1., labels[unlabeled], unlabeled = None)[1])


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
        if acc_class:
            acc_m = acc_classifier.get_m(labeled, labels[labeled]) # accuracy classifier is in +1, -1 classification
            acc.append(get_acc(acc_m, labels, unlabeled = unlabeled)[1])
        else:
            acc.append(get_acc(2*m-1., labels[unlabeled], unlabeled = None)[1])

        n_hf -= 1
    return acc, labeled




def test(w, v, labels, gamma, tau, n_eig,
         num_to_query, n_start, seed, filename, acqs):
    print(filename)
    if not os.path.exists(filename):
        os.makedirs(filename)
    try:
        acc = dict(np.load(filename + "acc.npz"))
        labeled = dict(np.load(filename + "labeled.npz"))
    except:
        acc = {}
        labeled = {}


    # IDEA: Save parameters in dictionary in this directory? Check the parameters before the test, in
    # case of running different acquisition functions, with different parameters than the rest
    # of the tests in the current directory?

    for acq in acqs:
        print(acq)
        if acq in acc and acq in labeled:
            print("Found existing results")
            continue
        if acq in ("modelchange_gr", "vopt_gr", "mbr_gr", "sopt_gr"):
            acc[acq], labeled[acq] = run_experiment(w, v, labels,
                        tau = tau, gamma = gamma,
                        n_eig = n_eig,
                        num_to_query = num_to_query,
                        n_start  = n_start, seed = seed,
                        exact_update = True,
                        acc_classifier_name="probit2", model_classifier_name="gr", acquisition=acq)
        elif acq in ("modelchange_p2", "vopt_p2", "mbr_p2", "vopt_new_p2"):
            acc[acq], labeled[acq] = run_experiment(w, v, labels,
                        tau = tau, gamma = gamma, n_eig = n_eig,
                        num_to_query = num_to_query,
                        n_start  = n_start, seed = seed,
                        exact_update = True,
                        acc_classifier_name="probit2", model_classifier_name="probit2",
                        acquisition=acq)
        elif acq in ("modelchange_p", "vopt_p", "mbr_p", "vopt_new_p"):
            acc[acq], labeled[acq] = run_experiment(w, v, labels,
                        tau = tau, gamma = gamma, n_eig = n_eig,
                        num_to_query = num_to_query,
                        n_start  = n_start, seed = seed,
                        exact_update = True,
                        acc_classifier_name="probit2", model_classifier_name="probit",
                        acquisition=acq)
        elif acq in ("modelchange_pNA", "vopt_pNA", "mbr_pNA"):
            acc[acq], labeled[acq] = run_experiment(w, v, labels,
                        tau = tau, gamma = gamma, n_eig = n_eig,
                        num_to_query = num_to_query,
                        n_start  = n_start, seed = seed,
                        exact_update = False,
                        acc_classifier_name="probit2", model_classifier_name="probit",
                        acquisition=acq[:-2])
        elif acq in ("modelchange_p2NA", "vopt_p2NA", "mbr_p2NA"):
            acc[acq], labeled[acq] = run_experiment(w, v, labels,
                        tau = tau, gamma = gamma, n_eig = n_eig,
                        num_to_query = num_to_query,
                        n_start  = n_start, seed = seed,
                        exact_update = False,
                        acc_classifier_name="probit2", model_classifier_name="probit2",
                        acquisition=acq[:-2])
        elif acq == "random":
            acc[acq], labeled[acq] = run_experiment(w, v, labels,
                        tau = tau, gamma = gamma, n_eig = n_eig,
                        num_to_query = num_to_query,
                        n_start  = n_start, seed = seed,
                        exact_update = False,
                        acc_classifier_name="probit2", model_classifier_name="probit2",
                        acquisition=acq)

        np.savez(filename + "acc.npz", **acc)
        np.savez(filename + "labeled.npz", **labeled)
    return acc, labeled



def test_hf(w, v, L_un, labels, gamma, tau, num_to_query, n_start, seed, filename, acqs):
    '''
        IDEA: Should merge this function into test function above... But worried about messing up previous
        code with the addition of unnormalized graph Laplacian L_un

        L_un is the unnormalized graph Laplacian, for use in the HF model.
    '''
    print(filename)
    if not os.path.exists(filename):
        os.makedirs(filename)
    try:
        print("Found data from the other test function (i.e. GR or Probit models), adding the results of  \
        these harmonic function tests to the .npz files found")
        acc = dict(np.load(filename + "acc.npz"))
        labeled = dict(np.load(filename + "labeled.npz"))
    except:
        acc = {}
        labeled = {}

    for acq in [a for a in acqs if a[-2:] == "hf"]:
        print(acq)
        if acq in acc and acq in labeled:
            print("Found existing results")
            continue
        if acq in ("vopt_hf", "sopt_hf", "mbr_hf"):
            acc[acq], labeled[acq] = run_experiment_hf(w, v, L_un, labels,
                        tau = tau, gamma = gamma,
                        num_to_query = num_to_query,
                        n_start  = n_start, seed = seed,
                        acc_classifier_name="hf", acquisition=acq)

        np.savez(filename + "acc.npz", **acc)
        np.savez(filename + "labeled.npz", **labeled)
    return acc, labeled



def acc_under_diff_model(labeled, labels, n_start=10, model_name='probit2', tau=0.1, gamma=0.1, w=None, v=None):
    N = len(labels)
    l_curr = list(labeled[:n_start])
    ul_curr = list(filter(lambda x: x not in l_curr, range(N)))
    classifier = Classifier(model_name, gamma, tau, v, w)
    m = classifier.get_m(l_curr, labels[l_curr])

    acc = []
    acc.append(get_acc(m, labels, unlabeled=ul_curr)[1])
    for k in labeled[n_start:]:
        l_curr += [k]
        ul_curr = list(filter(lambda x: x not in l_curr, range(N)))

        m = classifier.get_m(l_curr, labels[l_curr])
        acc.append(get_acc(m, labels, unlabeled=ul_curr)[1])

    return acc

def acc_under_hf_model(labeled, labels, n_start=10, tau=0.1, L_un=None):
    N = len(labels)
    l_curr = list(labeled[:n_start])
    ul_curr = list(filter(lambda x: x not in l_curr, range(N)))
    classifier = Classifier_HF(tau, L_un)
    labels_hf = labels.copy()
    labels_hf[labels_hf == -1] = 0

    m = classifier.get_m(l_curr, labels_hf[l_curr])
    C = classifier.get_C(l_curr)
    acc = []
    acc.append(get_acc(2*m - 1., labels[ul_curr])[1])
    n_hf = N - n_start

    for k in labeled[n_start:]:
        k_hf = ul_curr.index(k)
        l_curr += [k]

        ul_curr = list(filter(lambda x: x not in l_curr, range(N)))
        unl_hf = list(filter(lambda x: x != k_hf, range(n_hf)))
        m = m + (labels_hf[k] - m[k_hf])*C[k_hf,:]/C[k_hf,k_hf]
        m = m[unl_hf]
        C -= np.outer(C[k_hf,:], C[k_hf,:])/C[k_hf,k_hf]
        C = C[np.ix_(unl_hf, unl_hf)]
        acc.append(get_acc(2*m-1., labels[ul_curr])[1])

        n_hf -= 1

    return acc
