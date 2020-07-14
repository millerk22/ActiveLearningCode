import scipy as sp
import scipy.sparse
import scipy.linalg
import numpy as np
import copy
from .al_util import *
from .acquisition_batch import *
import time
import os
import os.path

def run_experiment_batch(L, v, w, labels, tau = 0.1, gamma = 0.1, n_eig= None,
                   num_to_query = 100, B = 5,
                   n_start  = 10, seed = 42,
                   exact_update = True,
                   acc_classifier_name=None,
                   model_classifier_name = "probit2",
                   acquisition="admm", X=None):

    """
    CURRENTLY have classifier as requiring the eigenvalues and eigenvectors of L.
    labels should be binary {-1, 1}
    Classifier:
        probit-st:      probit with spectral truncation
        probit:         probit without spectral truncation
        probit2-st:     probit with logistic likelihood with spectral truncation
                        (not implemented)
        probit2:        probit with logistic likelihood without spectral
                        truncation
    """
    if num_to_query % B != 0:
        print("batch size value B = %d does not divide num_to_query = %d" % (B, num_to_query))
        num_to_query -= (num_to_query % B)
        print("\tnum_to_query now is %d." % num_to_query)

    if n_eig:
        w = w[:n_eig]
        v = v[:, :n_eig]

    Lt = tau**(-2.)*(L + sp.sparse.diags(v.shape[0]*[tau**2.])) # ASSUMES L is sparse!

    model_classifier = Classifier(model_classifier_name, gamma, tau, v=v, w=w)
    if acc_classifier_name is None:
        acc_classifier = model_classifier
    else:
        acc_classifier = Classifier(acc_classifier_name, gamma, tau, v=v, w=w)
    np.random.seed(seed)

    labeled_orig =  list(np.random.choice(np.where(labels == -1)[0],
                        size=n_start//2, replace=False))
    labeled_orig += list(np.random.choice(np.where(labels ==  1)[0],
                        size=n_start//2, replace=False))

    m = model_classifier.get_m(labeled_orig, labels[labeled_orig]) # Current MAP estimator
    acc_m = acc_classifier.get_m(labeled_orig, labels[labeled_orig])
    if acquisition in ["modelchange_batch", "modelchange_batch_exact"]:
        C = model_classifier.get_C(labeled_orig, labels[labeled_orig], m)
    labeled = copy.deepcopy(labeled_orig)
    unlabeled = list(filter(lambda x: x not in labeled, range(len(labels))))


    acc = []
    acc.append(get_acc(acc_m, labels, unlabeled = unlabeled)[1])

    num_batches = num_to_query // B

    for i in range(num_batches):
        print("Batch {}/{}".format(i+1, num_batches))

        k_batch = get_k_batch(Lt, labels[labeled], labeled, B, acquisition, gamma=gamma, u0=m, X=X, labels=labels, C=C)
        labeled += k_batch

        #print(labeled)
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
        print("\n\n\n")
    return acc, labeled




#############NOT DONE YET....
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

    timing = {}
    # IDEA: Save parameters in dictionary in this directory? Check the parameters before the test, in
    # case of running different acquisition functions, with different parameters than the rest
    # of the tests in the current directory?

    for acq in acqs:
        print(acq)
        ts = time.time()
        if acq in acc and acq in labeled:
            print("Found existing results")
            continue
        if acq in ("modelchange_gr", "vopt_gr", "mbr_gr",
                   "sopt_gr", "uncertainty_gr"):
            acc[acq], labeled[acq] = run_experiment(w, v, labels,
                        tau = tau, gamma = gamma,
                        n_eig = n_eig,
                        num_to_query = num_to_query,
                        n_start  = n_start, seed = seed,
                        exact_update = True,
                        acc_classifier_name=None,
                        model_classifier_name="gr",
                        acquisition=acq)
        elif acq in ("modelchange_p2", "vopt_p2", "mbr_p2",
                     "vopt_new_p2", "uncertainty_p2"):
            acc[acq], labeled[acq] = run_experiment(w, v, labels,
                        tau = tau, gamma = gamma, n_eig = n_eig,
                        num_to_query = num_to_query,
                        n_start  = n_start, seed = seed,
                        exact_update = True,
                        acc_classifier_name=None, model_classifier_name="probit2",
                        acquisition=acq)
        elif acq in ("modelchange_p", "vopt_p", "mbr_p",
                     "vopt_new_p", "uncertainty_p"):
            acc[acq], labeled[acq] = run_experiment(w, v, labels,
                        tau = tau, gamma = gamma, n_eig = n_eig,
                        num_to_query = num_to_query,
                        n_start  = n_start, seed = seed,
                        exact_update = True,
                        acc_classifier_name=None, model_classifier_name="probit",
                        acquisition=acq)
        elif acq in ("modelchange_pNA", "vopt_pNA", "mbr_pNA"):
            acc[acq], labeled[acq] = run_experiment(w, v, labels,
                        tau = tau, gamma = gamma, n_eig = n_eig,
                        num_to_query = num_to_query,
                        n_start  = n_start, seed = seed,
                        exact_update = False,
                        acc_classifier_name=None, model_classifier_name="probit",
                        acquisition=acq[:-2])
        elif acq in ("modelchange_p2NA", "vopt_p2NA", "mbr_p2NA"):
            acc[acq], labeled[acq] = run_experiment(w, v, labels,
                        tau = tau, gamma = gamma, n_eig = n_eig,
                        num_to_query = num_to_query,
                        n_start  = n_start, seed = seed,
                        exact_update = False,
                        acc_classifier_name=None, model_classifier_name="probit2",
                        acquisition=acq[:-2])
        else:
            pass
        timing[acq] = time.time() - ts
        print("{} takes {} seconds".format(acqs, timing[acq]))
        np.savez(filename + "acc.npz", **acc)
        np.savez(filename + "labeled.npz", **labeled)
        np.savez(filename + "timing.npz", **timing)
    return acc, labeled, timing







def test_random(w, v, L_un, labels, gamma, tau, num_to_query,
                n_start, seed, filename):
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
    print("doing random choices tests...")
    np.random.seed(seed)
    labeled_orig =  list(np.random.choice(np.where(labels == -1)[0],
                        size=n_start//2, replace=False))
    labeled_orig += list(np.random.choice(np.where(labels ==  1)[0],
                        size=n_start//2, replace=False))
    unlabeled = list(filter(lambda x: x not in labeled_orig,
                     range(len(labels))))
    if "random_p2" in acc:
        print("found existing results for random tests")
    else:
        labeled["random"] = labeled_orig + list(np.random.choice(unlabeled,
                                           size = num_to_query, replace = False))
        acc["random_p2"] = acc_under_diff_model(labeled["random"], labels,
                           n_start = n_start, model_name = 'probit2',
                           tau = tau, gamma = gamma, w = w, v = v)
        acc["random_gr"] = acc_under_diff_model(labeled["random"], labels,
                           n_start = n_start, model_name = 'gr',
                           tau = tau, gamma = gamma, w = w, v = v)
        acc["random_hf"] = acc_under_hf_model(labeled["random"], labels,
                           n_start = n_start, tau = tau,
                           L_un = L_un)

    np.savez(filename + "acc.npz", **acc)
    np.savez(filename + "labeled.npz", **labeled)
    return acc, labeled
