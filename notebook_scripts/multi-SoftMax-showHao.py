# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %cd ~/Desktop/ActiveLearningCode/
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from datasets.data_loaders_mlflow import load_checkerboard
from util.Graph_manager import Graph_manager
import time
import copy
from util.gbssl import *
from util.activelearner import *
# %load_ext autoreload

from util.dataloaders import *
from sklearn.preprocessing import OneHotEncoder


# +
def softmax_map_alpha(Z_, y, d, v, gamma=1., newton=False):
    ''' Calculate the SoftMax MAP estimator
    
    Z_ : list of labeled nodes
    y : (len(Z_), nc) numpy array of onehot vectors as rows
    d : numpy array of size (M,) eigenvalues 
    v : numpy array of size (N, M) eigenvectors
    gamma : parameter in the softmax loss, e^{u_k/gamma}
    newton : bool, whether or not to use fprime for a second order optimization method for solving
    
    output : alpha, Hessian(alpha)
    '''
    
    y = np.array(y)
    num_labeled, nc = y.shape
    N,M = v.shape
    vZ_ = v[Z_,:]/gamma
    
    def f(x):
        pi_Z = np.exp(vZ_ @ x.reshape(nc, M).T)
        pi_Z /= np.sum(pi_Z, axis=1)[:, np.newaxis]
        vec = np.empty(M*nc)
        for c in range(nc):
            vec[c*M:(c+1)*M] = vZ_.T @ (pi_Z[:,c] - y[:,c])
        return np.tile(d, (1,nc)).flatten() * x + vec

    def fprime(x):
        pi_Z = np.exp(vZ_ @ x.reshape(nc, M).T)
        pi_Z /= np.sum(pi_Z, axis=1)[:, np.newaxis]
        H = np.empty((M*nc, M*nc))
        for c in range(nc):
            for m in range(c,nc):
                Bmc = 1.*(m == c)*pi_Z[:,c] - pi_Z[:,c]*pi_Z[:,m]
                H[c*M:(c+1)*M, m*M:(m+1)*M] = (vZ_.T * Bmc) @ vZ_
                if m != c:
                    H[m*M:(m+1)*M, c*M:(c+1)*M] = H[c*M:(c+1)*M, m*M:(m+1)*M]

        H.ravel()[::(M*nc+1)] += np.tile(d, (1, nc)).flatten()
        return H

    x0 = np.random.randn(N,nc)
    x0[labeled,:] = y
    x0 = (v.T @ x0).T.flatten()
    if newton:
        #print("Second Order")
        res = root(f, x0, jac=fprime, tol=1e-9)
    else:
        #print("First Order")
        res = root(f, x0, tol=1e-10, method='krylov')
    if not res.success:
        print(res.success)
        print(np.linalg.norm(f(res.x)))
        print(res.message)
        return res.x, fprime(res.x)
        
    return res.x, fprime(res.x)


def Hess_calc_alpha_multi(Z_, y, d, v, alpha, gamma=1.):
    ''' Hessian Calculation of Softmax, in the eigenvector space (M*n_c x M*n_c)
    Z_ : list of labeled nodes
    y : (len(Z_), nc) numpy array of onehot vectors as rows
    d : numpy array of size (M,) eigenvalues 
    v : numpy array of size (N, M) eigenvectors
    gamma : parameter in the softmax loss, e^{u_k/gamma}
    
    
    output:
    Hessian(alpha)
    
    '''
    vZ_ = v[Z_, :]/gamma
    num_labeled, nc = y.shape
    M = v.shape[1]
    pi_Z = np.exp(vZ_ @ alpha.reshape(nc, M).T)
    pi_Z /= np.sum(pi_Z, axis=1)[:, np.newaxis]
    H = np.empty((M*nc, M*nc))
    for c in range(nc):
        for m in range(c,nc):
            Bmc = 1.*(m == c)*pi_Z[:,c] - pi_Z[:,c]*pi_Z[:,m]
            H[c*M:(c+1)*M, m*M:(m+1)*M] = (vZ_.T * Bmc) @ vZ_
            if m != c:
                H[m*M:(m+1)*M, c*M:(c+1)*M] = H[c*M:(c+1)*M, m*M:(m+1)*M]
    
    H.ravel()[::(M*nc+1)] += np.tile(d, (1, nc)).flatten()
    # H = Lambda +  V_L^T (D_L - Pi_L Pi_L^T) V_L
    
    return H


def get_C_a_dumb(Z_, y, d, v, alpha, gamma=1.):
    ''' simple function that calls the Hessian calculation for Softmax in eigenvector space and then just inverts it
    '''
    
    return np.linalg.inv(Hess_calc_alpha_multi(Z_, y, d, v, alpha, gamma=gamma))



# +
def test_mc_dumb(C_a, alpha, v_Cand, Cand, uks, gamma=1., H_a=None):
    ''' Inefficient straightforward calculation of Softmax model change
        * for comparison with the methods hereafter *
    C_a : (M*n_c x M*n_c) numpy array, current Covariance matrix in eigenvector space
    alpha : (M*n_c, ) numpy array, current Softmax MAP in eigenvector space
    v_Cand: (|Cand|, M) numpy array, rows corresponding to Candidate nodes
    Cand: list of length |Cand|, corresponding to the indices of the candidate nodes in the original index ordering
    uks : (|Cand|, n_c) numpy array, equals v_Cand @ alpha.reshape(n_c, M).T
    gamma : softmax parameter 
    H_a : can pass in the eigenvector Hessian 
    
    output:
        acq_vals : Softmax model change acquisition function on each of the Candidate nodes in Cand
    '''
    piks = np.exp(uks/gamma)
    piks /= np.sum(piks, axis=1)[:,np.newaxis]

    num_cand, M = v_Cand.shape
    nc = alpha.shape[1]
    
    if H_a is None:
        print("Ha is None")
        C_a_inv = np.linalg.inv(C_a)
    else:
        C_a_inv = H_a.copy()
    
    mc_vals = []
    for k in range(num_cand):
        B_k = np.diag(piks[k,:]) - np.outer(piks[k,:], piks[k,:])
        Vk = np.kron(np.eye(nc), v_Cand[k,:][np.newaxis, :])/gamma
        Hess_inv_k = np.linalg.inv(C_a_inv + Vk.T @ B_k @ Vk)
        Mk = Hess_inv_k @ Vk.T
        Mkpi_k_yk_mat = Mk @ (np.tile(piks[k,:][:,np.newaxis], (1,nc))  - np.eye(nc))
        mc_vals_for_k = [np.linalg.norm(Mkpi_k_yk_mat[:,c]) for c in range(nc)]
        mc_vals.append(np.min(mc_vals_for_k))
        
        if np.argmin(mc_vals_for_k) != np.argmax(piks[k,:]):
            print("%d did Not choose choice that we thought" % Cand[k])
        
    return np.array(mc_vals)


def test_mc(C_a, alpha, v_Cand, Cand, uks, gamma=1.):
    ''' More efficient calculation of Softmax model change
        
    C_a : (M*n_c x M*n_c) numpy array, current Covariance matrix in eigenvector space
    alpha : (M*n_c, ) numpy array, current Softmax MAP in eigenvector space
    v_Cand: (|Cand|, M) numpy array, rows corresponding to Candidate nodes
    Cand: list of length |Cand|, corresponding to the indices of the candidate nodes in the original index ordering
    uks : (|Cand|, n_c) numpy array, equals v_Cand @ alpha.reshape(n_c, M).T
    gamma : softmax parameter 
    H_a : can pass in the eigenvector Hessian 
    
    output:
        acq_vals : Softmax model change acquisition function on each of the Candidate nodes in Cand
    '''
    v_Cand /= gamma
    piks = np.exp(uks/gamma)
    piks /= np.sum(piks, axis=1)[:,np.newaxis]

    num_cand, M = v_Cand.shape
    nc = alpha.shape[1]
        
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
            print("%d did Not choose choice that we thought" % Cand[k])

        
        mc_vals.append(np.min(mc_vals_for_k))
    
    return np.array(mc_vals)


def test_mc_fast(C_a, alpha, v_Cand, Cand, uks, gamma=1.):
    ''' Faster calculation of Softmax model change
        * Doesn't calculate value for each class, JUST the class that current MAP estimator alpha would guess*
    C_a : (M*n_c x M*n_c) numpy array, current Covariance matrix in eigenvector space
    alpha : (M*n_c, ) numpy array, current Softmax MAP in eigenvector space
    v_Cand: (|Cand|, M) numpy array, rows corresponding to Candidate nodes
    Cand: list of length |Cand|, corresponding to the indices of the candidate nodes in the original index ordering
    uks : (|Cand|, n_c) numpy array, equals v_Cand @ alpha.reshape(n_c, M).T
    gamma : softmax parameter 
    H_a : can pass in the eigenvector Hessian 
    
    output:
        acq_vals : Softmax model change acquisition function on each of the Candidate nodes in Cand
    '''
    v_Cand /= gamma
    piks = np.exp(uks/gamma)
    piks /= np.sum(piks, axis=1)[:,np.newaxis]

    num_cand, M = v_Cand.shape
    nc = alpha.shape[1]
        
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
        Mkpi_k_yk = Mk @ (piks[k,:] - np.array([1. if l == np.argmax(piks[k,:]) else 0. for l in range(nc)]))
        mc_vals.append(np.linalg.norm(Mkpi_k_yk))
        
    return np.array(mc_vals)


# -

# ## Checker-3 Data

X = np.random.rand(1000,2)
labels = []
for x in X:
    i, j = 0,0
    if 0.33333 <= x[0] and x[0] < 0.66666:
        i = 1
    elif 0.66666 <= x[0]:
        i = 2
    
    if 0.33333 <= x[1] and x[1] < 0.66666:
        j = 1
    elif 0.66666 <= x[1]:
        j = 2
    
    labels.append(3*j + i)
labels = np.array(labels)


print(np.unique(labels))

plt.scatter(X[:,0], X[:,1], c=labels)
plt.show()

labels[labels == 4] = 0
labels[labels == 8] = 0
labels[labels == 5] = 1
labels[labels == 6] = 1
labels[labels == 3] = 2
labels[labels == 7] = 2

print(np.unique(labels))

plt.scatter(X[:,0], X[:,1], c=labels)
plt.show()

# +
labeled_orig = []
for c in np.unique(labels):
    class_c = np.where(labels == c)[0]
    labeled_orig += list(np.random.choice(class_c,len(class_c)//100, replace=False))

enc = OneHotEncoder()
enc.fit(labels.reshape((-1, 1)))
onehot_labels = enc.transform(labels.reshape((-1, 1))).todense()

# +
gm = Graph_manager()
graph_params = {
    'knn' :10,
    'sigma' : 30.,
    'Ltype' : 'normed',
    'n_eigs' :200,
    'zp_k' : 7
}

w, v= gm.from_features(X, graph_params)



# -

# ## Model Instantiation

tau = 0.05
d = tau**(-2.)*(w + tau**2.)
#d = (w + tau**2.)

# +
# Copy the labeling data for use in Active learning loop 
labeled = copy.deepcopy(labeled_orig)
N = len(labels)
unlabeled = list(filter(lambda x: x not in labeled, range(N)))
y = np.array(onehot_labels[labeled])
print(len(labeled))
print(len(unlabeled))


# Set parameters for the Softmax model
gamma = .01   # softmax scalar parameter
M, nc = 200, len(np.unique(labels))  # number of eigenvalues and number of classes
vM = v[:,:M] # select the eigenvectors
dM = d[:M] # select the eigenvalues

# Calculate the initial Softmax MAP estimator for alpha
alpha, H_a = softmax_map_alpha(labeled, y, dM, vM, gamma=gamma, newton=True)

# Calculate the model in the full space and the Covariance in the eigenvectors space
uhat = vM @ (alpha.reshape(nc,M).T)
Ca = get_C_a_dumb(labeled, y, dM, vM, alpha, gamma=gamma)
uhat_classes = np.argmax(uhat, axis=1)
print(np.allclose(Ca, np.linalg.inv(H_a)))
# -

plt.figure(figsize=(4,4))
plt.scatter(X[:,0], X[:,1], c=np.argmax(uhat, axis=1))
plt.scatter(X[labeled, 0], X[labeled, 1], c='r', marker='s')
plt.show()

# %time acq_vals_dumb = test_mc_dumb(Ca, alpha.reshape(nc, M).T, vM[unlabeled,:], Cand=unlabeled, uks=uhat[unlabeled,:], gamma=gamma)

# %time acq_vals = test_mc(Ca, alpha.reshape(nc, M).T, vM[unlabeled,:], Cand=unlabeled, uks=uhat[unlabeled,:], gamma=gamma)

# %time acq_vals_fast = test_mc_fast(Ca, alpha.reshape(nc, M).T, vM[unlabeled,:], Cand=unlabeled, uks=uhat[unlabeled,:], gamma=gamma)

# ## Active Learning Loop

# +
num_batches = 1


acc = [get_acc_multi(uhat_classes, labels, unlabeled = unlabeled[:])[1]]
for _ in range(num_batches):
    print("Iteration %d" % (_ + 1))
    # Calculate acquisition function values over the different possible functions
    acq_vals = test_mc(Ca, alpha.reshape(nc, M).T, vM[unlabeled,:], Cand=unlabeled, uks=uhat[unlabeled,:], gamma=gamma)
    acq_vals_fast = test_mc_fast(Ca, alpha.reshape(nc, M).T, vM[unlabeled,:], Cand=unlabeled, uks=uhat[unlabeled,:], gamma=gamma)
    acq_vals_dumb = test_mc_dumb(Ca, alpha.reshape(nc, M).T, vM[unlabeled,:], Cand=unlabeled, uks=uhat[unlabeled,:], gamma=gamma, H_a=H_a)
    
    print("1 vs fast : " + str(np.allclose(acq_vals, acq_vals_fast)))
    print("1 vs dumb : " + str(np.allclose(acq_vals, acq_vals_dumb)))
    
    k = unlabeled[np.argmax(acq_vals)]
    
    # Plot the acquisition function
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.scatter(X[unlabeled, 0], X[unlabeled,1], c=acq_vals, marker='o')
    plt.scatter(X[labeled, 0], X[labeled, 1], c='r', marker='^')
    plt.scatter(X[k,0], X[k,1], c='r')
    plt.title("MC SoftMax")
    plt.subplot(1,2,2)
    plt.scatter(X[:,0], X[:,1], c=uhat_classes)
    plt.title("uhat")
    plt.show()
    
    
    # Query oracle for labels
    yk = np.array(onehot_labels[k,:])
    
    labeled += [k]
    unlabeled = list(filter(lambda x: x not in labeled, range(N)))
    y = np.vstack((y, yk.reshape(-1, nc)))
    
    
    # Retrain model
    alpha, H_a = softmax_map_alpha(labeled, y, dM, vM, gamma=gamma, newton=False)
    Ca = np.linalg.inv(H_a)
    uhat = vM @ alpha.reshape(nc,M).T
    uhat_classes = np.argmax(uhat, axis=1)
    
    acc.append(get_acc_multi(uhat_classes, labels, unlabeled = unlabeled[:])[1])


# -

plt.plot(range(len(acc)), acc)
plt.title("Accuracy of Softmax MC -- 3 Checker")
plt.show()
