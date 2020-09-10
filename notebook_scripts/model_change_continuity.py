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

# +
from util.plot_util import *
from util.experiment import *
import os
import copy
import time
from sklearn.datasets import make_moons
from datasets.Graph_manager import Graph_manager
from datasets.data_loaders_mlflow import load_checkerboard
import matplotlib.pyplot as plt
import matplotlib
import scipy.sparse.linalg
import scipy.sparse.csgraph
import scipy.sparse
import scipy as sp
import numpy as np
# %cd D: \Dropbox\working-directory\ActiveLearningCode
# %load_ext autoreload

# Plotting configurations
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
SMALL_SIZE = 16
MEDIUM_SIZE = 16
BIGGER_SIZE = 16
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

latex_name = {"modelchange_p2": "Probit MC",
              "vopt_p2": "Probit VOpt",
              "modelchange_gr": "GR MC",
              "sopt_gr": "GR SOpt",
              "vopt_gr": "GR VOpt",
              "mbr_gr": "GR MBR",
              "mbr_p2": "Probit MBR",
              "vopt_p2NA": "Probit VOpt NA",
              "modelchange_p2NA":  "Probit MC NA",
              "mbr_p2NA": "Probit MBR NA",
              "mbr_hf": "HF MBR",
              "vopt_hf": "HF VOpt",
              "sopt_hf": "HF SOpt"}
# -

# # Two moon

# +
N = 1000
seed = 42
X, labels = make_moons(N, noise=0.15)
gm = Graph_manager()
graph_params = {
    'knn': 15,
    'Ltype': 'normed',
    'sigma': 3.0,
    'zp_k': 7,
    'n_eigs': None
}

labels[labels == 0] = -1
ts = time.time()
W = gm.compute_similarity_graph(X, graph_params['knn'], graph_params['sigma'])
te = time.time()
print("Constructing similarity graph takes %f seconds" % (te - ts))
ts = time.time()
L = gm.compute_laplacian(W, graph_params['Ltype'])
w, v = gm.compute_spectrum(L, n_eigs=None)
te = time.time()
print("Computing eigenvectors takes %f seconds" % (te - ts))
plt.plot(w)
# -

# # MNIST

# +
digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
num_points_train = [400] * 10
num_points_test = [0] * 10
seed = 42

data_params = {
    'digits': digits,
    'num_points_train': num_points_train,
    'num_points_test': num_points_test,
    'seed': seed
}

X_train, X_test, labels_train, labels_test = load_MNIST_split(data_params)
X = np.concatenate((X_train, X_test))
labels = np.concatenate((labels_train, labels_test))
gm = Graph_manager()
graph_params = {
    'knn': 15,
    'Ltype': 'normed',
    'sigma': 380 * 380,
    'zp_k': None,
    'n_eigs': None
}
labels_bin = copy.deepcopy(labels)  # make labels binary
labels_bin[labels % 2 == 0] = -1
labels_bin[labels % 2 == 1] = 1
labels = labels_bin
ts = time.time()
W = gm.compute_similarity_graph(X/255.0, graph_params['knn'],
                                graph_params['sigma'], graph_params["zp_k"])
te = time.time()
print("Constructing similarity graph takes %f seconds" % (te - ts))
ts = time.time()
L = gm.compute_laplacian(W, graph_params['Ltype'])
L_un = gm.compute_laplacian(W, "unnormed").toarray()
w, v = gm.compute_spectrum(L, n_eigs=None)
te = time.time()
print("Computing eigenvectors takes %f seconds" % (te - ts))
plt.plot(w)
# -

from util.al_util import jac_calc, hess_calc, Classifier

seeds = 42
num_to_query = 50
n_start = 10
n_eig = None
gamma, tau = 0.1, 0.1

# +
model_classifier = Classifier('probit', gamma, tau, v=v, w=w)
acc_classifier = model_classifier
labeled_orig = list(np.random.choice(np.where(labels == -1)[0],
                                     size=n_start//2, replace=False))
labeled_orig += list(np.random.choice(np.where(labels == 1)[0],
                                      size=n_start//2, replace=False))

m = model_classifier.get_m(labeled_orig, labels[labeled_orig])
C_orig = model_classifier.get_C(labeled_orig, labels[labeled_orig], m)
acc_m = acc_classifier.get_m(labeled_orig, labels[labeled_orig])
C = copy.deepcopy(C_orig)    # get the original data, before AL choices
labeled = copy.deepcopy(labeled_orig)
unlabeled = list(filter(lambda x: x not in labeled, range(len(labels))))
# -

mc = [min(np.absolute(jac_calc(m[k], -1, gamma)/(1. + C[k, k]*hess_calc(m[k], -1, gamma))),
          np.absolute(jac_calc(m[k],  1, gamma)/(1. + C[k, k]*hess_calc(m[k],  1, gamma))))
        * np.linalg.norm(C[k, :]) for k in unlabeled]

from scipy.spatial.distance import pdist, squareform

mc_distance = pdist(np.array(mc).reshape((-1, 1)))
x_distance = pdist(X[unlabeled])
v_distance = pdist(v[unlabeled, 1].reshape((-1, 1)))

# %matplotlib inline

plt.figure()
plt.xlabel('$|x_i - x_j|$')
plt.ylabel('$|M(i) - M(j)|$')
plt.plot(x_distance, mc_distance, '.')

plt.figure()
plt.xlabel('$|v^{(1)}_i - v^{(1)}_j|$')
plt.ylabel('$|M(i) - M(j)|$')
plt.plot(v_distance, mc_distance, '.')

plt.figure()
i = 15
plt.xlabel('$|v^{(1)}_i - v^{(1)}_j|$')
plt.ylabel('$|M(i) - M(j)|$')
plt.plot(squareform(v_distance)[i,:], squareform(mc_distance)[i,:], '.')

plt.figure()
i = 15
plt.xlabel('$|x_i - x_j|$')
plt.ylabel('$|M(i) - M(j)|$')
plt.plot(squareform(x_distance)[i,:], squareform(mc_distance)[i,:], '.')
