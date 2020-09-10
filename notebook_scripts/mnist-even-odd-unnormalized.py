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

from util.experiment import run_experiment, test, test_hf, test_random
import copy
import time
from datasets.Graph_manager import Graph_manager
from datasets.data_loaders_mlflow import load_MNIST_split
import matplotlib.pyplot as plt
import matplotlib
import scipy.sparse.linalg
import scipy.sparse.csgraph
import scipy.sparse
import scipy as sp
import numpy as np
# %cd ~/Dropbox/working-directory/ActiveLearningCode/
# The results will be in "./results/mnist-even-odd/" folder
# Plotting configurations
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
markers = ['o', '*', 'v', '^', '<', '>', '8', 's', 'p', 'h']

# +
digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
num_points_train = [400] * 10
num_points_test = [0] * 10
seed = 42

data_params = {
    'digits' : digits,
    'num_points_train' : num_points_train,
    'num_points_test'  : num_points_test,
    'seed' : seed
}

X_train, X_test, labels_train, labels_test = load_MNIST_split(data_params)
X = np.concatenate((X_train, X_test))
labels = np.concatenate((labels_train, labels_test))
gm = Graph_manager()
graph_params = {
    'knn': 15,
    'Ltype': 'normed',
    'sigma': 380 * 380,
    'zp_k' : None,
    'n_eigs': None
}
labels_bin = copy.deepcopy(labels) # make labels binary
labels_bin[labels % 2 == 0] = -1
labels_bin[labels % 2 == 1] = 1
labels = labels_bin
ts = time.time()
W = gm.compute_similarity_graph(X/255.0, graph_params['knn'], 
                                graph_params['sigma'], graph_params["zp_k"])
te = time.time()
print("Constructing similarity graph takes %f seconds"% (te - ts))
ts = time.time()
L = gm.compute_laplacian(W, graph_params['Ltype'])
L_un = gm.compute_laplacian(W, "unnormed").toarray()
w, v  = gm.compute_spectrum(L, n_eigs=None)
te = time.time()
print("Computing eigenvectors takes %f seconds"% (te - ts))
plt.plot(w)
# -

# %load_ext autoreload
# %autoreload

# +
#v1: tau = 0.1, gamma = 0.1 N =  100, seed = 42,43,44
#v2: tau = 0.1, gamma = 0.1, neig = 1000, N = 100, seed = 42, 43, 44
#v3: tau = 0.1, gamma = 0.1, neig = None, N = 100, seed = 42, 43, 44, 
#    sigma = 1, knn =15, zpk = 7, unormalized
#v4: tau = 0.1, gamma = 0.1, neig = None, N = 100, seed = 42, 43, 44, 
#    sigma = 380 * 380, knn =10, zpk = none, unormalized
#v5: tau = 0.1, gamma = 0.01, neig = None, N = 100, seed = [42,59,34,25,1]
#    sigma = 380 * 380, knn =15, zpk = None, normalized
#v7: tau = 0.1, gamma = 0.01, neig = None, N = 100, seed = [42,59,34,25,1]
#    sigma = 380 * 380, knn =15, zpk = None, normalized
 
acqs = ["uncertainty_gr", "uncertainty_p2", "uncertainty_hf",
        "modelchange_gr", "vopt_gr", "sopt_gr", 
        "modelchange_p2", "vopt_p2", "modelchange_p2NA", "vopt_p2NA",
        "vopt_hf", "sopt_hf", 
        "mbr_hf", "mbr_p2NA", "mbr_gr", "mbr_p2"]
print(acqs)
filename = "./results/mnist-even-odd/"
version = 7
seeds = [42,59,34,25,1]
for e, seed in enumerate(seeds):
    print(e)
    test_random(w, v, L_un, labels, tau = 0.1, gamma = 0.01, num_to_query = 100,
                n_start = 10, seed = seed, filename = filename + "v{}-{}/".format(version, e))
    test_hf(w, v, L_un, labels, tau = 0.1, gamma = 0.01, num_to_query = 100, 
         n_start = 10, seed = seed, filename = filename + "v{}-{}/".format(version, e), acqs =  acqs)
    test(w, v, labels, tau = 0.1, gamma = 0.01, n_eig = None, num_to_query = 100, 
         n_start = 10, seed = seed, filename = filename + "v{}-{}/".format(version, e), acqs =  acqs)
# -

acqs = ["uncertainty_gr", "uncertainty_p2", "uncertainty_hf",
        "modelchange_gr", "vopt_gr", "sopt_gr", 
        "modelchange_p2", "vopt_p2", "modelchange_p2NA", "vopt_p2NA",
        "vopt_hf", "sopt_hf", 
        "mbr_hf", "mbr_p2NA", "mbr_gr"]
print(acqs)
filename = "./results/mnist-even-odd/"
version = 6
seeds = [42,59,34,25,1]
for e, seed in enumerate(seeds):
    print(e)
    ts = time.time()
    test_random(w, v, L_un, labels, tau = 0.1, gamma = 0.1, num_to_query = 100,
                n_start = 10, seed = seed, filename = filename + "v{}-{}/".format(version, e))
    test_hf(w, v, L_un, labels, tau = 0.1, gamma = 0.1, num_to_query = 100, 
         n_start = 10, seed = seed, filename = filename + "v{}-{}/".format(version, e), acqs =  acqs)
    test(w, v, labels, tau = 0.1, gamma = 0.1, n_eig = None, num_to_query = 100, 
         n_start = 10, seed = seed, filename = filename + "v{}-{}/".format(version, e), acqs =  acqs)
    
    te = time.time()
    print("Takes {} seconds".format(te-ts))

# +
from util.plot_util import *
from util.experiment import *
plt.rc('figure', figsize=(12,8))
# Plotting configurations
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
SMALL_SIZE = 20
MEDIUM_SIZE = 22
BIGGER_SIZE = 24
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels 
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('figure', figsize=(8,6))
plt.rc('legend', fontsize=20)    # legend fontsize
plt.rc('lines', markersize=8)
latex_name ={"modelchange_p2":"Probit MC","modelchange_p2NA" :  "Probit MC NA", "modelchange_gr":"GR MC",
            "vopt_p2":"Probit VOpt", "vopt_p2NA" : "Probit VOpt NA", "vopt_gr":"GR VOpt", "vopt_hf": "HF VOpt",
             "mbr_p2" : "Probit MBR", "mbr_p2NA" : "Probit MBR NA", "mbr_gr": "GR MBR", "mbr_hf":"HF MBR",
             "sopt_gr":"GR SOpt","sopt_hf":"HF SOpt",
            "random_gr":"GR Random","random":"Probit Random", "random_p2":"Probit Random","random_p":"Probit Random","random_hf":"HF Random", 
             "uncertainty_gr": "GR Uncertainty", "uncertainty_p2": "Probit Uncertainty", "uncertainty_hf" : "Probit Uncertainty"}


acqs_plot = ["modelchange_p2", "vopt_p2", "mbr_p2NA", "random_p2", "uncertainty_p2"]
ax = plot_acc_from_npz('results/mnist-even-odd/', "v5-", acqs_plot, err_bar= False, dataset_title='')
plt.savefig("mnist_p2.pdf", bbox_inch="tight")

acqs_plot = ["modelchange_gr", "vopt_gr", "mbr_gr", "sopt_gr", "random_gr", "uncertainty_gr"]
ax = plot_acc_from_npz('results/mnist-even-odd/', "v5-", acqs_plot, err_bar= False, dataset_title='')
plt.savefig("mnist_gr.pdf", bbox_inch="tight")

acqs_plot = ["vopt_hf", "mbr_hf", "sopt_hf", "random_hf", "uncertainty_hf"]
ax = plot_acc_from_npz('results/mnist-even-odd/', "v5-", acqs_plot, err_bar= False, dataset_title='')
plt.savefig("mnist_hf.pdf", bbox_inch="tight")

acqs_plot = ["vopt_p2",  "modelchange_p2", "vopt_p2NA", "modelchange_p2NA"]
ax = plot_acc_from_npz('results/mnist-even-odd/', "v5-", acqs_plot, err_bar= False, dataset_title='')
plt.savefig("mnist_NA.pdf", bbox_inch="tight")


# -

ts = time.time()
W = gm.compute_similarity_graph(X/255.0, 15, 
                                380 * 380, None)
te = time.time()
print("Constructing similarity graph takes %f seconds"% (te - ts))
ts = time.time()
L = gm.compute_laplacian(W, "normed")
L_un = gm.compute_laplacian(W, "unnormed").toarray()
w, v  = gm.compute_spectrum(L, n_eigs=None)
plt.plot(w)

acc1, _ = run_experiment(w, v, labels, tau = 0.1, gamma = 0.01, n_eig = None,
                   num_to_query = 100,
                   n_start  = 10, seed = 42,
                   exact_update = True,
                   acc_classifier_name=None,
                   model_classifier_name = "gr",
                   acquisition="vopt_gr")
acc2, _ = run_experiment(w, v, labels, tau = 0.1, gamma = 0.01, n_eig = None,
                   num_to_query = 100,
                   n_start  = 10, seed = 42,
                   exact_update = True,
                   acc_classifier_name=None,
                   model_classifier_name = "probit2",
                   acquisition="vopt_p2")     

acc3, l3 = run_experiment(w, v, labels, tau = 0.1, gamma = 0.1, n_eig = None,
                   num_to_query = 100,
                   n_start  = 10, seed = 42,
                   exact_update = True,
                   acc_classifier_name=None,
                   model_classifier_name = "probit2",
                   acquisition="random") 
acc4 =  acc_under_diff_model(l3, labels, n_start=10, model_name='gr', tau=0.1, gamma=0.1, w=w, v=v)

plt.plot(acc1, label = "vopt gr")
plt.plot(acc2, label = "vopt probit")
plt.plot(acc3, label = "random")
plt.plot(acc4, label = "random gr")
plt.legend()

plt.plot(acc1, label = "vopt gr")
plt.plot(acc2, label = "vopt probit")
plt.plot(acc3, label = "random")
plt.plot(acc4, label = "random gr")
plt.legend()

labeled = np.load("./results/mnist-even-odd/v5-0/labeled.npz")
print(labeled["random"])
print(labeled["modelchange_p2"])
