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

# %cd D:\Dropbox\working-directory\ActiveLearningCode
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from datasets.data_loaders_mlflow import load_checkerboard
from util.Graph_manager import Graph_manager
import time
import copy
from util.gbssl import *
# %load_ext autoreload

# # Generate Data set and Similarity Graph

# +
N = 2000
seed = 87
#X, labels = make_moons(N, noise=0.15, random_state=seed)

X, labels = load_checkerboard({"num_points":N, "seed":42})
labels[np.where(labels == 0)] = -1
ind_ord = list(np.where(labels == -1)[0]) + list(np.where(labels == 1)[0])
ind_ordnp = np.array(ind_ord)



n_start = 6
np.random.seed(seed)
labeled_orig =  list(np.random.choice(np.where(labels == -1)[0],
                    size=n_start//2, replace=False))
labeled_orig += list(np.random.choice(np.where(labels ==  1)[0],
                    size=n_start//2, replace=False))

plot_iter(labels, X, labels, labeled_orig, subplot=True)
plt.title("Checkerboard Dataset")
plt.show()

# Create similarity graph 
neig = None
graph_params = {
    'knn'    : 15,
    'sigma'  : 3.,
    'Ltype'  : 'normed',
    'n_eigs' : neig,
    'zp_k'   : 7
}
gm = Graph_manager()

print('Num eigenvalues used = %s' %(str(neig))) 
w, v = gm.from_features(X, graph_params, debug=True)
# -

tau, gamma = 0.1, 0.1

from sklearn.preprocessing import OneHotEncoder
labeled = copy.deepcopy(labeled_orig)
enc = OneHotEncoder()
enc.fit(labels.reshape((-1, 1)))
onehot_labels = enc.transform(labels.reshape((-1, 1))).todense()
unlabeled = list(filter(lambda x: x not in labeled, range(N)))
Nu =len(unlabeled)                                      # number of unlabeled points 
y = onehot_labels[labeled]

# # Test __calc_model__
# ## GR

# %load_ext autoreload

# %autoreload 2
FFg = MultiGraphBasedSSLModel('gr', gamma, tau, v=v, w=w)
RFg = MultiGraphBasedSSLModel('gr', gamma, tau, v=v[:,:300], w=w[:300])
FFg.calculate_model(labeled, y)
RFg.calculate_model(labeled, y)
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c = np.array(FFg.m[:,0]).flatten())
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c = np.array(FFg.m[:,1]).flatten())

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c = np.array(RFg.m[:,0]).flatten())
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c = np.array(RFg.m[:,1]).flatten())

# +
FF = MultiGraphBasedSSLModel('gr', gamma, tau, v=v, w=w)
RF = MultiGraphBasedSSLModel('gr', gamma, tau, v=v[:,:300], w=w[:300])

FF.calculate_model(labeled, y)
RF.calculate_model(labeled, y)

# +
Q = np.random.choice(FF.unlabeled, 5, replace=False)
yQ = onehot_labels[Q]

FF.update_model(Q, yQ)
RF.update_model(Q, yQ)
# -

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c = np.array(FF.m[:,0]).flatten())
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c = np.array(FF.m[:,1]).flatten())

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c = np.array(RF.m[:,0]).flatten())
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c = np.array(RF.m[:,1]).flatten())


