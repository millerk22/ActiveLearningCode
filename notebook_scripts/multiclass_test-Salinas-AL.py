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

# # Get Hyperspectral Data -- Salinas
#
# __NOTE__: We need to shape our data in the following way to align with how MATLAB does indexing for the graph construction. 
#
# If our HSI image is shaped $(n,m,s)$, then the labels image will be shaped $(n,m)$. But to get the proper indexing in the MATLAB create similarity graph, we need to perform the following 
#     '''
#     
#     labels = data['labels']
#     labels = labels.T.flatten() # to get labels in the ordering that is in the similarity graph
#     
#     # with m as the classifier  of shape (m*n, )
#     m_show = m.reshape(m,n).T
#     
#     '''
# This gives the proper reindexing and reshaping to align with the similarity graph indexing to fit the indexing in labels.

from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder

salinas_data = np.load('/Users/Kevin/Dropbox/ActiveLearning/salinas.npz')
labels_show = salinas_data['labels']
n,m = labels_show.shape
N = n*m
labels = labels_show.T.flatten()

salinas_mat_data = loadmat('/Users/Kevin/Dropbox/ActiveLearning/salinas_graph.mat')
w, v = salinas_mat_data['Dn'], salinas_mat_data['Vn']
w = w.reshape(w.shape[0])

print(labels.shape)

labels_show2 = labels.reshape(m,n).T

print(np.allclose(labels_show, labels_show2))

labeled_orig = []
num_classes = len(np.unique(labels))
for i in np.unique(labels):
    labels_i = np.where(labels == i)[0]
    
    labeled_orig += list(np.random.choice(labels_i,
                    size=int(len(labels_i)*0.7), replace=False))

print(len(labeled_orig))

labeled = copy.deepcopy(labeled_orig)
enc = OneHotEncoder()
enc.fit(labels.reshape((-1, 1)))
onehot_labels = enc.transform(labels.reshape((-1, 1))).todense()
unlabeled = list(filter(lambda x: x not in labeled, range(N)))
Nu =len(unlabeled)                                      # number of unlabeled points 
y = onehot_labels[labeled]

plt.imshow(labels_show.astype(float))
plt.title("Ground Truth")
plt.show()

# # Try Active Learning
#
# ### GR 'multi' model

tau, gamma = 0.1, .1
RF = MultiGraphBasedSSLModel('gr', gamma, tau, v=v[:,:100], w=w[:100])

RF.calculate_model(labeled, y)
RFm_show = np.array(np.argmax(RF.m, axis=1)).flatten().reshape(m,n).T

plt.imshow(RFm_show/6.)
plt.show()
plt.imshow(labels_show/6.)
plt.show()

# %autoreload

# +
num_batches = 5
batch_size = 10
debug = True
cand = 'rand'
method = 'top'
exact_update = True

AL = ActiveLearner('rand', candidate=cand)

acc = [get_acc_multi(np.array(np.argmax(RF.m, axis=1)).flatten(), labels, unlabeled = RF.unlabeled)[1]]
for _ in range(num_batches):
    print("Iteration %d" % (_ + 1))
    print(len(RF.labeled))
    # Select query points with ActiveLearner, using model change
    if debug and method == 'prop':
        Q, p, acq_vals, Cand = AL.select_query_points(RF, B=batch_size, method=method, debug=True)
        # Plot the sampling heatmap
        flat_f = np.zeros(N)
        flat_f[Cand] = p
        plot_f = flat_f.reshape(m,n).T
        plt.subplot(1,2,1)
        plt.imshow(plot_f)
        plt.colorbar()
        plt.title("Probabilities")
        plt.subplot(1,2,2)
        plt.imshow(np.array(np.argmax(RF.m, axis=1)).flatten().reshape(m,n).T)
        plt.title("RF.m")
        plt.show()
    else:
        Q = AL.select_query_points(RF, B=batch_size, method=method)
    
    
    # Query oracle for labels
    yQ = onehot_labels[Q,:]
    
    # Retrain model 
    RF.update_model(Q, yQ, exact=exact_update)
    plt.subplot(3,1,1)
    plt.hist(labels[Q])
    plt.title("Labeled Dist")
    plt.subplot(3,1,2)
    plt.hist(labels[RF.labeled])
    plt.title("Labeled Full Dist")
    plt.subplot(3,1,3)
    plt.hist(labels)
    plt.title("True Labeled Dist")
    plt.show()
    acc.append(get_acc_multi(np.array(np.argmax(RF.m, axis=1)).flatten(), labels, unlabeled = RF.unlabeled)[1])
# -

## Random Acq function
RFm_show = np.array(np.argmax(RF.m, axis=1)).flatten().reshape(m,n).T
print(acc)
plt.imshow(RFm_show)
plt.title("RFm")
plt.show()
plt.imshow(labels_show)
plt.title("Ground Truth")
plt.show()

# +
### MC with rand candidate selection
# -

RFm_show = np.array(np.argmax(RF.m, axis=1)).flatten().reshape(m,n).T
print(acc)
plt.imshow(RFm_show)
plt.title("RFm")
plt.show()
plt.imshow(labels_show)
plt.title("Ground Truth")
plt.show()

# ### Observations
#
# Not getting a huge accuracy boost with doing AL, though it is improving... would need to do a bigger test. Moving on to 1 v all test.

# ## 1 vs All

labeled = copy.deepcopy(labeled_orig)
enc = OneHotEncoder()
enc.fit(labels.reshape((-1, 1)))
onehot_labels = np.array(enc.transform(labels.reshape((-1, 1))).todense())
unlabeled = list(filter(lambda x: x not in labeled, range(N)))
Nu =len(unlabeled)                                      # number of unlabeled points 
y = np.array(onehot_labels[labeled])

y[y == 0] = -1.

print(np.unique(labels))

Labels = [labels.copy().astype(int) for i in range(y.shape[1])]
for i, labels_i in enumerate(Labels):
    labels_i[labels_i != i] = -1

print(labels[:10])

print(Labels[0][:10])

tau, gamma = 0.1, .01
RFS = [BinaryGraphBasedSSLModelReduced('gr', gamma, tau, v=v[:,:200], w=w[:200]) for i in range(y.shape[1])]

for i, RF in enumerate(RFS):
    RF.calculate_model(list(labeled), list(y[:,i]))
    RFm_show = np.array(RF.m).flatten().reshape(m,n).T
    plt.imshow(RFm_show)
    plt.title("Class %d" % (i))
    plt.show()

# %autoreload

# +
num_batches = 50
num_class = y.shape[1]
class_batch_size = 5
batch_size = class_batch_size*num_class

debug = True
cand = 'rand'    # candidate set selection in ['rand', 'full']
method = 'top'    # method for choosing query set ['top', 'prop']
exact_update = True

AL = ActiveLearner('rand', candidate=cand) # CAN CHANGE acquisition function here ['rand', 'mc', 'uncertainty']

m_multi = np.zeros((N, num_class))
for c, RF in enumerate(RFS):
    m_multi[:,c] = np.array(RF.m).flatten()

acc = [get_acc_multi(np.array(np.argmax(m_multi, axis=1)).flatten(), labels, unlabeled = RFS[0].unlabeled)[1]]
for _ in range(num_batches):
    print("Iteration %d" % (_ + 1))
    print(len(RF.labeled))
    Q = []
    for c in range(y.shape[1]):
        # Select query points with ActiveLearner, using model change
        if debug and method == 'prop':
            q, p, acq_vals, Cand = AL.select_query_points(RFS[c], B=class_batch_size, method=method, debug=True)
            # Plot the sampling heatmap
            flat_f = np.zeros(N)
            flat_f[Cand] = p
            plot_f = flat_f.reshape(m,n).T
            plt.subplot(1,2,1)
            plt.imshow(plot_f)
            plt.colorbar()
            plt.title("Probabilities")
            plt.subplot(1,2,2)
            plt.imshow(np.array(np.argmax(RFS[c].m, axis=1)).flatten().reshape(m,n).T)
            plt.title("RF.m")
            plt.show()
        else:
            q = AL.select_query_points(RFS[c], B=class_batch_size, method=method)
        Q += q
    
    # Query oracle for labels
    yQ = onehot_labels[Q,:]
    yQ[yQ == 0.] = -1. 
    
    # Retrain models
    m_multi = np.zeros((N, num_class))
    for c, RF in enumerate(RFS):
        RF.update_model(Q, list(np.array(yQ[:,c])), exact=False)
        m_multi[:,c] = np.array(RF.m).flatten()
        
    acc.append(get_acc_multi(np.array(np.argmax(m_multi, axis=1)).flatten(), labels, unlabeled = RFS[0].unlabeled)[1])
# -

print("B = 5,  candidate set = rand, selection method = top")
print(acc)

# ### Observations 
# Looks like MC does indeed have an improvement over random sampling, though it seems that the limitation really is mostly in the model?

# +
labeled = copy.deepcopy(labeled_orig)
enc = OneHotEncoder()
enc.fit(labels.reshape((-1, 1)))
onehot_labels = np.array(enc.transform(labels.reshape((-1, 1))).todense())
unlabeled = list(filter(lambda x: x not in labeled, range(N)))
Nu =len(unlabeled)                                      # number of unlabeled points 
y = np.array(onehot_labels[labeled])
y[y == 0] = -1.

Labels = [labels.copy().astype(int) for i in range(y.shape[1])]
for i, labels_i in enumerate(Labels):
    labels_i[labels_i != i] = -1

# +
tau, gamma = 0.1, .1
RFS = [BinaryGraphBasedSSLModelReduced('probit-log', gamma, tau, v=v[:,:200], w=w[:200]) for i in range(y.shape[1])]

for i, RF in enumerate(RFS):
    RF.calculate_model(list(labeled), list(y[:,i]))
    RFm_show = np.array(RF.m).flatten().reshape(m,n).T
    plt.imshow(RFm_show)
    plt.title("Class %d" % (i))
    plt.show()

# +
num_batches = 50
num_class = y.shape[1]
class_batch_size = 5
batch_size = class_batch_size*num_class

debug = True
cand = 'rand'
method = 'top'
exact_update = True

AL = ActiveLearner('mc', candidate=cand)

m_multi = np.zeros((N, num_class))
for c, RF in enumerate(RFS):
    m_multi[:,c] = np.array(RF.m).flatten()

acc = [get_acc_multi(np.array(np.argmax(m_multi, axis=1)).flatten(), labels, unlabeled = RFS[0].unlabeled)[1]]
for _ in range(num_batches):
    print("Iteration %d" % (_ + 1))
    print(len(RF.labeled))
    Q = []
    for c in range(y.shape[1]):
        # Select query points with ActiveLearner, using model change
        if debug and method == 'prop':
            q, p, acq_vals, Cand = AL.select_query_points(RFS[c], B=class_batch_size, method=method, debug=True)
            # Plot the sampling heatmap
            flat_f = np.zeros(N)
            flat_f[Cand] = p
            plot_f = flat_f.reshape(m,n).T
            plt.subplot(1,2,1)
            plt.imshow(plot_f)
            plt.colorbar()
            plt.title("Probabilities")
            plt.subplot(1,2,2)
            plt.imshow(np.array(np.argmax(RFS[c].m, axis=1)).flatten().reshape(m,n).T)
            plt.title("RF.m")
            plt.show()
        else:
            q = AL.select_query_points(RFS[c], B=class_batch_size, method=method)
        Q += q
    
    # Query oracle for labels
    yQ = onehot_labels[Q,:]
    yQ[yQ == 0.] = -1. 
    
    # Retrain models
    m_multi = np.zeros((N, num_class))
    for c, RF in enumerate(RFS):
        RF.update_model(Q, list(np.array(yQ[:,c])), exact=False)
        m_multi[:,c] = np.array(RF.m).flatten()
        
    acc.append(get_acc_multi(np.array(np.argmax(m_multi, axis=1)).flatten(), labels, unlabeled = RFS[0].unlabeled)[1])
# -

print(acc)

print(len(RFS[0].labeled))

# # Urban Dataset

urban_data = np.load('/Users/Kevin/Dropbox/ActiveLearning/urban.npz')
labels_show = urban_data['labels']
n,m = labels_show.shape
N = n*m
labels = labels_show.T.flatten()

urban_mat_data = loadmat('/Users/Kevin/Dropbox/ActiveLearning/urban_graph.mat')
w, v = urban_mat_data['Dn'], urban_mat_data['Vn']
w = w.reshape(w.shape[0])

labeled_orig = []
num_classes = len(np.unique(labels))
for i in np.unique(labels):
    labels_i = np.where(labels == i)[0]
    
    labeled_orig += list(np.random.choice(labels_i,
                    size=len(labels_i)//1000, replace=False))
print(len(labeled_orig))

labeled = copy.deepcopy(labeled_orig)
enc = OneHotEncoder()
enc.fit(labels.reshape((-1, 1)))
onehot_labels = enc.transform(labels.reshape((-1, 1))).todense()
unlabeled = list(filter(lambda x: x not in labeled, range(N)))
Nu =len(unlabeled)                                      # number of unlabeled points 
y = np.array(onehot_labels[labeled])
y[y == 0] = -1.
Labels = [labels.copy().astype(int) for i in range(y.shape[1])]
for i, labels_i in enumerate(Labels):
    labels_i[labels_i != i] = -1
plt.imshow(labels_show.astype(float))
plt.title("Ground Truth")
plt.show()

# +
tau, gamma = 0.1, .1
RFS = [BinaryGraphBasedSSLModelReduced('probit-log', gamma, tau, v=v[:,:20], w=w[:20]) for i in range(y.shape[1])]

for i, RF in enumerate(RFS):
    RF.calculate_model(list(labeled), list(y[:,i]))
    RFm_show = np.array(RF.m).flatten().reshape(m,n).T
    plt.imshow(RFm_show)
    plt.title("Class %d" % (i))
    plt.show()

# +
num_batches = 50
num_class = y.shape[1]
class_batch_size = 5
batch_size = class_batch_size*num_class

debug = True
cand = 'rand'
method = 'top'
exact_update = True

AL = ActiveLearner('mc', candidate=cand)

m_multi = np.zeros((N, num_class))
for c, RF in enumerate(RFS):
    m_multi[:,c] = np.array(RF.m).flatten()

acc = [get_acc_multi(np.array(np.argmax(m_multi, axis=1)).flatten(), labels, unlabeled = RFS[0].unlabeled)[1]]
for _ in range(num_batches):
    print("Iteration %d" % (_ + 1))
    print(len(RF.labeled))
    Q = []
    for c in range(y.shape[1]):
        # Select query points with ActiveLearner, using model change
        if debug and method == 'prop':
            q, p, acq_vals, Cand = AL.select_query_points(RFS[c], B=class_batch_size, method=method, debug=True)
            # Plot the sampling heatmap
            flat_f = np.zeros(N)
            flat_f[Cand] = p
            plot_f = flat_f.reshape(m,n).T
            plt.subplot(1,2,1)
            plt.imshow(plot_f)
            plt.colorbar()
            plt.title("Probabilities")
            plt.subplot(1,2,2)
            plt.imshow(np.array(np.argmax(RFS[c].m, axis=1)).flatten().reshape(m,n).T)
            plt.title("RF.m")
            plt.show()
        else:
            q = AL.select_query_points(RFS[c], B=class_batch_size, method=method)
        Q += q
    
    # Query oracle for labels
    yQ = onehot_labels[Q,:]
    yQ[yQ == 0.] = -1.
    
    plot_choices = np.zeros(N)
    plot_choices[Q] = 1.
    plt.imshow(plot_choices.reshape(m,n).T)
    plt.title("Iteration %d Choices" % _)
    plt.show()
    
    # Retrain models
    m_multi = np.zeros((N, num_class))
    for c, RF in enumerate(RFS):
        RF.update_model(Q, list(np.array(yQ[:,c])), exact=False)
        m_multi[:,c] = np.array(RF.m).flatten()
        
    acc.append(get_acc_multi(np.array(np.argmax(m_multi, axis=1)).flatten(), labels, unlabeled = RFS[0].unlabeled)[1])
# -

plt.hist(labels[RFS[0].labeled])
plt.title("Labeled Data Proportions")
plt.show()
plt.hist(labels)
plt.title("True Labeled Data proportions")
plt.show()

plt.hist(labels[RFS[0].labeled])
plt.title("Labeled Data Proportions")
plt.show()
plt.hist(labels)
plt.title("True Labeled Data proportions")
plt.show()

print(acc3)

print(acc2)

print(acc)

plt.plot(range(len(acc)), acc, label='mc')
plt.plot(range(len(acc)), acc2, label='rand')
plt.plot(range(len(acc)), acc3, label='unc')
plt.ylim(0.8,0.9)
plt.legend()
plt.show()

print(len(RFS[0].labeled))


