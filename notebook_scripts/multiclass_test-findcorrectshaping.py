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

print(labels.shape)

labels_show2 = labels.reshape(m,n).T

print(np.allclose(labels_show, labels_show2))

labeled_orig = []
num_classes = len(np.unique(labels))
for i in np.unique(labels):
    labels_i = np.where(labels == i)[0]
    
    labeled_orig += list(np.random.choice(labels_i,
                    size=len(labels_i)//10, replace=False))

print(len(labeled_orig))

labeled = copy.deepcopy(labeled_orig)
enc = OneHotEncoder()
enc.fit(labels.reshape((-1, 1)))
onehot_labels = enc.transform(labels.reshape((-1, 1))).todense()
unlabeled = list(filter(lambda x: x not in labeled, range(N)))
Nu =len(unlabeled)                                      # number of unlabeled points 
y = onehot_labels[labeled]

print(y)

plt.imshow(labels_show.astype(float))
plt.title("Ground Truth")
plt.show()

salinas_mat_data = loadmat('/Users/Kevin/Dropbox/ActiveLearning/salinas_graph.mat')
w, v = salinas_mat_data['Dn'], salinas_mat_data['Vn']
w = w.reshape(w.shape[0])

tau, gamma = 0.1, .1
RF = MultiGraphBasedSSLModel('gr', gamma, tau, v=v[:,:20], w=w[:20])

RF.calculate_model(labeled, y)
RFm_show = np.array(np.argmax(RF.m, axis=1)).flatten().reshape(m,n).T

plt.imshow(RFm_show/6.)
plt.show()
plt.imshow(labels_show/6.)
plt.show()



# ## Class assignment not lining up correctly, not getting correct accuracy!

print(np.unique(labels))
print(np.unique(RFm_show))


100/7138

print(np.unique(RFm_flat), np.unique(labels_show))

RFm_flat

labels_show

plt.subplot(1,2,1)
plt.imshow(labels_show.astype(float))
plt.title('Ground Truth')
plt.subplot(1,2,2)
RFm_show = RFm_flat.reshape(n,m)
plt.imshow(RFm_show.astype('float'))
plt.title('RF.m')
plt.show()

plt.subplot(1,2,1)
plt.imshow(labels_show.astype(float))
plt.title('Ground Truth')
plt.subplot(1,2,2)
RFm_show = RFm_flat.reshape(n,m)
plt.imshow(RFm_show.astype('float'))
plt.title('RF.m')
plt.show()

RFm_flat_sup = RFm_flat.copy()
RFm_flat_sup[unlabeled] = 7
plt.imshow(RFm_flat_sup.reshape(n,m))
plt.title('RF.m_sup')
plt.show()

labels_flat_sup = labels.copy()
labels_flat_sup[unlabeled] = 7
plt.imshow(labels_flat_sup.reshape(n,m))
plt.title('labels_flat_sup')
plt.show()

print(np.unique(labels))

data = loadmat('/Users/Kevin/Desktop/hyperspectral-data/matlab/salinas/salinas-gr-m.mat')

m_matlab, y = data['m'], data['yy']

y = y.astype(float)

# +
#mm_matlab, mH_matlab = data['mm'], data['mH']

# +
# print(mm_matlab.shape)
#print(mH_matlab.shape)

# +
# plt.imshow(mH_matlab/7.)
# plt.show()
# -

m_matlab_flat = np.argmax(m_matlab, axis=1)
m_matlab_show = m_matlab_flat.reshape(m,n).T

plt.imshow(m_matlab_show/7.)
plt.show()
plt.imshow(labels_show/7.)
plt.show()

print(m_matlab_show[:5,:5])
print(RFm_show[:5,:5])
print(labels_show[:5,:5])

len(np.where(m_matlab_show != labels_show)[0])/N



tau = 0.1
gamma = 1.

labeled = data['labeled']
labeled = list(labeled.reshape(labeled.shape[0]) - 1)

vn, wn, dtau = data['Vn'], data['Dn'], data['Dtau']
wn = wn.reshape(wn.shape[0])

# +
#w = w.reshape(w.shape[0])
# -

print(np.allclose(v, vn))
print(np.allclose(w,wn))

print(wn.shape)
print(len(labeled))
print(y.shape)

C_matlab = data['C']
post_matlab = data['post']

model = MultiGraphBasedSSLModel('gr', gamma, tau, v=vn, w=wn)
m_python = model.get_m(labeled, y)
C_python = model.get_C(labeled, y, m_python)
d = model.d

print(d[:5])
print(dtau[:5])

print(np.allclose(d, dtau.flatten()))


# +
def GRC(Z, gamma, d, v):
    H_d = len(Z) *[1./gamma**2.]
    vZ = v[Z,:]
    post = sp.sparse.diags(d, format='csr') \
           + vZ.T @ sp.sparse.diags(H_d, format='csr') @ vZ
    print(np.allclose(post_matlab, post))
    return v @ sp.linalg.inv(post) @ v.T

def GRm(Z, y, gamma, d, v):
    C = GRC(Z, gamma, d, v)
    return (C[:,Z] @ y)/(gamma * gamma)


# -

vnZ = vn[labeled,:]
post_local = np.diag(d) + vnZ.T @ vnZ

print(np.allclose(post_local, post_matlab))



C_local = GRC(labeled, gamma, d, vn)



print(np.allclose(C_python, C_matlab))

m_python_flat = np.argmax(m_python, axis=1)
m_python_show = m_python_flat.reshape(m,n).T

plt.imshow(m_python_show.astype(float)/7.)
plt.show()

print(np.allclose(m_python_show, m_matlab_show))

1./(1.**2.)

Hd = np.eye(len(labeled))
vZ = vn[labeled, :]
evals = (wn + tau**2.)/(tau**2.)

print(evals[:4])

temp = np.diag(evals) + vZ.T @ Hd @ vZ
C_local = vn @ sp.linalg.inv(temp) @ vn.T

print(C_local.shape)

m_local = C_local[:,labeled] @ y.astype(float)

print(np.allclose(m_matlab, m_local))

print(np.allclose(m_local, m_python))

print(m_matlab[:3,:3])
print()
print(m_local[:3,:3])
print()
print(m_python[:3,:3])

m_matlab_flat = np.array(np.argmax(m_matlab, axis=1)).flatten()
m_python_flat = np.array(np.argmax(m_python, axis=1)).flatten()
m_matlab_show = m_matlab_flat.reshape(n,m)
m_python_show = m_python_flat.reshape(n,m)

plt.subplot(1,2,1)
plt.imshow(m_matlab_show.astype(float))
plt.title('MATLAB')
plt.subplot(1,2,2)
plt.imshow(m_python_show.astype('float'))
plt.title('Python')
plt.show()

print(m_matlab[:5,:5])

plt.imshow(m_matlab[:20,:])

m_matlab_flat = np.array(np.argmax(m_matlab, axis=1)).flatten()
print(m_matlab_flat)

labels_show.shape


