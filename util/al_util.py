import numpy as np
import scipy as sp
import scipy.sparse as sps
import scipy.sparse.linalg
from scipy.optimize import lsq_linear
import scipy.linalg as sla
import matplotlib.pyplot as plt
from itertools import product
from scipy.optimize import newton, root_scalar
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import csgraph
import time
from heapq import *
from sklearn.datasets import make_moons
import copy
from itertools import permutations



def get_init_post(C_inv, labeled, gamma2):
    """
    calculate the risk of each unlabeled point
    """
    N = C_inv.shape[0]
    #unlabeled = list(filter(lambda x: x not in labeled, range(N)))
    B_diag = [1 if i in labeled else 0 for i in range(N)]
    B = sp.sparse.diags(B_diag, format='csr')
    return sp.linalg.inv(C_inv + B/gamma2)


def calc_next_m(m, C, y, lab, k, y_k, gamma2):
    ck = C[k,:]
    ckk = ck[k]
    ip = np.dot(ck[lab], y[lab])
    val = ((gamma2)*y_k -ip )/(gamma2*(gamma2 + ckk))
    m_k = m + val*ck
    return m_k

def get_probs(m, sigmoid=False):
    if sigmoid:
        return 1./(1. + np.exp(-3.*m))
    m_probs = m.copy()
    # simple fix to get probabilities that respect the 0 threshold
    m_probs[np.where(m_probs >0)] /= 2.*np.max(m_probs)
    m_probs[np.where(m_probs <0)] /= -2.*np.min(m_probs)
    m_probs += 0.5
    return m_probs



def EEM_full(k, m, C, y, lab, unlab, m_probs, gamma2):
    N = C.shape[0]
    m_at_k = m_probs[k]
    m_k_p1 = calc_next_m(m, C, y, lab, k, 1., gamma2)
    m_k_p1 = get_probs(m_k_p1)
    risk = m_at_k*np.sum([min(m_k_p1[i], 1.- m_k_p1[i]) for i in range(N)])
    m_k_m1 = calc_next_m(m, C, y, lab, k, -1., gamma2)
    m_k_m1 = get_probs(m_k_m1)
    risk += (1.-m_at_k)*np.sum([min(m_k_m1[i], 1.- m_k_m1[i]) for i in range(N)])
    return risk

def EEM_opt_record(m, C, y, labeled, unlabeled, gamma2):
    m_probs = get_probs(m)
    N = C.shape[0]
    risks = [EEM_full(j, m, C, y, labeled, unlabeled, m_probs, gamma2) for j in range(N)]
    k = np.argmin(risks)
    return k, risks



def V_opt(C, unlabeled, gamma2):
    ips = np.array([np.inner(C[k,:], C[k,:]) for k in unlabeled]).flatten()
    v_opt = ips/(gamma2 + np.diag(C)[unlabeled])
    v_opt = (v_opt - min(v_opt))/(max(v_opt) - min(v_opt))
    colors = [(x, 0.5,(1-x)) for x in v_opt]
    plt.scatter(X[unlabeled, 0],X[unlabeled,1], c=colors)
    plt.show()
    k_max = unlabeled[np.argmax(v_opt)]
    return k_max

def Sigma_opt(C, unlabeled, gamma2):
    sums = np.sum(C[np.ix_(unlabeled,unlabeled)], axis=1)
    sums = np.asarray(sums).flatten()**2.
    s_opt = sums/(gamma2 + np.diag(C)[unlabeled])
    k_max = unlabeled[np.argmax(s_opt)]
    return k_max




def V_opt_record(C, unlabeled, gamma2):
    ips = np.array([np.inner(C[k,:], C[k,:]) for k in unlabeled]).flatten()
    v_opt = ips/(gamma2 + np.diag(C)[unlabeled])
    k_max = unlabeled[np.argmax(v_opt)]
    return k_max, v_opt

def V_opt_record2(u, C, unlabeled, gamma2, lam):
    ips = np.array([np.inner(C[k,:], C[k,:]) for k in unlabeled]).flatten()
    v_opt = ips/(gamma2 + np.diag(C)[unlabeled])
    print(np.max(v_opt), np.min(v_opt))
    v_opt += lam*(np.max(v_opt) + np.min(v_opt))*0.5*(1./np.absolute(u[unlabeled])) # add term that bias toward decision boundary
    k_max = unlabeled[np.argmax(v_opt)]
    return k_max, v_opt



def Sigma_opt_record(C, unlabeled, gamma2):
    sums = np.sum(C[np.ix_(unlabeled,unlabeled)], axis=1)
    sums = np.asarray(sums).flatten()**2.
    s_opt = sums/(gamma2 + np.diag(C)[unlabeled])
    k_max = unlabeled[np.argmax(s_opt)]
    return k_max, s_opt





def plot_iter(m, X, labels, labeled, k_next=-1):
    '''
    Assuming labels are +1, -1
    '''
    m1 = np.where(m >= 0)[0]
    m2 = np.where(m < 0)[0]

    #sup1 = list(set(labeled).intersection(set(np.where(labels == 1)[0])))
    #sup2 = list(set(labeled).intersection(set(np.where(labels == -1)[0])))

    corr1 = list(set(m1).intersection(set(np.where(labels == 1)[0])))
    incorr1 = list(set(m2).intersection(set(np.where(labels == 1)[0])))
    corr2 = list(set(m2).intersection(set(np.where(labels == -1)[0])))
    incorr2 = list(set(m1).intersection(set(np.where(labels == -1)[0])))

    print("num incorrect = %d" % (len(incorr1) + len(incorr2)))

    if k_next >= 0:
        plt.scatter(X[k_next,0], X[k_next,1], marker= 's', c='y', alpha= 0.7, s=200) # plot the new points to be included
        plt.title(r'Dataset with Label for %s added' % str(k_next))
    elif k_next == -1:
        plt.title(r'Dataset with Initial Labeling')

    plt.scatter(X[corr1,0], X[corr1,1], marker='x', c='b', alpha=0.2)
    plt.scatter(X[incorr1,0], X[incorr1,1], marker='x', c='r', alpha=0.2)
    plt.scatter(X[corr2,0], X[corr2,1], marker='o', c='r',alpha=0.15)
    plt.scatter(X[incorr2,0], X[incorr2,1], marker='o', c='b',alpha=0.15)

    sup1 = list(set(labeled).intersection(set(np.where(labels == 1)[0])))
    sup2 = list(set(labeled).intersection(set(np.where(labels == -1)[0])))
    plt.scatter(X[sup1,0], X[sup1,1], marker='x', c='k', alpha=1.0)
    plt.scatter(X[sup2,0], X[sup2,1], marker='o', c='k', alpha=1.0)
    plt.axis('equal')
    plt.show()

    return
