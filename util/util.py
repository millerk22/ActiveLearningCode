import numpy as np
import scipy.sparse as sps
import scipy.linalg as sla
import matplotlib.pyplot as plt
from itertools import permutations

def threshold1D(u, zero_one=False):
    if zero_one:
        u[u >= 0.5] = 1.
        u[u < 0.5] = 0.
    else:
        u[u >= 0.0] = 1.
        u[u < 0.0] = -1.
    return u

def threshold2D(u, one_hot=True):
    thresh_ind = np.argmax(u, axis=1)
    if not one_hot:
        return thresh_ind
    u *= 0.
    u[[i for i in range(u.shape[0])], thresh_ind] = 1.
    return u

def threshold2D_many(u_samples):
    print('Not working... will cause error')
    thresh_ind = np.argmax(u_samples, axis=1)
    u_samples[:,:,:] = 0
    first_ind = np.array([[i for i in range(u_samples.shape[0])] for j in range(u_samples.shape[2])])
    third_ind = np.array([[i for j in range(u_samples.shape[2])] for i in range(u_samples.shape[0])])
    print(first_ind)
    print(first_ind.flatten())
    print(third_ind)
    print(third_ind.flatten())
    u_samples[np.ix_(first_ind.flatten(), thresh_ind.flatten(), third_ind.flatten())] = 1.
    return u_samples

def threshold1D_avg(u_samples):
    return np.average(1.*(u_samples >= 0.), axis=1) # ith entry is estimated mean that ith node is of class 1

def threshold2D_avg(u_samples):
    N, num_class, n_samples = u_samples.shape
    thresh_ind = np.argmax(u_samples, axis=1)
    u_thresh_avg = np.zeros((N, num_class))
    for i in range(N):
        u_thresh_avg[i,:] = np.bincount(thresh_ind[i,:],minlength=num_class)/n_samples
    return u_thresh_avg



def plot_iter(stats, X, k_next=-1):
    corr1 = stats['corr1']
    corr2 = stats['corr2']
    sup1 = stats['sup1']
    sup2 = stats['sup2']
    incorr1 = stats['incorr1']
    incorr2 = stats['incorr2']
    if type(k_next) == type([1, 2]):
        plt.scatter(X[np.ix_(k_next,[0])], X[np.ix_(k_next,[1])], marker= 's', c='y', alpha= 0.7, s=200) # plot the new points to be included
        plt.title('Dataset with Label for %s added' % str(k_next))
    elif k_next >= 0:
        plt.scatter(X[k_next,0], X[k_next,1], marker= 's', c='y', alpha= 0.7, s=200) # plot the new points to be included
        plt.title('Dataset with Label for %s added' % str(k_next))
    elif k_next == -1:
        plt.title('Dataset with Initial Labeling')

    plt.scatter(X[corr1,0], X[corr1,1], marker='x', c='b', alpha=0.2)
    plt.scatter(X[incorr1,0], X[incorr1,1], marker='x', c='r', alpha=0.2)
    plt.scatter(X[corr2,0], X[corr2,1], marker='o', c='r',alpha=0.15)
    plt.scatter(X[incorr2,0], X[incorr2,1], marker='o', c='b',alpha=0.15)
    plt.scatter(X[sup1,0], X[sup1,1], marker='x', c='b', alpha=1.0)
    plt.scatter(X[sup2,0], X[sup2,1], marker='o', c='r', alpha=1.0)
    plt.axis('equal')
    plt.show()
    return




def calc_stats_multi(m, fid, gt_flipped, _print=False):
    stats = {}

    N = m.shape[0]
    classes = list(fid.keys())
    offset = min(classes)

    if offset == -1:
        m1 = np.where(m >= 0)[0]
        m2 = np.where(m < 0)[0]


        sup1 = fid[1]
        sup2 = fid[-1]
        corr1 = list(set(m1).intersection(set(gt_flipped[1])))
        incorr1 = list(set(m2).intersection(set(gt_flipped[1])))
        corr2 = list(set(m2).intersection(set(gt_flipped[-1])))
        incorr2 = list(set(m1).intersection(set(gt_flipped[-1])))

        stats['corr1'] = corr1
        stats['corr2'] = corr2
        stats['sup1'] = sup1
        stats['sup2'] = sup2
        stats['incorr1'] = incorr1
        stats['incorr2'] = incorr2

        error = ((len(incorr1) + len(incorr2))/N )

    else:
        m_class = np.argmax(m,axis=1) + offset # get the class labelings for the current m
        m_class_ind = {c:np.where(m_class == c)[0] for c in classes}

        corr = {c : list(set(m_class_ind[c]).intersection(set(gt_flipped[c]))) for c in classes}
        incorr = {}
        for corr_class, wr_class in permutations(classes, 2):
            if corr_class not in incorr.keys():
                incorr[corr_class] = [(wr_class, list(set(m_class_ind[wr_class]).intersection(set(gt_flipped[corr_class]))))]
            else:
                incorr[corr_class].append((wr_class, list(set(m_class_ind[wr_class]).intersection(set(gt_flipped[corr_class])))))

        stats['corr'] = corr
        stats['incorr'] = incorr

        tot_correct = 0.
        for corr_c_nodes in corr.values():
            tot_correct += len(corr_c_nodes)
        acc = tot_correct/N
        error = 1. - acc


    if _print:
        print('Error = %f' % error )

    return error, stats


COLORS = ['b', 'r', 'g',  'cyan', 'k', 'y']
MARKERS = ['x', 'o', '^', '+', 'v']

def plot_iter_multi(stats, X, fid, k_next=-1):
    corr = stats['corr']
    incorr = stats['incorr']
    if type(k_next) == type([1, 2]):
        plt.scatter(X[np.ix_(k_next,[0])], X[np.ix_(k_next,[1])], marker= 's', c='y', alpha= 0.7, s=200) # plot the new points to be included
        plt.title('Dataset with Label for %s added' % str(k_next))
    elif k_next >= 0:
        plt.scatter(X[k_next,0], X[k_next,1], marker= 's', c='y', alpha= 0.7, s=200) # plot the new points to be included
        plt.title('Dataset with Label for %s added' % str(k_next))
    elif k_next == -1:
        plt.title('Dataset with Initial Labeling')

    classes = corr.keys()
    ofs = min(classes)
    num_class = len(classes)

    if num_class > len(COLORS):
        print('Not enough colors in COLORS list, reusing colors with other markers.')


    for c, c_nodes in corr.items():
        plt.scatter(X[c_nodes,0], X[c_nodes,1], marker=MARKERS[(c-ofs)%num_class], c=COLORS[(c-ofs)%num_class], alpha=0.2)
    for corr_c, vals in incorr.items():
        incorr_c, incorr_nodes = zip(*vals)
        for l in range(len(incorr_c)):
            plt.scatter(X[incorr_nodes[l],0], X[incorr_nodes[l],1], marker=MARKERS[(corr_c-ofs)%num_class],
                                        c=COLORS[(incorr_c[l]-ofs)%num_class], alpha=0.2)
    for c, fid_c in fid.items():
        plt.scatter(X[fid_c,0], X[fid_c,1], marker=MARKERS[(c-ofs)%num_class], c=COLORS[(c-ofs)%num_class], alpha=1.0)
    plt.axis('equal')
    plt.show()
    return




def calc_GR_posterior(v, w, fid, labeled, unlabeled, tau, alpha, gamma2):
    N = v.shape[0]
    classes = fid.keys()
    num_class = len(classes)
    ofs = min(classes)

    if -1 in fid.keys():
        y = np.zeros(N)  # this will already be in the expanded size, as if (H^Ty)
        y[fid[1]] = 1.
        y[fid[-1]] = -1.
    else:
        y = np.zeros((N, num_class))  # this will already be in the expanded size, as if (H^Ty)
        for c, fid_c in fid.items():
            y[fid_c,c-ofs] = 1    # TODO: make it sparse

    N_prime = len(labeled)
    #w_inv = (tau ** (2 * alpha)) * np.power(w + tau**2., -alpha)     # diagonalization of C_t,e
    w_inv =  np.power(w + tau**2., -alpha)     # diagonalization of C_t,e
    C_tau = v.dot((v*w_inv).T)
    C_ll = C_tau[np.ix_(labeled, labeled)]
    C_all_l = C_tau[:,labeled]
    C_ll[np.diag_indices(N_prime)] += gamma2  # directly changing C_ll
    A_inv = sla.inv(C_ll)
    Block1 = C_all_l.dot(A_inv)
    C = C_tau - Block1.dot(C_all_l.T)
    m = Block1.dot(y[labeled])
    if -1 in fid.keys():
        m = np.asarray(m).flatten()
    else:
        m = np.asarray(m)
    return m, np.asarray(C), y




""" Summary Stats object and associated functions """

def summary_stats(K, N, gt, calc):
    C = np.zeros((K, K), dtype=np.float32)
    for i in range(N):
        C[gt[i], calc[i]] += 1.
    recall_confmat = C / np.sum(C,axis=1)[:,np.newaxis]
    precision_confmat = C.T /np.sum(C, axis=0)[:,np.newaxis]
    recall = np.average(np.diagonal(recall_confmat))
    precision = np.average(np.diagonal(precision_confmat))
    acc = len(np.where(calc == gt)[0])/N
    return C, recall_confmat, precision_confmat, recall, precision, acc

def summary_stats_1(N, gt, calc):
    C = np.zeros((2,2), dtype=np.float32)
    gt_shift = ((gt + 1)/2).astype(int)
    calc_shift = ((calc + 1)/2).astype(int)
    for i in range(N):
        C[gt_shift[i],calc_shift[i]] += 1.
    recall_confmat = C / np.sum(C,axis=1)[:,np.newaxis]
    precision_confmat = C.T /np.sum(C, axis=0)[:,np.newaxis]
    recall = np.average(np.diagonal(recall_confmat))
    precision = np.average(np.diagonal(precision_confmat))
    acc = len(np.where(calc == gt)[0])/N
    return C, recall_confmat, precision_confmat, recall, precision, acc



class SummaryStats(object):
    def __init__(self):
        self.C = None
        self.recall_cm = None
        self.precision_cm = None
        self.recall = None
        self.precision = None
        self.acc = None

    def compute(self, gt, calc, N, K):
        classes = np.unique(gt)
        if K == 2 and -1 in classes:
            sum_stats = summary_stats_1(N, gt, calc)
        else:
            sum_stats = summary_stats(K, N, gt, calc)
        self.C, self.recall_cm, self.precision_cm, self.recall, self.precision, self.acc = sum_stats
