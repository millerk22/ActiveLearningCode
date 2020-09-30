import numpy as np
import scipy.linalg as sla
import matplotlib.pyplot as plt

def sqdist(X, Y):
    """
    Computes dense pairwise euclidean distance between X and Y
    """
    m = X.shape[1]
    n = Y.shape[1]
    Yt = Y.T
    XX = np.sum(X*X, axis=0)
    YY = np.sum(Yt*Yt, axis=1).reshape(n, 1)
    return np.tile(XX, (n, 1)) + np.tile(YY, (1, m)) - 2*Yt.dot(X)

def nystrom(X, num_samples, metric='euclidean', fid=None, sigma=-1):
    '''

        * fid needs to be a list
    '''
    N, d = X.shape
    if fid is None:
        sampled = np.random.choice(range(N), num_samples, replace=False)
    else:
        nonfid = list(filter(lambda x : x not in fid, range(N)))
        sampled = fid + list(np.random.choice(nonfid, num_samples - len(fid), replace=False))
    other = list(filter(lambda x: x not in sampled, range(N)))
    perm_inds = sampled + other
    X_sampled, X_other = X[sampled, :], X[other, :]

    if metric == 'euclidean':
        A, B = sqdist(X_sampled.T, X_sampled.T), sqdist(X_other.T, X_sampled.T)
        if sigma == -1:
            # "auto" scaling -- from Xiyang's code
            sigma = np.percentile(A.flatten(), 18)*1.1
            A, B = np.exp(-A/sigma), np.exp(-B/sigma)
        elif sigma > 0:
            A, B = np.exp(-A/sigma), np.exp(-B/sigma)
        else:
            raise NotImplementedError("Haven't implemented other option besides auto scaling...")
    else:
        raise NotImplementedError("Haven't implemented other metric besides euclidean")

    # Normalize A and B using row sums
    sumB = np.sum(B, axis=1)

    d1hat = np.sqrt(1./(np.sum(A, axis=1) + sumB))
    d2hat = np.sqrt(1./(np.sum(B, axis=0) + B.T @ sla.inv(A) @ sumB))
    A *= np.outer(d1hat, d1hat)
    B *= np.outer(d1hat, d2hat)

    D, Bx = sla.eigh(A)
    # plt.scatter(range(len(D)), D)
    # plt.show()
    S = (Bx * (1./np.sqrt(D))) @ Bx.T   # S = Bx D^{-1/2}Bx^T
    SB = S @ B
    R = A + SB @ SB.T
    if not np.allclose(R, R.T):
        print("R is not symmetric")
        R = 0.5*(R + R.T)
    #R = 0.5*(R + R.T)
    E, U = sla.eigh(R)
    U = U[:,::-1]
    E = E[::-1]
    # plt.scatter(range(len(E)), E)
    # plt.show()

    Phi = np.concatenate((Bx * np.sqrt(D), B.T @ (Bx * 1./np.sqrt(D))))
    BxTUEi = Bx.T @ (U * 1./np.sqrt(E))
    Phi = Phi @ BxTUEi

    Phi = Phi[perm_inds,:]


    return 1. - E, Phi
