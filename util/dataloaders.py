import numpy as np
import scipy.sparse as sps
import scipy.linalg as sla
from sklearn.datasets import make_moons, load_digits
import sys
import os
import requests
import pickle
from io import BytesIO
from zipfile import ZipFile
import gzip
import matplotlib.pyplot as plt
from scipy.io import loadmat
sys.path.append('../util/')
from util.util import plot_iter_multi, plot_iter, calc_stats_multi




class Data_obj(object):
    """
        Object for storing the important, dataset specific attributes for use by the
        MCMC_Sampler classes in ``mcmc_classes.py``. Instantiation requires:
            - X : (N x d) input data, though can be None if data is too large.
            - evals : (num_eigs, ) numpy array of eigenvalues of graph Laplacian
            - evecs : (N x num_eigs) numpy array, columns are the eigenvectors
            - fid : python dict with (class : indices) of the labeled nodes for the dataset
            - ground_truth : (N, ) numpy array with the ground truth labelings for each corresponding
                index's node.

        Member Functions:
            - get_useful_structs() : After instantiating the Data_obj with these required attributes, calculate
                useful structures for later use in the sampling algorithms.
            - plot_initial(): If data in X is 2-dimensional, plot the dataset. (Currently
                only implemented for 2D data)
            - plot_iteration(u): If data in X is 2-dimensional, plot the dataset with colors according
                to values in given parameter u ( a (N,) numpy array). (Currently only implemented
                for 2D data)

    """

    def __init__(self, X, evals, evecs, fid, ground_truth):
        self.X = X
        self.evals = evals
        self.evecs = evecs
        self.fid = fid
        self.ground_truth = ground_truth.astype(int)
        self.N = self.ground_truth.shape[0]
        self.classes = list(self.fid.keys())
        self.num_class = len(self.classes)
        self.have_useful = False
        #self.get_useful_structs()  # maybe call this function right away instead of waiting?


    def get_useful_structs(self):
        self.org_indices = []  # org_indices organizes indices of nodes according to ground truth class assignments
        for i in self.classes:
            self.org_indices.extend(list(np.where(self.ground_truth == i)[0]))

        indices = np.array(list(range(self.N)))  # 0, 1, 2, ... for use in other calculations
        self.labeled = set()        # labeled nodes list
        self.gt_flipped = {}      # ground_truth "flipped" k,v pairs -> i : class(i). don't need??
        for i in self.classes:
            i_mask = indices[self.ground_truth == i]
            self.gt_flipped[i] = i_mask
            self.labeled = self.labeled.union(set(self.fid[i]))
        self.unlabeled = sorted(list(set(indices) - self.labeled))
        self.labeled = sorted(list(self.labeled))

        self.have_useful = True


    def plot_initial(self):
        if not self.have_useful:
            self.get_useful_structs()
        if self.X is None:
            raise ValueError('Sorry data points in ambient space were not saved...')
        if self.X.shape[1] > 2:
            raise NotImplementedError('Sorry, plotting for higher dimensional datasets is not yet implemented..')

        if min(self.classes) == -1:
            _, stats = calc_stats_multi(self.ground_truth, self.fid, self.gt_flipped)
            plot_iter(stats, self.X, k_next=-1)
        else:
            indices = np.array(list(range(self.N)))  # 0, 1, 2, ... for use in other calculations
            m_dummy = np.zeros((self.N, self.num_class))
            for i in range(self.num_class):
                i_mask = indices[self.ground_truth == self.classes[i]]
                m_dummy[i_mask, i] = 1.
            _, stats = calc_stats_multi(m_dummy, self.fid, self.gt_flipped)
            plot_iter_multi(stats, self.X, self.fid, k_next=-1)


    def plot_iteration(self, u):
        if not self.have_useful:
            self.get_useful_structs()
        if self.X is None:
            raise ValueError('Sorry data points in ambient space were not saved...')
        if self.X.shape[1] > 2:
            raise NotImplementedError('Sorry, plotting for higher dimensional datasets is not yet implemented..')
        if min(self.classes) == -1:
            _, stats = calc_stats_multi(self.ground_truth, self.fid, self.gt_flipped)
            plot_iter(stats, self.X, k_next=-1)
        else:
            indices = np.array(list(range(self.N)))  # 0, 1, 2, ... for use in other calculations
            m_dummy = np.zeros((self.N, self.num_class))
            for i in range(self.num_class):
                i_mask = indices[self.ground_truth == self.classes[i]]
                m_dummy[i_mask, i] = 1.
            _, stats = calc_stats_multi(m_dummy, self.fid, self.gt_flipped)
            plot_iter_multi(stats, self.X, self.fid, k_next=-1)


    """TODO for Active Learning: Need a way to have an "original" fid dictionary, that can then pass to
     different runs of Active Learning with UQ samples. Or just have function to
     undo all of the added fidelity points."""



""" Functions that load different datasets into Data_obj objects. """



def load_2_moons(N=2000, noise=0.2, sup_percent=0.05, normed_lap=False, num_eig=None, seed=None, zero_one=False, knn=5, sigma=1.):
    print("Loading the 2 moons data with %d total points..." % N)
    # call the sklearn function for making moons data
    #X, ground_truth = make_moons(n_samples=N, noise=noise, random_state=seed, shuffle=False)
    X, ground_truth = make_moons(n_samples=N, noise=noise, shuffle=False)
    # if want +1, -1 classes, change the corresponding entries
    if not zero_one:
        ground_truth[np.where(ground_truth == 0)] = -1

    # Define the fidelity dictionary
    classes = np.unique(ground_truth)
    indices = np.array(list(range(N)))
    fid = {}
    np.random.seed(seed)
    for i in classes:
        i_mask = indices[ground_truth ==i]
        np.random.shuffle(i_mask)
        n_i = len(i_mask)
        fid[i] = list(i_mask[:int(sup_percent*n_i)])

    # Graph Generation and Eigen-Calculation
    W = make_sim_graph(X, k_nn=knn, sigma=sigma)
    evals, evecs = get_eig_Lnorm(W, num_eig=num_eig, normed_=normed_lap)

    return Data_obj(X, evals, evecs, fid, ground_truth)


def load_gaussian_cluster(Ns, means, covs, sup_percent=0.05, k_nn=5, normed_lap=False, seed=None):
    print("Loading the Gaussian Cluster data with %d clusters..." % len(means))
    if len(Ns) != len(means):
        raise ValueError('Must have same number of means as clusters in Ns')
    if len(covs) != len(means):
        raise ValueError('Must have same number of means as covariance matrices')
    N = sum(Ns)
    classes = [i for i in range(len(Ns))]
    X, W, ground_truth = generate_gauss_clus_graphs(Ns, means, covs, k_nn=k_nn)
    evals, evecs = get_eig_Lnorm(W, normed_=normed_lap)

    indices = np.array(list(range(N)))
    fid = {}
    np.random.seed(seed)
    for i in classes:
        i_mask = indices[ground_truth ==i]
        np.random.shuffle(i_mask)
        n_i = len(i_mask)
        fid[i] = list(i_mask[:int(sup_percent*n_i)])

    return Data_obj(X, evals, evecs, fid, ground_truth)

def load_MNIST(digits=[1,4,7,9], num_points=4*[500], num_eig=300, Ltype='n', sup_percent=0.05, k_nn=15, seed=10, full=True):
    print("Loading the MNIST data with digits %s ..." % str(digits))
    if len(digits) != len(num_points):
        raise ValueError('Length of digits and num_points must be the same')

    # filepath and filename creation for checking if this data has already been computed before
    filename = "".join([str(d)+"_" for d in digits])
    filename += "".join([str(n)+"_" for n in num_points])
    filename += "%d_%s_%d.npz" % (num_eig, Ltype, k_nn)
    file_path = "./data/MNIST/"

    if not os.path.exists(file_path):
        print('Folder for MNIST not already created, creating...')
        os.mkdir(file_path)

    if os.path.isfile(file_path + filename):
        print('Found MNIST data already saved\n')
        mnist = np.load(file_path+filename)
        X, evals, evecs, ground_truth = mnist['X'], mnist['evals'], mnist['evecs'], mnist['ground_truth']
        np.random.seed(seed) # set the random seed to allow for consistency in choosing fidelity point
        fid = {}
        for i in np.unique(ground_truth).astype(int):
            i_ind = np.where(ground_truth == i)[0]
            np.random.shuffle(i_ind)
            fid[i] = list(i_ind[:int(sup_percent*num_points[i])])

        del mnist

    else:
        print("Couldn't find already saved MNIST data for the given setup, downloading from http://yann.lecun.com/exdb/mnist/")

        """ Loading data from MNIST website"""
        resp = requests.get('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz').content
        file = BytesIO(resp)
        f = gzip.open(file, 'r')
        f.read(16)
        num_images = 60000  # read ALL the images
        img_size = 28
        buf = f.read(img_size * img_size * num_images)
        imgs = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        imgs = imgs.reshape(num_images, img_size * img_size)

        resp = requests.get('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz').content
        file = BytesIO(resp)
        f = gzip.open(file, 'r')
        f.read(8)
        buf = f.read(num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)

        """ Processing to get the desired subset of digits"""
        dig_ind = []
        for j in range(len(digits)):
            d_ind = np.where(labels == digits[j])[0]
            np.random.shuffle(d_ind)
            dig_ind.extend(list(d_ind[:num_points[j]]))


        """ Define fid, ground_truth, and X datums"""
        X = imgs[dig_ind, :]
        labels_sub = labels[dig_ind]
        ground_truth = np.zeros(len(dig_ind))
        fid = {}
        if len(digits) == 2: # Binary case -- make the classes -1, 1
            m1_ind = np.where(labels_sub == digits[0])[0]
            ground_truth[m1_ind] = -1
            np.random.shuffle(m1_ind)
            fid[-1] = list(m1_ind[:int(sup_percent*num_points[0])])

            p1_ind = np.where(labels_sub == digits[1])[0]
            ground_truth[p1_ind] = 1
            np.random.shuffle(p1_ind)
            fid[1] = list(p1_ind[:int(sup_percent*num_points[1])])

        else:
            for i in range(len(digits)):
                i_ind = np.where(labels_sub == digits[i])[0]
                ground_truth[i_ind] = i
                np.random.shuffle(i_ind)
                fid[i] = list(i_ind[:int(sup_percent*num_points[i])])

        """ Create the similarity graph and calculate eigenval/vecs """
        W = make_sim_graph(X, k_nn=k_nn)
        if Ltype == 'n':
            evals, evecs = get_eig_Lnorm(W, num_eig=num_eig, normed_=True)
        else:
            evals, evecs = get_eig_Lnorm(W, num_eig=num_eig, normed_=False)

        # whether or not we will be keeping track of the FULL dataset
        if not full:
            X = None

        print('Saving MNIST data to %s' % (file_path+filename))
        # NOT SAVING fid, since will allow for different runs on fidelity
        np.savez(file_path + filename, X=X, evals=evals, evecs=evecs, ground_truth=ground_truth)

    # Reset the seed for random choices to be random, so as to allow for different runs
    np.random.seed(None)
    return Data_obj(X, evals, evecs, fid, ground_truth)








def load_gas_plume(num_eig=1000, Ltype='n', sup_percent=0.05, k_nn=15, seed=10):
    print("Loading the GasPlume data...")
    # filepath and filename creation for checking if this data has already been computed before
    filename = "%d_%s_%d.npz" % (num_eig, Ltype, k_nn)
    file_path = "./datasets/GasPlume/"

    if not os.path.exists(file_path):
        print('Folder for Gas Plume not already created, creating...')
        os.mkdir(file_path)

    if os.path.isfile(file_path + filename):
        print('Found Gas Plume data already saved\n')
        GP = np.load(file_path+filename)
        X, evals, evecs, ground_truth = GP['X'], GP['evals'], GP['evecs'], GP['ground_truth']
        del GP
        np.random.seed(seed) # set the random seed to allow for consistency in choosing fidelity point
        fid = {}
        for i in np.unique(ground_truth).astype(int):
            i_ind = np.where(ground_truth == i)[0]
            np.random.shuffle(i_ind)
            fid[i] = list(i_ind[:int(sup_percent*len(i_ind))])

    else:
        print("Didn't find already saved with parameter settings... calculating anew")
        if os.path.exists(file_path + "*.npz"):
            print('Using other graph construction for this specific param setting...')
        else:
            raise NotImplementedError('Havent implemented the capability to calculate all the data here')




    if os.path.isfile(file_path + filename):
        return Data_obj(X, evals, evecs, fid, ground_truth)
    else:
        pass

    return X, ground_truth


FILENAME = 'HUJI_data_tau_-2_K_40_Laplacian_n_Metric_Euclidean_numsample_400_.mat'

def load_HUJI(filepath='./datasets/HUJI/', sup_percent=0.1, seed=10):
    print("Loading the HUJI data...")
    huji = loadmat(filepath + FILENAME)
    evals, evecs, ground_truth, num_tr = huji['E'], huji['phi'], huji['ground_truth'], huji['ratio']
    num_tr = num_tr[0][0]
    ground_truth = ground_truth.flatten() - 1

    # Define the fidelity
    np.random.seed(seed)
    fid = {}
    for cl in np.unique(ground_truth):
        cl_ind = np.where(ground_truth == cl)[0]
        cl_ind = cl_ind[cl_ind <= num_tr]
        np.random.shuffle(cl_ind)
        fid[cl] = cl_ind[:int(sup_percent*len(cl_ind))]

    return Data_obj(None, evals.flatten(), evecs, fid, ground_truth)

def load_CITESEER(filepath='./datasets/CITESEER/', Ltype='n', num_eig=None, sup_percent=0.1, seed=1):
    if num_eig is None:
        num_eig = 2110
    print("Note that the current code here assumes you already have the file %s/CITESEER_%d_%s.npz already saved." % (filepath, num_eig, Ltype))
    data = np.load(filepath + 'CITESEER_%d_%s.npz' %(num_eig, Ltype))
    evals, evecs, ground_truth = data['evals'], data['evecs'], data['ground_truth']
    np.random.seed(seed)
    fid = {}
    for cl in np.unique(ground_truth):
        cl_ind = np.where(ground_truth == cl)[0]
        np.random.shuffle(cl_ind)
        fid[cl] = cl_ind[:int(sup_percent*len(cl_ind))]

    return Data_obj(None, evals, evecs, fid, ground_truth)


def load_voting_records(filepath='./datasets/VOTING-RECORD/',
                        Ltype='n', num_eig=None,
                        sup_percent=0.1, seed=1):
    filename = 'house-votes-84.data'
    # load the file directly into X format
    with open(filepath + filename, 'r') as f:
        lines = list(f)
        f.close()
    vote2num = {'y' : 1, 'n' : -1, '?' : 0} # maps y, n, ? to numerical values
    party2num = {'democrat' : 1, 'republican' : -1}
    X = [list(map(lambda x: vote2num[x], line.strip().split(',')[1:])) for line in lines]
    X = np.array(X)
    labels = [party2num[line.split(',')[0]] for line in lines]
    labels = np.array(labels)
    del lines
    # setup fid
    # -1 case
    fid = {}
    m1_ind = np.where(labels == -1)[0]
    np.random.shuffle(m1_ind)
    fid[-1] = list(m1_ind[:int(sup_percent*len(m1_ind))])
    # +1 case
    p1_ind = np.where(labels == 1)[0]
    np.random.shuffle(p1_ind)
    fid[1] = list(p1_ind[:int(sup_percent*len(p1_ind))])
    # setup similarity graphs etc
    W = make_sim_graph(X, k_nn=len(labels)-1) # fully connected
    if Ltype == 'n':
        evals, evecs = get_eig_Lnorm(W, num_eig=num_eig, normed_=True)
    else:
        evals, evecs = get_eig_Lnorm(W, num_eig=num_eig, normed_=False)
    # WIP
    return Data_obj(X, evals, evecs, fid, labels)


################# Graph generation and other calculations ######################

def sqdist(X, Y):
    # Michael Luo code  - X is (d x m) np array, Y is (d x n) np array
    # returns D_ij = |x_i - y_j|^2 matrix
    # better than KD tree for larger dimensions?
    m = X.shape[1]
    n = Y.shape[1]
    Yt = Y.T
    XX = np.sum(X*X, axis=0)
    YY = np.sum(Yt*Yt, axis=1).reshape(n,1)
    return np.tile(XX, (n,1)) + np.tile(YY, (1,m)) - 2*Yt.dot(X)

# Make the similarity graph from rows of X
def make_sim_graph(X, k_nn=5, sigma=1.):
    N = X.shape[0]
    # Make weighted similarity graph, in W
    D = sqdist(X.T,X.T)
    ind_knn = np.argsort(D, axis=1)[:,1:k_nn+1]
    Dknn = D[(np.arange(N).reshape(N,1),ind_knn)]

    I = np.tile(np.arange(N).reshape(N,1), (1,k_nn)).flatten()
    J = ind_knn.flatten()
    Dmean = (np.sum(Dknn, axis=1)/k_nn).reshape(N,1)
    w_sp = np.divide(Dknn, Dmean)
    w_sp = np.exp(-(w_sp * w_sp)/sigma)
    W = sps.csr_matrix((w_sp.flatten() , (I, J)), shape=(N,N))
    W = 0.5*(W+W.T)

    return W


# Gaussian clusters data
def generate_gauss_clus_graphs(Ns, means, Covs, k_nn=5, seed=None):
    '''
    inputs   :
      Ns    : class sizes, an array of length K
      means : class means, list of mean vectors in R^d
      Covs  : class covariances, list of Cov matrices of size d x d
    returns  : adjacency matrix
    '''
    N = sum(Ns)
    d = len(means[0])
    X = np.zeros((N, d))
    ground_truth = np.zeros(N)
    offset = 0
    np.random.seed(seed)
    for i in range(len(Ns)):
        Ni = Ns[i]
        X[offset:offset+Ni,:] = np.random.multivariate_normal(means[i], Covs[i], Ni)
        ground_truth[offset: offset+Ni] = i
        offset += Ni


    # Make weighted similarity graph, in W
    W = make_sim_graph(X, k_nn)
    return X, W, ground_truth


def get_eig_Lnorm(W, num_eig=None, normed_=True, return_L=False):
    # If num_eig is None, we will calculate all eigenvalues
    if num_eig is None:
        num_eig = W.shape[0]
    print('Creating Laplacian')
    L = sps.csgraph.laplacian(W, normed=normed_)
    print('Finished making Laplacian, now calculating the eval/evecs')
    if num_eig > int(W.shape[0]/2):
        print('Converting to dense since computing more than half of eigenvectors...')
        w, v = sla.eigh(L.toarray(), eigvals=(0,num_eig-1))
    else:
        w, v = sps.linalg.eigsh(L, k=num_eig, which='SM')
    if return_L:
        return w, v, L.toarray()
    return w, v
