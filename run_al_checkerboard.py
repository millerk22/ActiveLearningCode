from util.al_util import *
from util.Graph_manager import Graph_manager
from sklearn.datasets import make_moons, load_breast_cancer
from sklearn.decomposition import PCA
import argparse
from util.acquisition import *
from util.acquisition_visualize import ActiveLearningAcquisition as ALA
from datasets.data_loaders_mlflow import load_checkerboard
import os
import time


def plot_criterion(vals, k, unlabeled, labeled, X, title='Model Change', **kwargs):
    if type(vals) == type([]):
        vals = np.array(vals)
    if title[:3] == 'MBR':
        vals *= -1
    valsn = (vals - np.min(vals))/(np.max(vals) - np.min(vals))
    colors = [(1.-x, 0.5, x) for x in valsn]
    plt.scatter(X[unlabeled, 0], X[unlabeled,1], marker='o', c=colors)
    plt.scatter(X[labeled, 0], X[labeled, 1], marker='^', c='g', s=60, label='labeled')
    plt.scatter(X[k,0], X[k,1], marker='x', c='k', s=70, label='AL choice')
    plt.legend()
    plt.title(title)

    if 'saveloc' in kwargs.keys():
        plt.savefig(kwargs['saveloc'] + ''.join(title.split()) + '.png')
    if 'subplot' in kwargs.keys():
        if not kwargs['subplot']:
            plt.show()
    else:
        plt.show()
    return

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description = 'Run Active Learning ')
    #argparser.add_argument('--Ltype', default='normed', type=str, help='Type of Graph Laplacian matrix')
    argparser.add_argument('--tau', default=0.1, type=float, help='tau parameter in Gaussian prior')
    argparser.add_argument('--gamma', default=0.1, type=float, help='gamma parameter in probit likelihood')
    argparser.add_argument('--subplot', default=True, type=str2bool, help='option to fit all plots onto same figure')
    argparser.add_argument('--saveloc', default='./figures/p-gr-hf-comp/', type=str, help='file location to store AL plots')
    argparser.add_argument('--init_plot', default=True, type=str2bool, help='option to show initial classification results plot')
    argparser.add_argument('--seed', default=42, type=int, help='seed for reproducibility of results')
    argparser.add_argument('--n_start', default=10, type=int, help='number of labeled points to begin with')
    args= argparser.parse_args()


    if not os.path.isdir(args.saveloc):
        os.mkdir(args.saveloc)


    # Setup Dataset
    N = 1500
    X, labels = load_checkerboard({'num_points':N, 'seed':args.seed})
    labels[np.where(labels == 0)] = -1
    ind_ord = list(np.where(labels == -1)[0]) + list(np.where(labels == 1)[0])
    ind_ordnp = np.array(ind_ord)

    # Choose labeled points
    n_start = args.n_start
    np.random.seed(args.seed)
    labeled =  list(np.random.choice(np.where(labels == -1)[0],  size=n_start//2, replace=False))
    labeled += list(np.random.choice(np.where(labels ==  1)[0], size=n_start//2, replace=False))
    unlabeled = list(filter(lambda x: x not in labeled, range(N)))


    # create the vector H^Ty, already in N -dim!
    y = np.zeros(N)
    y[labeled] = labels[labeled]
    m_hf_full = y.copy()
    m_hf_full[np.where(y == -1)] = 0.
    y_hf = m_hf_full[labeled]


    # Create similarity graphs -- Normalized and Unnormalized GL
    neig = None
    knn_ = 15
    graph_params = {
        'knn'    : knn_,
        'sigma'  : 3.,
        'Ltype'  : 'normed',
        'n_eigs' : neig,
        'zp_k'   : 5
    }

    gm = Graph_manager()
    wn, vn = gm.from_features(X, graph_params, debug=True)

    graph_params['Ltype'] = 'unnormalized'
    w, v = gm.from_features(X, graph_params, debug=True)




    # Construct prior covariance and precision matrix for Unnormalized
    tau, gamma = args.tau, args.gamma
    d = (tau ** (2.)) * ((w + tau**2.) ** (-1.))  # # unnormalized
    Ct = v @ sp.sparse.diags(d, format='csr') @ v.T
    Lt = v @ sp.sparse.diags(1./d, format='csr') @ v.T
    dn = (tau ** (2.)) * ((wn + tau**2.) ** (-1.))  # normalized
    Ctn = vn @ sp.sparse.diags(dn, format='csr') @ vn.T
    Ltn = vn @ sp.sparse.diags(1./dn, format='csr') @ vn.T



    # Probit model MAP estimator and posterior covariance
    m = probit_map_dr(labeled, labels[labeled], gamma, Ctn) # normalized
    H = Hess(m, labels[labeled], labeled, Ltn, gamma)
    C = sp.linalg.inv(H)


    # Gaussian Regression model mean and posterior covariance
    C_gr = get_init_post(Lt, labeled, gamma**2.) # unnormalized
    m_gr = (C_gr @ y)/(gamma**2.)
    C_grn = get_init_post(Ltn, labeled, gamma**2.)  #normalized
    m_grn = (C_grn @ y)/(gamma**2.)

    # Harmonic Function model mean and posterior covariance
    # C_hf = np.linalg.inv(tau**2. * Lt[np.ix_(unlabeled, unlabeled)])  # remove the extra scaling in the front of Lt to exactly follow HF
    # m_hf = -C_hf@ (tau**2. * Lt)[np.ix_(unlabeled, labeled)] @ y_hf
    C_hf = np.linalg.inv(Lt[np.ix_(unlabeled, unlabeled)])  # remove the extra scaling in the front of Lt to exactly follow HF
    m_hf = -C_hf@ (Lt)[np.ix_(unlabeled, labeled)] @ y_hf
    m_hf_full[unlabeled] = m_hf

    if args.init_plot:
        # print('Dataset')
        # plot_iter(labels, X, labels, labeled, title='Dataset (no classifier)')

        print('Showing plots of different classifiers')
        print('probit (normalized GL)')
        plt.subplot(2,2,1)
        plot_iter(m, X, labels, labeled, title='Probit (normalized G.L.)', subplot=True)
        print('gr (normalized GL)')
        plt.subplot(2,2,3)
        plot_iter(m_grn, X, labels, labeled, title='GR (normalized GL)', subplot=True)
        print('gr (unnormalized GL)')
        plt.subplot(2,2,2)
        plot_iter(m_gr, X, labels, labeled, title='GR (unnormalized GL)', subplot=True)

        print('hf (unnormalized GL, w/o front scaling)')
        plt.subplot(2,2,4)
        plot_iter(2.*m_hf_full - 1., X, labels, labeled, title='HF (unnormalized GL)', subplot=True)
        plt.suptitle('Classification Visualization')
        plt.savefig(args.saveloc + 'class-visual-checkerboard')
        plt.show()


        print("Showing the MAP estimators of the different models")
        plt.subplot(1,2,1)
        plt.scatter(range(N), m[ind_ord], marker='.', s=20, label='probit')
        plt.scatter(range(N), m_grn[ind_ord], marker='^', s=10, label='GR n')
        plt.legend()
        plt.subplot(1,2,2)
        plt.scatter(range(N), m_gr[ind_ord], c='r', marker='x', s=10, label='GR u')
        plt.scatter(range(N), 2.*m_hf_full[ind_ord] - 1., c='g', marker='o', s=10, label='HF')
        plt.legend()
        plt.suptitle('MAP Estimator Visualization')
        plt.savefig(args.saveloc + 'MAP-visual-checkerboard')
        plt.show()




    if args.subplot:
        plt.figure(figsize=(12,6))


    # Active Learning Visualization of different criterion
    ala = ALA()


    ofs = 0
    for i, f in enumerate([ala.vopt_hfv,  ala.mbr_hfv,  ala.sopt_hfv]):
        tic = time.clock()
        k, vals = f(C=C_hf, m=m_hf, y=y)
        toc = time.clock()
        print("Harmonic function %d took %f ms" % (i, 1000*(toc - tic)))
        k = unlabeled[k] # need to get the proper index in unlabeled since this criterion only considers the unlabeled submatrix
        if args.subplot:
            if i == 2:
                plt.subplot(3,4,4*ofs+i+2)
            else:
                plt.subplot(3,4,4*ofs+i+1)
            plot_criterion(vals, k, unlabeled, labeled, X, title=ala.get_name(), subplot=args.subplot)
        else:
            pass
            #plot_criterion(vals, k, unlabeled, labeled, X, title=ala.get_name(), saveloc=args.saveloc)

    ofs = 1
    for i, f in enumerate([ala.vopt_grv,  ala.mbr_grv, ala.modelchange_grv, ala.sopt_grv]):
        tic = time.clock()
        k, vals = f(C=C_grn, unlabeled=unlabeled, gamma=gamma, m=m_grn, y=y)
        toc = time.clock()
        print("GR %d took %f ms" % (i, 1000*(toc - tic)))
        if args.subplot:
            plt.subplot(3,4,4*ofs+i+1)
            plot_criterion(vals, k, unlabeled, labeled, X, title=ala.get_name(), subplot=args.subplot)
        else:
            pass
            #plot_criterion(vals, k, unlabeled, labeled, X, title=ala.get_name(), saveloc=args.saveloc)


    ofs = 2
    for i, f in enumerate([ala.vopt_pv, ala.mbr_pv, ala.modelchange_pv]):
        tic = time.clock()
        k, vals = f(C=C, unlabeled=unlabeled, gamma=gamma, m=m)
        toc = time.clock()
        print("Probit %d took %f ms" % (i, 1000*(toc - tic)))
        if args.subplot:
            plt.subplot(3,4,4*ofs+i+1)
            plot_criterion(vals, k, unlabeled, labeled, X, title=ala.get_name(), subplot=args.subplot)
        else:
            pass
            #plot_criterion(vals, k, unlabeled, labeled, X, title=ala.get_name(), saveloc=args.saveloc)



    if args.subplot:
        plt.suptitle('Active Learning Plots')
        plt.savefig(args.saveloc + 'active-learning-checkerboard')
        plt.show()
