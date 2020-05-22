from util.al_util import *
from util.Graph_manager import Graph_manager
from sklearn.datasets import make_moons
import argparse
from util.acquisition import *
from util.acquisition_visualize import ActiveLearningAcquisition as ALA
import os


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
        plt.savefig(saveloc + ''.join(title.split()) + '.png')
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
    argparser.add_argument('--subplot', default=False, type=str2bool, help='option to fit all plots onto same figure')
    argparser.add_argument('--saveloc', default='./', type=str, help='file location to store AL plots')
    argparser.add_argument('--init_plot', default=True, type=str2bool, help='option to show initial classification results plot')
    args= argparser.parse_args()


    # Setup graph
    # 2 moons
    N = 1000
    X, labels = make_moons(N, noise=0.15)
    labels[np.where(labels == 0)] = -1
    ind_ord = list(np.where(labels == -1)[0]) + list(np.where(labels == 1)[0])
    ind_ordnp = np.array(ind_ord)


    n_start = 5
    ans = 'n'
    while ans != 'y':
        labeled = list(np.random.choice(list(range(N)), size=n_start, replace=False))
        while sum(labels[labeled]) == n_start or sum(labels[labeled]) == 0:
            print('Rechoosing to get both clusters')
            labeled = list(np.random.choice(list(range(N)), size=n_start, replace=False))

        unlabeled = list(filter(lambda x: x not in labeled, range(N)))
        plot_iter(labels, X, labels, labeled)

        print("Is this setup acceptable? Please input 'y' or 'n':")
        ans = input()

    y = np.zeros(N)
    y[labeled] = labels[labeled]

    # Create similarity graph -- Normalized GL
    neig = None
    knn_ = 15
    graph_params = {
        'knn'    : knn_,
        'sigma'  : 3.,
        'Ltype'  : 'normed',
        'n_eigs' : neig,
        'zp_k'   : None
    }
    gm = Graph_manager()
    w, v = gm.from_features(X, graph_params, debug=True)


    # Construct prior covariance and precision matrix
    tau, gamma = args.tau, args.gamma
    d = (tau ** (2.)) * ((w + tau**2.) ** (-1.))
    Ct = v @ sp.sparse.diags(d, format='csr') @ v.T
    Lt = v @ sp.sparse.diags(1./d, format='csr') @ v.T



    m = probit_map_dr2(labeled, labels[labeled], gamma, Ct)
    H = Hess2(m, labels[labeled], labeled, Lt, gamma)
    C = sp.linalg.inv(H)

    if args.init_plot:
        plot_iter(m, X, labels, labeled)

    if args.subplot:
        plt.figure(figsize=(10,12))
    # Active Learning Visualization of different criterion
    ala = ALA()
    for i, f in enumerate([ala.vopt_grv, ala.vopt_pv, ala.mbr_grv, ala.mbr_pv, \
                    ala.modelchange_grv, ala.modelchange_pv, ala.sopt_grv]):
        k, vals = f(C=C, unlabeled=unlabeled, gamma=gamma, m=m, y=y)
        if args.subplot:
            plt.subplot(4,2,i+1)
            plot_criterion(vals, k, unlabeled, labeled, X, title=ala.get_name(), subplot=args.subplot)
        else:
            plot_criterion(vals, k, unlabeled, labeled, X, title=ala.get_name(), saveloc=args.saveloc)

    if args.subplot:
        plt.suptitle('Normalized Graph Laplacian Plots')
        plt.savefig(args.saveloc + 'active-learning-2moons-n')
        plt.show()



    # Unnormalized Graph Laplacian calculations
    graph_params = {
        'knn'    : knn_,
        'sigma'  : 3.,
        'Ltype'  : 'unnormalized',
        'n_eigs' : neig,
        'zp_k'   : None
    }
    gm = Graph_manager()
    w, v = gm.from_features(X, graph_params, debug=True)


    # Construct prior covariance and precision matrix
    tau, gamma = args.tau, args.gamma
    d = (tau ** (2.)) * ((w + tau**2.) ** (-1.))
    Ct = v @ sp.sparse.diags(d, format='csr') @ v.T
    Lt = v @ sp.sparse.diags(1./d, format='csr') @ v.T



    m = probit_map_dr2(labeled, labels[labeled], gamma, Ct)
    H = Hess2(m, labels[labeled], labeled, Lt, gamma)
    C = sp.linalg.inv(H)

    if args.init_plot:
        plot_iter(m, X, labels, labeled)
    if args.subplot:
        plt.figure(figsize=(10,12))
    # Active Learning Visualization of different criterion
    ala = ALA()
    for i, f in enumerate([ala.vopt_grv, ala.vopt_pv, ala.mbr_grv, ala.mbr_pv, \
                    ala.modelchange_grv, ala.modelchange_pv, ala.sopt_grv]):
        k, vals = f(C=C, unlabeled=unlabeled, gamma=gamma, m=m, y=y)
        if args.subplot:
            plt.subplot(4,2,i+1)
            plot_criterion(vals, k, unlabeled, labeled, X, title=ala.get_name(), subplot=args.subplot)
        else:
            plot_criterion(vals, k, unlabeled, labeled, X, title=ala.get_name(), saveloc=args.saveloc)

    if args.subplot:
        plt.suptitle('Unnormalized Graph Laplacian Plots')
        plt.savefig(args.saveloc + 'active-learning-2moons-u')
        plt.show()
    #os.system("say 'done with test'")
