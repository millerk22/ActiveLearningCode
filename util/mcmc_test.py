from mcmc_classes import *
from datasets.dataloaders import *
from util.util import *
import numpy as np
from IPython import embed
import sys
import time
import argparse



def test2moons(show_plot=False, run_ipy=False):
    print("Running 2 moons test of MCMC sampler, Gaussian Regression")
    # load the 2 moons data, plot initial distribution of labeled and unlabeled
    data = load_2_moons()
    print(data.evals.shape)
    if show_plot:
        data.plot_initial()


    # Create the GR sampler, default params : gamma = 0.001, tau = 0.01, alpha = 1.0
    GR_sampler = Gaussian_Regression_Sampler()
    GR_sampler.load_data(data)

    # Get 100 samples (i.e. calculate the posterior mean and covariance, then sample)
    GR_sampler.run_sampler(10000)
    print(GR_sampler.u_mean.shape)
    print(GR_sampler.u_t_mean.shape)

    # embed the current namespace in an iPython session in the terminal
    if run_ipy:
        embed()

    return

def testG3_GR(show_plot=False, run_ipy=False):
    print("Running Gaussian Cluster test of MCMC sampler, Gaussian Regression")
    # load Gaussian Clusters data, plot initial distribution of labeled and unlabeled
    Ns = [100,200,100]
    mus = [np.array([1., 0.]), np.array([-1., 0.]), np.array([0., 1.5])]
    sigma = 0.15
    Covs = [sigma*np.eye(2) for j in range(len(mus))]
    data = load_gaussian_cluster(Ns, mus, Covs)
    if show_plot:
        data.plot_initial()

    # Create the GR sampler, default params : gamma = 0.001, tau = 0.01, alpha = 1.0
    GR_sampler = Gaussian_Regression_Sampler()
    GR_sampler.load_data(data)

    # Get 100 samples (i.e. calculate the posterior mean and covariance, then sample)
    GR_sampler.run_sampler(1000)
    print(GR_sampler.u_mean[:5,:])
    print(GR_sampler.m[:5,:])
    print(GR_sampler.u_t_mean[:5,:])

    # embed the current namespace in an iPython session in the terminal
    if run_ipy:
        embed()

    return

def testG3_GPS(show_plot=False, run_ipy=False):
    print("Running Gaussian Cluster test of MCMC sampler, Gibbs Probit")
    # load Gaussian Clusters data, plot initial distribution of labeled and unlabeled
    Ns = [100,200,100]
    mus = [np.array([1., 0.]), np.array([-1., 0.]), np.array([0., 1.5])]
    sigma = 0.15
    Covs = [sigma*np.eye(2) for j in range(len(mus))]
    data = load_gaussian_cluster(Ns, mus, Covs)
    if show_plot:
        data.plot_initial()

    # Create the GR sampler, default params : gamma = 0.001, tau = 0.01, alpha = 1.0
    GPS = Gibbs_Probit_Sampler()
    GPS.load_data(data)

    # Get 100 samples (i.e. calculate the posterior mean and covariance, then sample)
    tic = time.process_time()
    GPS.run_sampler(1000)
    print('Sampling took %f seconds ' % (time.process_time() - tic))

    # embed the current namespace in an iPython session in the terminal
    if run_ipy:
        embed()

    return


def testMNIST(run_ipy=False):

    mnist = load_MNIST()
    mnist.get_useful_structs()

    """
    print('\n\nRunning Gaussian Regression Sampler')
    GR = Gaussian_Regression_Sampler()
    GR.load_data(mnist)

    print('\nRunning Sampling...')
    tic = time.process_time()
    GR.run_sampler(1100)
    print('\tGR Sampling took %f seconds' % (time.process_time() - tic))
    gr_acc_u, gr_acc_u_t = GR.comp_mcmc_stats()
    print("Accuracy of GR: acc_u = %f, acc_u_t = %f" % (gr_acc_u, gr_acc_u_t))
    print("Accuracy of GR: acc_m = %f" % GR.acc_m)
    """

    print('Running MNIST test with Gibbs_Probit_Sampler\n')
    GPS = Gibbs_Probit_Sampler()
    GPS.load_data(mnist)

    print('\nRunning Sampling...')
    tic_ = time.process_time()
    GPS.run_sampler(10000, burnIn=5000)
    print('\tGPS Sampling took %f seconds' % (time.process_time() - tic_))
    gps_acc_u, gps_acc_u_t = GPS.comp_mcmc_stats()
    print("Accuracy of GPS: acc_u = %f, acc_u_t = %f" % (gps_acc_u, gps_acc_u_t))


    if run_ipy:
        embed()



def testGP_GR(run_ipy=False):
    gas_data = load_gas_plume()
    gas_data.get_useful_structs()

    GR = Gaussian_Regression_Sampler()
    GR.load_data(gas_data)

    num_samples = 1500
    print("Running GR Sampler for %d points on Gas Plume data" % num_samples)
    tic =time.process_time()
    GR.run_sampler(1500)
    toc = time.process_time()
    print('\tSampling took %f seconds' % (toc - tic))

    if run_ipy:
        embed()

def test_pCn(run_ipy=False):
    beta = 0.3
    print("Preparing pCN_*_Sampler test with 2 moons data, beta = %f" % beta)
    data = load_2_moons()
    #print("Preparing pCN_*_Sampler test with MNIST data, beta = %f" % beta)
    #data = load_MNIST(digits=[4,9], num_points=[1000,1000], sup_percent=0.05)


    pcnprob = pCN_Probit_Sampler(beta=beta)
    #pcnprob = pCN_BLS_Sampler(beta=beta)
    pcnprob.load_data(data)


    GAMMAS = [2.*0.1**j for j in range(4)]
    for gamma in GAMMAS[::-1]:
        print(len(data.labeled))
        pcnprob.gamma = gamma
        print('gamma = %f' % gamma)
        print('Running sampling...')
        pcnprob.run_sampler(4000, burnIn=2000)
        print('Sampling finished, calculating statistics...')

        acc_u, acc_u_t = pcnprob.comp_mcmc_stats()
        print("Accuracy of pCN Probit: acc_u = %f, acc_u_t = %f" % (acc_u, acc_u_t))
        #print("Accuracy of pCN BLS: acc_u = %f, acc_u_t = %f" % (acc_u, acc_u_t))
        print()
        #pcnprob.plot_u(pcnprob.u_mean)

    if run_ipy:
        embed()


def test_pCn2(run_ipy=False):
    beta = 0.3
    print("Preparing pCN_*_Sampler test with 2 moons data, beta = %f" % beta)
    data = load_2_moons(seed=20)
    #print("Preparing pCN_*_Sampler test with MNIST data, beta = %f" % beta)
    #data = load_MNIST(digits=[4,9], num_points=[1000,1000], sup_percent=0.05)


    #pcnprob = pCN_Probit_Sampler(beta=beta, tau=0.1, alpha=1.5)
    pcnprob = pCN_BLS_Sampler(beta=beta, tau=0.1, alpha=1.5)
    pcnprob.load_data(data)
    print('Running sampling...')
    pcnprob.run_sampler(4000, burnIn=2000, seed=10)
    print('Sampling finished, calculating statistics...')
    acc_u, acc_u_t = pcnprob.comp_mcmc_stats()
    print("Accuracy of %s: acc_u = %f, acc_u_t = %f" % (str(pcnprob), acc_u, acc_u_t))
    print()
    print('Running sampling...')
    pcnprob.run_sampler(4000, burnIn=2000, seed=10)
    print('Sampling finished, calculating statistics...')
    acc_u, acc_u_t = pcnprob.comp_mcmc_stats()
    print("Accuracy of %s: acc_u = %f, acc_u_t = %f" % (str(pcnprob), acc_u, acc_u_t))
    if run_ipy:
        embed()


def test_GProb(run_ipy=False):
    print("Preparing Gibbs-Probit comparison test with MNIST data")
    #data = load_MNIST()
    data = load_2_moons()
    plt.plot(np.arange(len(data.evals)), data.evals)
    plt.title('Evals')
    plt.show()

    gprob2 = Gibbs_Probit_Sampler(gamma=0.1)
    gprob2.load_data(data)
    gprob2.run_sampler(2000, burnIn=1000)

    acc_u, acc_u_t = gprob2.comp_mcmc_stats()
    print("Accuracy of Gibbs-Probit: acc_u = %f, acc_u_t = %f" % (acc_u, acc_u_t))
    print()

    if run_ipy:
        embed()

def test_GProb2(run_ipy=False):
    print("Preparing Gibbs-Probit comparison test with MNIST data")
    data = load_MNIST()

    gprob = Gibbs_Probit_Sampler(gamma=0.1, tau=0.1, alpha=1.5)
    gprob.load_data(data)
    gprob.run_sampler(2000, burnIn=1000, seed=10)

    acc_u, acc_u_t = gprob.comp_mcmc_stats()
    print("Accuracy of Gibbs-Probit: acc_u = %f, acc_u_t = %f" % (acc_u, acc_u_t))
    print()

    gprob.run_sampler(2000, burnIn=1000, seed=10)

    acc_u, acc_u_t = gprob.comp_mcmc_stats()
    print("Accuracy of Gibbs-Probit: acc_u = %f, acc_u_t = %f" % (acc_u, acc_u_t))
    print()


    if run_ipy:
        embed()


def test_HUJI(run_ipy=False):
    print("Preparing Gibbs-Probit comparison test with HUJI data")
    data = load_HUJI()
    gprob = Gibbs_Probit_Sampler(gamma=0.1)
    gprob.load_data(data)
    gprob.run_sampler(1000, burnIn=500)

    acc_u, acc_u_t = gprob.comp_mcmc_stats()
    print("Accuracy of Gibbs-Probit: acc_u = %f, acc_u_t = %f" % (acc_u, acc_u_t))
    print()

    if run_ipy:
        embed()


def test_citeseer(run_ipy):
    cite = load_CITESEER()





if __name__ == "__main__":

    # Test script command line parameter parsing
    show_plot = False
    run_ipy = False
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument('--show')
        parser.add_argument('--embed')
        args = parser.parse_args()
        if args.show is not None:
            show_plot = int(args.show)
        if args.embed is not None:
            run_ipy = int(args.embed)



    #test2moons(show_plot, run_ipy)
    #testG3_GR(show_plot, run_ipy)
    #testG3_GPS(show_plot, run_ipy)
    testMNIST(run_ipy)
    #test_pCn(run_ipy)
    #test_pCn2(run_ipy)
    #test_GProb2(run_ipy)
    #test_HUJI(run_ipy)
