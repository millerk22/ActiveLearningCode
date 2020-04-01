import numpy as np
import matplotlib.pyplot as plt
from util.util import calc_GR_posterior, threshold1D, threshold2D
from util.util import threshold2D_avg, threshold1D_avg, SummaryStats
import scipy.sparse as sps
from scipy.linalg import eigh
from scipy.stats import norm
import emcee
import time
import matlab.engine
from util.trnm import TruncRandNormMulticlass


class MCMC_Sampler(object):
    """ Base MCMC Sampler object. The different classes that inherit from this base
    class are: Gaussian_Regression_Sampler, Gibbs_Probit_Sampler, pCN_Probit_Sampler,
    and pCN_BLS_Sampler. Each model has posterior distribution that is proportional
    to:
        P(u | y) ~ exp(-1/2 (<u, (L + tau^2)^alpha u> + Phi(u; y, gamma)))

        where Phi(u; y) depend on the noise model we choose (i.e. the different
        sampler classes).

    MCMC_Sampler basics (see individual member functions for use of functions)
        - MCMC_Samplers are instantiated with model parameters
    gamma (noise param), tau (Laplacian shift), alpha (Laplacian power exponent).
        - load_data() : MCMC_Sampler objects "load in" data objects (Data_obj class), that contain the
    spectral properties of the respective data set we will be using to run our sampling
    on.
        - run_sampler(): According to the sampler type, obtain MCMC samples, keeping
        track of the mean of the samples. Also keep track of the mean of a function of
        the samples (default is thresholding/sign function)
        - comp_mcmc_stats() : after sampling has been done, calculates useful statistics
        of the sampling run, stored in a SummaryStats object.
        - active_learning_choices() : NOT IMPLEMENTED YET.
    """
    def __init__(self, gamma, tau=0., alpha=1.):
        self.gamma = gamma
        self.gamma2 = gamma**2.
        self.tau = tau
        self.alpha = alpha
        self.stats = None
        self.Data = None
        self.sum_stats = SummaryStats()
        self.sum_stats_t = SummaryStats()
        self.name = "MCMC"

    def load_data(self, Data, plot_=False):
        self.Data = Data
        self.evals_mod = np.power(self.Data.evals + self.tau**2., self.alpha)
        if not self.Data.have_useful:
            self.Data.get_useful_structs()
        if plot_:
            self.Data.plot_initial()


    def run_sampler(self, num_samples, burnIn=0):
        if self.Data is None:
            raise ValueError("No data has been loaded. First load data via member function load_data(Data)")
        self.num_samples = num_samples
        self.burnIn = burnIn
        self.print_ = False
        if num_samples+burnIn > 1000:
            self.print_ = True


    def comp_mcmc_stats(self, return_acc=True):
        print("Computing summary statistics...")

        if self.Data is None:
            raise ValueError('No Data object loaded yet...')
        if self.u_mean is None:
            raise ValueError('Have not sampled yet... need to run sampler to obtain stats.')

        # Get class labelings of computed means, for use in summary stats computation
        # (this gives a 1D np array with computed class labeling)
        if -1 in self.Data.classes:
            u_mean_t = threshold1D(self.u_mean.copy())
            u_t_mean_t = threshold1D(self.u_t_mean.copy())
        elif 0 in self.Data.classes and len(self.u_mean.shape) == 1:
            u_mean_t = threshold1D(self.u_mean.copy(), True)
            u_t_mean_t = threshold1D(self.u_t_mean.copy(), True)
        else:
            u_mean_t = threshold2D(self.u_mean.copy(), False)
            u_t_mean_t = threshold2D(self.u_t_mean.copy(), False)

        self.sum_stats.compute(self.Data.ground_truth, u_mean_t, self.Data.N, self.Data.num_class)
        self.sum_stats_t.compute(self.Data.ground_truth, u_t_mean_t, self.Data.N, self.Data.num_class)


        """ Should we return accuracy always? """
        if return_acc:
            return self.sum_stats.acc, self.sum_stats_t.acc
        return


    def active_learning_choices(self, method, num_to_label):
        pass



    def plot_u(self, u):
        """ Note this plotting assumes that the input u is 1D numpy array. So,
        if our problem is multiclass, you must be passing only one column to this
        plotting function.
        """
        plt.scatter(np.arange(self.Data.N), u)
        plt.scatter(self.Data.labeled, u[self.Data.labeled], c='g')
        plt.title('Plot of u')
        plt.show()

    def __str__(self):
        return "%s(gamma=%1.3f, tau=%1.3f, alpha=%1.3f)" % (self.name, self.gamma, self.tau, self.alpha)








class Gaussian_Regression_Sampler(MCMC_Sampler):
    """
    Gaussian Regression model:
        Phi(u; y, gamma) = 1/gamma^2 * ||Hu - y||^2
            - H : Z -> Z'. projection onto labeled nodes
            - norm is 2 norm for binary classification, and Frobenius for multiclass

        In this case the prior and posterior are conjugate, since all are Gaussian.
        Have closed form solution for the posterior mean, m, and covariance, C.

        m = 1/gamma^2 C H^T y, C = ((L + tau^2I)^alpha + 1/gamma^2 B)^(-1)
            - (B = H^T H)

        Though closed for solution for mean is available, still do sampling to see how mean of
        thresholded samples computes.


        Note that the mean and mode (corresponding to the MAP estimator) align for this
        posterior.
    """
    def __init__(self, gamma=0.01, tau=0.01, alpha=1.0):
        MCMC_Sampler.__init__(self, gamma, tau, alpha)
        self.name = "Gaussian_Regression"

    def load_data(self, Data, plot_=False):
        MCMC_Sampler.load_data(self, Data, plot_)


    def run_sampler(self, num_samples, f='thresh'):
        """
        Run MCMC sampling for the loaded dataset.
        Inputs:
            num_samples : int , number of desired samples from this sampling.
                (Note for GR here no burnIn needed)
            f : str 'thresh' or function handle, samples will compute values related to
                E[f(u)].

        Outputs:
            Saves to the Sampler object:
                self.samples : (N x num_class x num_samples) numpy array of pre-thresholded samples
                self.u_mean : (N x num_class) numpy array of pre-thresholded sample mean
                self.v_mean : (N x num_class) numpy array of thresholded sample mean
                self.m : (N x num_class) numpy array analytical mean (GR-special)
                self.C : (N x N) numpy array analytical covariance operator (GR-special)
                self.y : (N' x num_class) numpy array of labelings on labeled set

        Note that for the binary case, we have {-1,+1} classes and the entries of self.v_mean
        represent the empirical probability of being class +1 from the samples.
        """
        MCMC_Sampler.run_sampler(self, num_samples)
        print("Running Gaussian Regression sampling to get %d samples, with no burnIn samples" % num_samples)
        ## run Gaussian Regression method here -- ignoring burnIn
        self.m, self.C, self.y = calc_GR_posterior(self.Data.evecs, self.Data.evals, self.Data.fid,
                                self.Data.labeled, self.Data.unlabeled, self.tau, self.alpha, self.gamma2)

        # binary class case
        if -1 in self.Data.classes:
            samples = np.random.multivariate_normal(self.m, self.C, num_samples).T
            self.u_mean = np.average(samples, axis=1)
            if f == 'thresh':
                self.u_t_mean = threshold1D_avg(samples)

        # multiclass sampling case
        else:
            samples = np.array([np.random.multivariate_normal(self.m[:,i], self.C,
                            self.num_samples).T for i in range(self.Data.num_class)]).transpose((1,0,2))
            self.u_mean = np.average(samples, axis=2)
            if f == 'thresh':
                self.u_t_mean = threshold2D_avg(samples)

        # delete the samples for sake of memory
        del samples
        return

    def comp_mcmc_stats(self):
        self.acc_u, self.acc_u_t = MCMC_Sampler.comp_mcmc_stats(self)
        """ In addition to the stats from the sampling from the posterior, calculate
        accuracy of thresholded analytic posterior mean, threshold*D(self.m)"""
        if -1 in self.Data.fid.keys():
            m_t = threshold1D(self.m)
        else:
            m_t = threshold2D(self.m, False)
        self.acc_m = len(np.where(m_t == self.Data.ground_truth)[0])/self.Data.N
        return self.acc_u, self.acc_u_t



class Gibbs_Probit_Sampler(MCMC_Sampler):
    """
    Probit Likelihood model:
    Phi(u; y, gamma) = - log(Psi(u_j(1) y_j(1); gamma)) - log(Psi(u_j(1) y_j(1); gamma))
                        ... -log(Psi(u_j(N') y_j(N'); gamma))
        where Psi(v; gamma) is CDF of normal distribution with standard deviation gamma
        and j(i) is the ith labeled node.

    Gibbs sampling iterates between sampling from truncated normal distributions
    on the labeled set, and Karhuenen-Loeve expansion for sampling based on the prior.
    """
    def __init__(self, gamma=0.01, tau=0.01, alpha=1.):
        MCMC_Sampler.__init__(self, gamma, tau, alpha)
        self.name = "Gaussian_Probit"

    def load_data(self, Data, plot_=False):
        MCMC_Sampler.load_data(self, Data, plot_)
        if -1 in self.Data.fid.keys():
            print("Noticed you gave Gibbs-Probit classes with -1. Converting to 0 for the current implementation...")
            self.Data.fid[0] = self.Data.fid[-1]
            del self.Data.fid[-1]
            self.Data.ground_truth[self.Data.ground_truth == -1] = 0.
            self.Data.classes = [0,1]
        # initiate the trandn_multiclass object
        self.TRNM = TruncRandNormMulticlass(self.Data.num_class)


    def run_sampler(self, num_samples, burnIn=0, f='thresh', seed=None):
        MCMC_Sampler.run_sampler(self, num_samples, burnIn)
        print("Running Gibbs-Probit sampling to get %d samples, with %d burnIn samples" % (num_samples, burnIn))

        # instantiate the u_mean (unthresh), u_t_mean (unthres) variables
        self.u_mean = np.zeros((self.Data.N, self.Data.num_class))
        self.u_t_mean = np.zeros_like(self.u_mean)

        # fixed initialization of u.
        u = np.zeros_like(self.u_mean)

        # fidv contains the indices of the fidelity nodes in the list "labeled"
        fidv = {}
        for c, ind in self.Data.fid.items():
            u[np.ix_(ind, len(ind)*[c])] = 1.
            fidv[c] = [self.Data.labeled.index(j) for j in ind]


        # sample Gaussian noise in batch to begin
        np.random.seed(seed)
        z_all = np.random.randn(len(self.Data.evals), self.Data.num_class, num_samples+burnIn)

        # Compute the projections for use in the KL expansion for sampling u | v

        V_KJ = self.Data.evecs[self.Data.labeled,:]
        P_KJ = V_KJ.T.dot(V_KJ)/self.gamma2
        for i in range(len(self.evals_mod)):
            P_KJ[i,i] += self.evals_mod[i]
        #P_KJ = 0.5*(P_KJ + P_KJ.T)  # do we need? seems to be symmetric already...
        S_KJ, Q_KJ = eigh(P_KJ)
        inv_skj = 1./np.sqrt(S_KJ)[:,np.newaxis]


        # Main iterations
        self.U = []
        for k in range(burnIn + num_samples):
            if self.print_ and k % 500 == 0:
                print('\tIteration %d of sampling...' % k)
            z_k = z_all[:,:,k] # the white noise samples for use in sampling u | v

            # Sample v ~ P(v | u).
            v = np.zeros((len(self.Data.labeled), self.Data.num_class))
            for cl, ind_cl in self.Data.fid.items():
                v[fidv[cl],:] = self.TRNM.gen_samples(u[ind_cl,:], self.gamma, cl)


            # Sample u ~ P(u | v) via KL expansion
            temp = V_KJ.T.dot(v)/self.gamma2
            temp = Q_KJ.T.dot(temp)
            temp /= S_KJ[:,np.newaxis]
            m_hat = Q_KJ.dot(temp)
            u_hat = Q_KJ.dot(z_k * inv_skj) + m_hat
            u = self.Data.evecs.dot(u_hat)


            # If past the burn in period, calculate the updates to the means we're recording
            if k > burnIn:
                self.U.append(u)
                k_rec = k - burnIn
                if k_rec == 1:
                    self.u_mean = u
                    if f == 'thresh':
                        self.u_t_mean = threshold2D(u.copy())
                else:
                    self.u_mean = ((k_rec-1) * self.u_mean + u)/k_rec
                    if f == 'thresh':
                        self.u_t_mean = ((k_rec-1) * self.u_t_mean + threshold2D(u.copy()))/k_rec

        return



class Gibbs_Probit_Sampler_record(MCMC_Sampler):
    """
    Probit Likelihood model:
    Phi(u; y, gamma) = - log(Psi(u_j(1) y_j(1); gamma)) - log(Psi(u_j(1) y_j(1); gamma))
                        ... -log(Psi(u_j(N') y_j(N'); gamma))
        where Psi(v; gamma) is CDF of normal distribution with standard deviation gamma
        and j(i) is the ith labeled node.

    Gibbs sampling iterates between sampling from truncated normal distributions
    on the labeled set, and Karhuenen-Loeve expansion for sampling based on the prior.
    """
    def __init__(self, gamma=0.01, tau=0.01, alpha=1.):
        MCMC_Sampler.__init__(self, gamma, tau, alpha)
        self.name = "Gaussian_Probit"

    def load_data(self, Data, plot_=False):
        MCMC_Sampler.load_data(self, Data, plot_)
        if -1 in self.Data.fid.keys():
            print("Noticed you gave Gibbs-Probit classes with -1. Converting to 0 for the current implementation...")
            self.Data.fid[0] = self.Data.fid[-1]
            del self.Data.fid[-1]
            self.Data.ground_truth[self.Data.ground_truth == -1] = 0.
            self.Data.classes = [0,1]
        # initiate the trandn_multiclass object
        self.TRNM = TruncRandNormMulticlass(self.Data.num_class)


    def run_sampler(self, num_samples, burnIn=0, f='thresh', seed=None):
        MCMC_Sampler.run_sampler(self, num_samples, burnIn)
        print("Running Gibbs-Probit sampling to get %d samples, with %d burnIn samples" % (num_samples, burnIn))

        # instantiate the u_mean (unthresh), u_t_mean (unthres) variables
        self.u_mean = np.zeros((self.Data.N, self.Data.num_class))
        self.u_t_mean = np.zeros_like(self.u_mean)

        # fixed initialization of u.
        u = np.zeros_like(self.u_mean)

        # fidv contains the indices of the fidelity nodes in the list "labeled"
        fidv = {}
        for c, ind in self.Data.fid.items():
            u[np.ix_(ind, len(ind)*[c])] = 1.
            fidv[c] = [self.Data.labeled.index(j) for j in ind]


        # sample Gaussian noise in batch to begin
        np.random.seed(seed)
        z_all = np.random.randn(len(self.Data.evals), self.Data.num_class, num_samples+burnIn)

        # Compute the projections for use in the KL expansion for sampling u | v

        V_KJ = self.Data.evecs[self.Data.labeled,:]
        P_KJ = V_KJ.T.dot(V_KJ)/self.gamma2
        for i in range(len(self.evals_mod)):
            P_KJ[i,i] += self.evals_mod[i]
        #P_KJ = 0.5*(P_KJ + P_KJ.T)  # do we need? seems to be symmetric already...
        S_KJ, Q_KJ = eigh(P_KJ)
        inv_skj = 1./np.sqrt(S_KJ)[:,np.newaxis]


        # Main iterations
        self.uuT = np.zeros((self.Data.N, self.Data.N))
        self.ututT = np.zeros((self.Data.N, self.Data.N))
        #self.U = []
        for k in range(burnIn + num_samples):
            if self.print_ and k % 500 == 0:
                print('\tIteration %d of sampling...' % k)
            z_k = z_all[:,:,k] # the white noise samples for use in sampling u | v

            # Sample v ~ P(v | u).
            v = np.zeros((len(self.Data.labeled), self.Data.num_class))
            for cl, ind_cl in self.Data.fid.items():
                v[fidv[cl],:] = self.TRNM.gen_samples(u[ind_cl,:], self.gamma, cl)


            # Sample u ~ P(u | v) via KL expansion
            temp = V_KJ.T.dot(v)/self.gamma2
            temp = Q_KJ.T.dot(temp)
            temp /= S_KJ[:,np.newaxis]
            m_hat = Q_KJ.dot(temp)
            u_hat = Q_KJ.dot(z_k * inv_skj) + m_hat
            u = self.Data.evecs.dot(u_hat)


            # If past the burn in period, calculate the updates to the means we're recording
            if k > burnIn:
                #print(self.uuT.shape, u.shape)
                self.uuT += np.outer(u[:,0], u[:,0])
                self.ututT += np.outer(threshold1D(u[:,0].copy()), threshold1D(u[:,0].copy()))
                #self.U.append(u)
                k_rec = k - burnIn
                if k_rec == 1:
                    self.u_mean = u
                    if f == 'thresh':
                        self.u_t_mean = threshold2D(u.copy())
                else:
                    self.u_mean = ((k_rec-1) * self.u_mean + u)/k_rec
                    if f == 'thresh':
                        self.u_t_mean = ((k_rec-1) * self.u_t_mean + threshold2D(u.copy()))/k_rec

        return







class pCN_Probit_Sampler(MCMC_Sampler):
    """
    Probit Likelihood model:
        Phi(u; y, gamma) = - log(Psi(u_j(1) y_j(1); gamma)) - log(Psi(u_j(1) y_j(1); gamma))
                        ... -log(Psi(u_j(N') y_j(N'); gamma))
            where Psi(v; gamma) is CDF of normal distribution with standard deviation gamma
            and j(i) is the ith labeled node.

    pCN (pre-conditioned Crank Nicholson) is MH-inspired sampler that produces proposal
    steps by:
        w^k = sqrt(1 - beta^2)w^k + beta z^k,   z^k ~ N(0, L^(-1))
    with acceptance probability:
        a(v,w) = min( 1 , Phi(v; y, gamma) - Phi(w; y, gamma) )

    **** NOTE this is currently only written for Binary classification! ****
    """
    def __init__(self, beta, gamma=0.1, tau=0., alpha=1.):
        MCMC_Sampler.__init__(self, gamma, tau, alpha)
        self.name = "pCN_Probit"
        self.beta = beta

    def load_data(self, Data, plot_=False):
        MCMC_Sampler.load_data(self, Data, plot_)
        if max(self.Data.fid.keys()) > 1:
            raise NotImplementedError("Multiclass sampling for pCN Probit is not yet implemented")

        # self.y has ordering as given in variable self.labeled
        ofs = min(self.Data.fid.keys())
        self.y = np.ones(len(self.Data.labeled))
        mask = [np.where(self.Data.labeled == v)[0] for v in self.Data.fid[ofs]]
        self.y[mask] = ofs


    def run_sampler(self, num_samples, burnIn=0, f='thresh', seed=None):
        MCMC_Sampler.run_sampler(self, num_samples, burnIn)
        print("Running pCN Probit sampling to get %d samples, with %d burnIn samples" % (num_samples, burnIn))
        norm_rv = norm(scale=self.gamma)  # normal rv for generating the cdf values

        # Helper functions for running the MH sampling
        def log_like(x):
            return -np.sum(np.log(norm_rv.cdf(x * self.y)))

        def alpha(u, w):
            u_, w_ = u[self.Data.labeled], w[self.Data.labeled]

            return np.min([1., np.exp(log_like(u_) - log_like(w_))])

        # Sample Gaussian noise in batch
        np.random.seed(seed)
        if self.tau > 0:
            KL_scaled_evecs =  self.Data.evecs * self.evals_mod**-0.5
            z = np.random.randn(self.evals_mod.shape[0], num_samples+burnIn)
        else:
            KL_scaled_evecs =  self.Data.evecs[:,1:] * self.evals_mod[1:]**-0.5
            z = np.random.randn(self.evals_mod.shape[0]-1, num_samples+burnIn)

        # instantiate sample
        u = np.ones(self.Data.N, dtype=np.float32)*np.average(list(self.Data.fid.keys()))
        for v in self.Data.fid.keys():
            u[self.Data.fid[v]] = v



        # Main iterations
        for k in range(burnIn + num_samples):
            if self.print_ and k % 500 == 0:
                print('\tIteration %d of sampling...' % k)
            # proposal step
            w_k = np.sqrt(1. - self.beta**2.)*u + KL_scaled_evecs.dot(z[:,k])*self.beta

            # calc acceptance prob, and accept/reject proposal step accordingly
            acc_prob = alpha(u, w_k)
            if np.random.rand() <= acc_prob:
                u = w_k

            # Record mean if past burn-in stage
            if k >= burnIn:
                k_rec = k - burnIn
                if k_rec == 0:
                    self.u_mean = u
                    if f == 'thresh':
                        self.u_t_mean = threshold1D(u.copy())
                else:
                    self.u_mean = ((k_rec) * self.u_mean + u)/(k_rec + 1)
                    if f == 'thresh':
                        self.u_t_mean = ((k_rec) * self.u_t_mean + threshold1D(u.copy()))/(k_rec+1)



class pCN_Probit_Sampler_record(MCMC_Sampler):
    """
    Probit Likelihood model:
        Phi(u; y, gamma) = - log(Psi(u_j(1) y_j(1); gamma)) - log(Psi(u_j(1) y_j(1); gamma))
                        ... -log(Psi(u_j(N') y_j(N'); gamma))
            where Psi(v; gamma) is CDF of normal distribution with standard deviation gamma
            and j(i) is the ith labeled node.

    pCN (pre-conditioned Crank Nicholson) is MH-inspired sampler that produces proposal
    steps by:
        w^k = sqrt(1 - beta^2)w^k + beta z^k,   z^k ~ N(0, L^(-1))
    with acceptance probability:
        a(v,w) = min( 1 , Phi(v; y, gamma) - Phi(w; y, gamma) )

    **** NOTE this is currently only written for Binary classification! ****
    """
    def __init__(self, beta, gamma=0.1, tau=0., alpha=1.):
        MCMC_Sampler.__init__(self, gamma, tau, alpha)
        self.name = "pCN_Probit"
        self.beta = beta

    def load_data(self, Data, plot_=False):
        MCMC_Sampler.load_data(self, Data, plot_)
        if max(self.Data.fid.keys()) > 1:
            raise NotImplementedError("Multiclass sampling for pCN Probit is not yet implemented")

        # self.y has ordering as given in variable self.labeled
        ofs = min(self.Data.fid.keys())
        self.y = np.ones(len(self.Data.labeled))
        mask = [np.where(self.Data.labeled == v)[0] for v in self.Data.fid[ofs]]
        self.y[mask] = ofs


    def run_sampler(self, num_samples, burnIn=0, f='thresh', seed=None):
        MCMC_Sampler.run_sampler(self, num_samples, burnIn)
        print("Running pCN Probit sampling to get %d samples, with %d burnIn samples" % (num_samples, burnIn))
        norm_rv = norm(scale=self.gamma)  # normal rv for generating the cdf values

        # Helper functions for running the MH sampling
        def log_like(x):
            return -np.sum(np.log(norm_rv.cdf(x * self.y)))

        def alpha(u, w):
            u_, w_ = u[self.Data.labeled], w[self.Data.labeled]

            return np.min([1., np.exp(log_like(u_) - log_like(w_))])

        # Sample Gaussian noise in batch
        np.random.seed(seed)
        if self.tau > 0:
            KL_scaled_evecs =  self.Data.evecs * self.evals_mod**-0.5
            z = np.random.randn(self.evals_mod.shape[0], num_samples+burnIn)
        else:
            KL_scaled_evecs =  self.Data.evecs[:,1:] * self.evals_mod[1:]**-0.5
            z = np.random.randn(self.evals_mod.shape[0]-1, num_samples+burnIn)

        # instantiate sample
        u = np.ones(self.Data.N, dtype=np.float32)*np.average(list(self.Data.fid.keys()))
        for v in self.Data.fid.keys():
            u[self.Data.fid[v]] = v

        self.u_samples = []



        # Main iterations
        num_acc = 0
        for k in range(burnIn + num_samples):
            if self.print_ and k % 500 == 0:
                print('\tIteration %d of sampling...' % k)
            # proposal step
            w_k = np.sqrt(1. - self.beta**2.)*u + KL_scaled_evecs.dot(z[:,k])*self.beta

            # calc acceptance prob, and accept/reject proposal step accordingly
            acc_prob = alpha(u, w_k)
            if np.random.rand() <= acc_prob:
                num_acc += 1
                u = w_k

            # Record mean if past burn-in stage
            if k >= burnIn:
                k_rec = k - burnIn
                self.u_samples.append(u)
                if k_rec == 0:
                    self.u_mean = u
                    if f == 'thresh':
                        self.u_t_mean = threshold1D(u.copy())
                else:
                    self.u_mean = ((k_rec) * self.u_mean + u)/(k_rec + 1)
                    if f == 'thresh':
                        self.u_t_mean = ((k_rec) * self.u_t_mean + threshold1D(u.copy()))/(k_rec+1)


        print("number of accepted points = %d" % num_acc)







class pCN_BLS_Sampler(MCMC_Sampler):
    """
    Bayesian Level-Set (BLS) Likelihood model:

        Phi(u; y, gamma) = 1/gamma^2 ||y_j - S(u_j)||^2, where S(v) = sgn(v).

    pCN (pre-conditioned Crank Nicholson) is MH-inspired sampler that produces proposal
    steps by:
        w^k = sqrt(1 - beta^2)w^k + beta z^k,   z^k ~ N(0, L^(-1))
    with acceptance probability:
        a(v,w) = min( 1 , Phi(v; y, gamma) - Phi(w; y, gamma) )

    **** NOTE this is currently only written for Binary classification! ****
    """
    def __init__(self, beta, gamma=0.1, tau=0., alpha=1.):
        MCMC_Sampler.__init__(self, gamma, tau, alpha)
        self.name = "pCN_BLS"
        self.beta = beta

    def load_data(self, Data, plot_=False):
        MCMC_Sampler.load_data(self, Data, plot_)
        if max(self.Data.fid.keys()) > 1:
            raise NotImplementedError("Multiclass sampling for pCN BLS is not yet implemented")

        # self.y has ordering as given in variable self.labeled
        self.y = self.Data.ground_truth[self.Data.labeled]
        self.zero_one = False
        if 0 in self.Data.fid.keys():
            self.zero_one = True
        print("classes contains 0 is %s" % str(self.zero_one))

    def run_sampler(self, num_samples, burnIn=0, f='thresh', seed=None):
        MCMC_Sampler.run_sampler(self, num_samples, burnIn)
        print("Running pCN BLS sampling to get %d samples, with %d burnIn samples" % (num_samples, burnIn))

        norm_rv = norm(scale=self.gamma)  # normal rv for generating the cdf values

        # Helper functions for running the MH sampling
        def log_like(x):
            return len(np.where(x != self.y)[0])/(np.sqrt(2.)*self.gamma2)

        def alpha(u, w):
            u_, w_ = threshold1D(u[self.Data.labeled].copy(), self.zero_one), threshold1D(w[self.Data.labeled].copy(), self.zero_one)
            return np.min([1., np.exp(log_like(u_) - log_like(w_))])

        # Sample Gaussian noise in batch
        np.random.seed(seed)
        if self.tau > 0:
            KL_scaled_evecs =  self.Data.evecs * self.evals_mod**-0.5
            z = np.random.randn(self.evals_mod.shape[0], num_samples+burnIn)
        else:
            KL_scaled_evecs =  self.Data.evecs[:,1:] * self.evals_mod[1:]**-0.5
            z = np.random.randn(self.evals_mod.shape[0]-1, num_samples+burnIn)

        # instantiate sample
        u = np.ones(self.Data.N, dtype=np.float32)*np.average(list(self.Data.fid.keys()))
        for v in self.Data.fid.keys():
            u[self.Data.fid[v]] = v

        # Main iterations
        self.accepted = 0
        for k in range(burnIn + num_samples):
            if self.print_ and k % 500 == 0:
                print('\tIteration %d of sampling...' % k)
            # proposal step
            w_k = np.sqrt(1. - self.beta**2.)*u + KL_scaled_evecs.dot(z[:,k])*self.beta


            # calc acceptance prob, and accept/reject the proposal step accordingly
            acc_prob = alpha(u, w_k)
            if np.random.rand() <= acc_prob:
                u = w_k
                self.accepted += 1

            # Record mean if past burn-in stage
            if k >= burnIn:
                k_rec = k - burnIn
                if k_rec == 0:
                    self.u_mean = u.copy()
                    if f == 'thresh':
                        self.u_t_mean = threshold1D(u.copy(), self.zero_one)
                else:
                    self.u_mean = ((k_rec) * self.u_mean + u)/(k_rec + 1)
                    if f == 'thresh':
                        self.u_t_mean = ((k_rec) * self.u_t_mean + threshold1D(u.copy(), self.zero_one))/(k_rec+1)






""" OLD MATLAB based probit gibbs sampler

class Gibbs_Probit_Sampler_MATLAB(MCMC_Sampler):
    def __init__(self, gamma=0.01, tau=0., alpha=1.):
        MCMC_Sampler.__init__(self, gamma, tau, alpha)
        if tau != 0. or alpha != 1.:
            print('Sampling for tau != 0. and alpha != 1. is not yet implemented, proceeding with default values')

    def load_data(self, Data, plot_=False):
        MCMC_Sampler.load_data(self, Data, plot_)

        # initiate the trandn_multiclass object
        print("Gibbs Probit Sampler's current implementation uses MATLAB functions...")
        print('\tInstantiating the matlab objects for trandn_multiclass sampler')
        tic = time.process_time()
        self.eng = matlab.engine.start_matlab()
        print('\tmatlab engine instantiation took %s seconds' % str(time.process_time() - tic ))
        self.eng.evalc("trand_obj = trandn_multiclass(%s);" % str(Data.num_class))
        self.eng.evalc("fid = {};")
        for c, ind in self.Data.fid.items():
            self.eng.evalc("fid{%d} = %s;"%(c+1,str([a+ 1 for a in ind])))
        print('\tFinished MATLAB initialization')


    def run_sampler(self, num_samples, burnIn=0, f='thresh'):
        MCMC_Sampler.run_sampler(self, num_samples, burnIn)

        # instantiate the u_mean (unthresh), u_t_mean (unthres) variables
        self.u_mean = np.zeros((self.Data.N, self.Data.num_class))
        self.u_t_mean = np.zeros_like(self.u_mean)

        # fixed initialization of u.
        u = np.zeros_like(self.u_mean)

        # fidv contains the indices of the fidelity nodes in the list "labeled"
        fidv = {}
        for c, ind in self.Data.fid.items():
            u[np.ix_(ind, len(ind)*[c])] = 1.
            fidv[c] = [self.Data.labeled.index(j) for j in ind]

        # sample Gaussian noise in batch to begin
        z_all = np.random.randn(len(self.evals_mod), self.Data.num_class, num_samples+burnIn)

        # Compute the projections for use in the KL expansion for sampling u | v
        V_KJ = self.Data.evecs[self.Data.labeled,:]
        P_KJ = (1./self.gamma2)*V_KJ.T.dot(V_KJ)
        for i in range(len(self.evals_mod)):
            P_KJ[i,i] += self.evals_mod[i]
        #P_KJ = 0.5*(P_KJ + P_KJ.T)  # do we need? seems to be symmetric already...
        S_KJ, Q_KJ = eigh(P_KJ)
        inv_skj = 1./np.sqrt(S_KJ)[:,np.newaxis]

        # Main iterations
        for k in range(burnIn + num_samples):
            if self.print_ and k % 500 == 0:
                print('\tIteration %d of sampling...' % k)
            z_k = z_all[:,:,k] # the white noise samples for use in sampling u | v

            # Sample v ~ P(v | u). Uses the MATLAB function for trandn_multiclass MATLAB object
            v = np.zeros((len(self.Data.labeled), self.Data.num_class))
            #tic = time.process_time()
            for cl, ind_cl in self.Data.fid.items():
                self.eng.workspace['u_cl'] = matlab.double(u[ind_cl,:].tolist())
                self.eng.evalc("v_cl = trand_obj.gen_samples(u_cl, %f, %d);" % (self.gamma2**0.5, cl+1))
                v[fidv[cl],:] = np.asarray(self.eng.workspace['v_cl'])


            # Sample u ~ P(u | v) via KL expansion
            temp = V_KJ.T.dot(v)/self.gamma2
            temp = Q_KJ.T.dot(temp)
            temp *= inv_skj
            m_hat = Q_KJ.dot(temp)
            u_hat = Q_KJ.dot(z_k * inv_skj) + m_hat
            u = self.Data.evecs.dot(u_hat)

            # If past the burn in period, calculate the updates to the means we're recording
            if k > burnIn:
                k_rec = k - burnIn
                if k_rec == 1:
                    self.u_mean = u
                    if f == 'thresh':
                        self.u_t_mean = threshold2D(u.copy())
                else:
                    self.u_mean = ((k_rec-1) * self.u_mean + u)/k_rec
                    if f == 'thresh':
                        self.u_t_mean = ((k_rec-1) * self.u_t_mean + threshold2D(u.copy()))/k_rec

        return

"""
