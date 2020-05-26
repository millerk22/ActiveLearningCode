# author: Kevin Miller
import numpy as np
from scipy.stats import norm
from util.acquisition import *
from util.al_util import *


'''
Graph Based SSL Models Considered:

    + Harmonic Functions : see "Combining Active Learning and Semi-Supervised
        Learning Using Gaussian Fields and Harmonic Functions" by Zhu, Gharamani,
        and Lafferty (2003)
        y_u = -L_uu L_ul y_l     --- harmonic function solution, defined on only
                                    the unlabeled nodes. (i.e. "hard labelings")


    + Gaussian Regression : Similar to Harmonic Functions model, but we allow
        noise in the observed labelings with parameter gamma. Resulting posterior
        distribution is Gaussian, now always defined over ALL of the nodes in the
        similarity graph. See "Posterior Consistency in Graph Based Semi-Supervised
        Regression" by Li, Miller, et al (2020).

        p(u | y) ~ exp(-1/2[ < u, L u> + 1/gamma^2 ||Hu - y||_2^2 ])
                = N (m, C), where m = -1/gamma^2 CH^Ty, and
                                    C^{-1} = L + 1/gamma^2 B    (B = H^TH)


    + Probit : Adapt the Bayesian model to follow a probit likelihood potential.
        That is, observations are assumed to follow the noise model:
            y_j = sgn( u_j + eta_j ), where eta_j ~ N(0, gamma^2) or Logistic(0, gamma)

        We approximate this intractable posterior via a Laplace (Gaussian) approximation
        so that our estimated probit posterior follows:
            P(u | y) ~ N(m, C), where
                m = argmin_u 1/2<u, Lu> - \sum_{j \in labeled} \log Psi_gamma(u_j y_j)  and
                C^{-1} = L + \sum_{j \in labeled} F'(m_j, y_j) e_j e_j^T
'''

class ActiveLearningAcquisition(object):
    def __init__(self):
        self.name = 'Unchosen'
        return

    def get_name(self):
        return self.name


    ################ Harmonic Functions Active Learning Acquisition Functions ##########

    def vopt_hfv(self, C, **kwargs):
        '''
        Compute the original V-opt criterion under the Harmonic Functions model.
            ** NOTE : this returns the index IN the submatrix that C is, not in the
            original index {1, 2, ..., N}
        '''
        self.name = 'V opt-hf'
        v_opt = [np.inner(C[i,:], C[i,:])/C[i,i] for i in range(C.shape[0])]
        k_max = np.argmax(v_opt)
        return k_max, v_opt

    def sopt_hfv(self, C, **kwargs):
        '''
        Compute the original Sigma-opt criterion under the Harmonic Functions model.
            ** NOTE : this returns the index IN the submatrix that C is, not in the
            original index {1, 2, ..., N}
        '''
        self.name = 'Sigma opt-hf'
        s_opt = np.sum(C, axis=1)**2. / np.diag(C)
        k_max = np.argmax(s_opt)
        return k_max, s_opt

    #
    # def EE_hf(k, m, ck):
    #     mk1 = m + (1. - m[k])*ck/ck[k]
    #     risk = np.sum([min(mk1[i], 1. - mk1[i]) for i in range(m.shape[0])])
    #     mk_1 = m - m[k]*ck/ck[k]
    #     risk += np.sum([min(mk_1[i], 1. - mk_1[i]) for i in range(m.shape[0])])
    #     return risk

    def mbr_hfv(self, C, m, **kwargs):
        '''
        Compute the original MBR (AKA EEM) criterion under the Harmonic Functions model.
            ** NOTE : NEED 0-1 LABELING FOR THIS CRITERION.
            Also, this returns the index IN the submatrix that C is, not in the
            original index {1, 2, ..., N}
        '''
        self.name = 'MBR-hf'
        if np.max(m) > 1. or np.min(m) < 0.:
            raise ValueError("Harmonic Function must have values between 0 and 1. Either your  \
            Graph Laplacian is not unnormalized, or you have given invalid labelings other than \
            0 and 1...")

        ee = [EE_hf(k, m, C[k,:]) for k in range(m.shape[0])]
        k_mbr = np.argmin(ee)
        return k_mbr, ee






    ################ Gaussian Regression Active Learning Acquisition Functions ##########

    def vopt_grv(self, C, unlabeled, gamma, **kwargs):
        '''
        Compute the V-opt criterion adapted to the Gaussian Regression model
        '''
        self.name = 'V opt-gr'
        ips = np.array([np.inner(C[k,:], C[k,:]) for k in unlabeled]).flatten()
        v_opt = ips/(gamma**2. + np.diag(C)[unlabeled])
        k_max = unlabeled[np.argmax(v_opt)]
        return k_max, v_opt

    def sopt_grv(self, C, unlabeled, gamma, **kwargs):
        '''
        Compute the Sigma-opt criterion adapted to the Gaussian Regression model
        '''
        self.name = 'Sigma opt-gr'
        sums = np.sum(C[np.ix_(unlabeled,unlabeled)], axis=1)
        s_opt = (sums**2.)/(gamma**2. + np.diag(C)[unlabeled])
        k_max = unlabeled[np.argmax(s_opt)]
        return k_max, s_opt



    def mbr_grv(self, C, unlabeled, gamma, m, y, **kwargs):
        '''
        Compute the MBR acquisition choice, adapted to the Gaussian Regression model.
        '''
        self.name = 'MBR-gr'
        m_probs = get_probs_gr(m)
        labeled = list(filter(lambda i: i not in unlabeled, range(C.shape[0])))
        risks = [EE_gr(j, m, C, m_probs, gamma**2) for j in unlabeled]
        k_gbr = unlabeled[np.argmin(risks)]
        return k_gbr, risks


    def modelchange_grv(self, C, unlabeled, gamma, m, **kwargs):
        '''
        Compute the model change criterion under the probit model (with Gaussian approximations of Probit posterior).
        '''
        self.name = 'Model Change-gr'
        mc = [min(np.absolute(1. - m[k])/(gamma**2. + C[k,k]), np.absolute(-1. - m[k])/(gamma**2. + C[k,k])) \
                           * np.linalg.norm(C[k,:]) for k in unlabeled]
        k_mc = unlabeled[np.argmax(mc)]
        return k_mc, mc





    ########################### Probit Active Learning Acquisition Functions ###########

    def vopt_pv(self, C, unlabeled, gamma, m, dumb=False, probit_norm=False, **kwargs):
        '''
        Compute V-opt criterion adapted to the Probit model (i.e. the Gaussian Approximation
        of the Probit posterior).
        '''

        ips = np.array([np.inner(C[k,:], C[k,:]) for k in unlabeled]).flatten()
        if dumb:
            v_opt = ips/(gamma**2. + np.diag(C)[unlabeled])
        else:
            if probit_norm:
                self.name = 'V opt-probit norm'
                # take the "best case" labeling -- MAYBE do weighted average?
                v_opt = ips * np.array([hess_calc(m[k], np.sign(m[k]), gamma)/(hess_calc(m[k], np.sign(m[k]), gamma)*C[k,k] + 1.) for k in unlabeled])
            else:
                self.name = 'V opt-probit log'
                # take the "best case" labeling -- MAYBE do weighted average?
                v_opt = ips * np.array([hess_calc2(m[k], np.sign(m[k]), gamma)/(hess_calc2(m[k], np.sign(m[k]), gamma)*C[k,k] + 1.) for k in unlabeled])
        k_max = unlabeled[np.argmax(v_opt)]
        return k_max, v_opt


    def modelchange_pv(self, C, unlabeled, gamma, m, probit_norm=False, **kwargs):
        '''
        Compute the model change criterion under the probit model (with Gaussian approximations of Probit posterior).
        '''
        if probit_norm:
            self.name = 'Model Change-probit norm'
            mc = [min(np.absolute(jac_calc(m[k], -1, gamma)/(1. + C[k,k]*hess_calc(m[k], -1, gamma ))), \
               np.absolute(jac_calc(m[k], 1, gamma)/(1. + C[k,k]*hess_calc(m[k], 1, gamma )))) \
                           * np.linalg.norm(C[k,:]) for k in unlabeled]
        else:
            self.name = 'Model Change-probit log'
            mc = [min(np.absolute(jac_calc2(m[k], -1, gamma)/(1. + C[k,k]*hess_calc2(m[k], -1, gamma ))), \
               np.absolute(jac_calc2(m[k], 1, gamma)/(1. + C[k,k]*hess_calc2(m[k], 1, gamma )))) \
                           * np.linalg.norm(C[k,:]) for k in unlabeled]
        k_mc = unlabeled[np.argmax(mc)]
        return k_mc, mc



    def mbr_pv(self, C, unlabeled, gamma, m, probit_norm=False, **kwargs):
        '''
        I think we can do MBR (EEM) in the Probit case, maybe this is useful?
        '''
        if probit_norm:
            self.name = 'MBR-probit norm'
        else:
            self.name = 'MBR-probit log'
        mbr = [EE_p(k, m, C, gamma, probit_norm) for k in unlabeled]
        k_mbr = unlabeled[np.argmin(mbr)]
        return k_mbr, mbr
