# author: Kevin Miller
import numpy as np
from scipy.stats import norm
from util.al_util import *
import time


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
def get_k(C, unlabeled, acquisition, gamma=0.1, m = None, y=None):
    if acquisition == "modelchange_p":
        return modelchange_p(C, unlabeled, gamma, m, probit_norm=True)
    elif acquisition == "modelchange_p2":
        return  modelchange_p(C, unlabeled, gamma, m, probit_norm=False)
    elif acquisition == "vopt_p":
        return  vopt_p(C, unlabeled, gamma, m, probit_norm=True)
    elif acquisition == "vopt_p2":
        return vopt_p(C, unlabeled, gamma, m, probit_norm=False)
    elif acquisition == "mbr_p":
        return mbr_p(C, unlabeled, gamma, m, probit_norm=True)
    elif acquisition == "mbr_p2":
        return mbr_p(C, unlabeled, gamma, m, probit_norm=False)
    elif acquisition == "vopt_gr":
        return vopt_gr(C, unlabeled, gamma)
    elif acquisition == "sopt_gr":
        return sopt_gr(C, unlabeled, gamma)
    elif acquisition == "mbr_gr":
        return mbr_gr(C, unlabeled, gamma, m)
    elif acquisition == "modelchange_gr":
        return modelchange_gr(C, unlabeled, gamma, m)
    elif acquisition == "vopt_new_p":
        return vopt_p_new(C, unlabeled, gamma, m, probit_norm=True)
    elif acquisition == "vopt_new_p2":
        return vopt_p_new(C, unlabeled, gamma, m, probit_norm=False)
    elif acquisition in ("uncertainty_gr", "uncertainty_p", "uncertainty_p2"):
        return uncertainty(unlabeled, m)
    elif acquisition == "uncertainty_hf":
        return uncertainty_hf(m)
    elif acquisition == "vopt_hf": # Note Harmonic Function acquisitions return in the index in the unlabeled subset,
        return vopt_hf(C)           # NOT in the full index set of total nodes
    elif acquisition == "sopt_hf":
        return sopt_hf(C)
    elif acquisition == "mbr_hf":
        return mbr_hf(C, m)
    else:
        pass

################ Harmonic Functions Active Learning Acquisition Functions ##########

def vopt_hf(C):
    '''
    Compute the original V-opt criterion under the Harmonic Functions model.
        ** NOTE : this returns the index IN the submatrix that C is, not in the
        original index {1, 2, ..., N}
    '''
    v_opt = [np.inner(C[i,:], C[i,:])/C[i,i] for i in range(C.shape[0])]
    k_max = np.argmax(v_opt)
    return k_max

def sopt_hf(C):
    '''
    Compute the original Sigma-opt criterion under the Harmonic Functions model.
        ** NOTE : this returns the index IN the submatrix that C is, not in the
        original index {1, 2, ..., N}
    '''
    s_opt = np.sum(C, axis=1)**2. / np.diag(C)
    k_max = np.argmax(s_opt)
    return k_max


def EE_hf(k, m, ck):
    mk1 = m + (1. - m[k])*ck/ck[k]
    risk = np.sum([min(mk1[i], 1. - mk1[i]) for i in range(m.shape[0])])
    mk_1 = m - m[k]*ck/ck[k]
    risk += np.sum([min(mk_1[i], 1. - mk_1[i]) for i in range(m.shape[0])])
    return risk

def mbr_hf(C, m):
    '''
    Compute the original MBR (AKA EEM) criterion under the Harmonic Functions model.
        ** NOTE : NEED 0-1 LABELING FOR THIS CRITERION.
        Also, this returns the index IN the submatrix that C is, not in the
        original index {1, 2, ..., N}
    '''
    if np.max(m) > 1. or np.min(m) < 0.:
        raise ValueError("Harmonic Function must have values between 0 and 1. Either your  \
        Graph Laplacian is not unnormalized, or you have given invalid labelings other than \
        0 and 1...")

    ee = [EE_hf(k, m, C[k,:]) for k in range(m.shape[0])]
    k_mbr = np.argmin(ee)
    return k_mbr






################ Gaussian Regression Active Learning Acquisition Functions ##########

def vopt_gr(C, unlabeled, gamma):
    '''
    Compute the V-opt criterion adapted to the Gaussian Regression model
    '''
    ips = np.array([np.inner(C[k,:], C[k,:]) for k in unlabeled]).flatten()
    v_opt = ips/(gamma**2. + np.diag(C)[unlabeled])
    k_max = unlabeled[np.argmax(v_opt)]
    return k_max

def sopt_gr(C, unlabeled, gamma):
    '''
    Compute the Sigma-opt criterion adapted to the Gaussian Regression model
    '''
    #sums = np.sum(C[np.ix_(unlabeled,unlabeled)], axis=1) # this is wrong...
    sums = np.sum(C[unlabeled,:], axis=1)
    s_opt = sums/(gamma**2. + np.diag(C)[unlabeled])
    k_max = unlabeled[np.argmax(s_opt)]
    return k_max



''' Helper functions for EEM function in Gaussian Regression model'''

def next_m_gr_old(m, C, y, lab, k, y_k, gamma2):
    '''
    Calculate the "plus k, y_k" posterior mean update in the Gaussian Regression
    model.
    '''
    ck = C[k,:]
    ckk = ck[k]
    ip = np.dot(ck[lab], y) # this calculation is unstable with ck[lab] having entries close to 0
    val = ((gamma2)*y_k -ip )/(gamma2*(gamma2 + ckk))
    m_k = m + val*ck
    return m_k


def next_m_gr(m, C, k, y_k, gamma2):
    '''
    Calculate the "plus k, y_k" posterior mean update in the Gaussian Regression
    model.

    This is more numerically stable version of the above function
    '''
    ck = C[k,:]
    ckk = ck[k]
    return m + (y_k - m[k])*ck/(gamma2 + ckk)


def get_probs_gr(m, sigmoid=False):
    '''
    Convert the Gaussian Regression mean vector into probabilities for use in EEM.
    '''
    if sigmoid:
        return 1./(1. + np.exp(-3.*m))
    m_probs = m.copy()
    # simple fix to get probabilities that respect the 0 threshold
    m_probs[np.where(m_probs > 0)] /= 2.*np.max(m_probs)
    m_probs[np.where(m_probs < 0)] /= -2.*np.min(m_probs)
    m_probs += 0.5
    return m_probs


def EE_gr(k, m, C, m_probs, gamma):
    '''
    Calculate the EE (expected error) in the Gaussian Regression model; that is,
    adapted the MBR criterion from Harmonic functions to fit the Gaussian Regression
    model (in which we have +1, -1 labels, and soft labelings).
    '''
    N = C.shape[0]
    m_at_k = m_probs[k]
    m_k_p1 = next_m_gr(m, C, k, 1., gamma**2.)
    m_k_p1 = get_probs_gr(m_k_p1)
    risk = m_at_k*np.sum([min(m_k_p1[i], 1.- m_k_p1[i]) for i in range(N)])
    m_k_m1 = next_m_gr(m, C, k, -1., gamma**2.)
    m_k_m1 = get_probs_gr(m_k_m1)
    risk += (1.-m_at_k)*np.sum([min(m_k_m1[i], 1.- m_k_m1[i]) for i in range(N)])
    return risk

def EE_gr_old(k, m, C, m_probs, labeled, y, gamma):
    '''
    Calculate the EE (expected error) in the Gaussian Regression model; that is,
    adapted the MBR criterion from Harmonic functions to fit the Gaussian Regression
    model (in which we have +1, -1 labels, and soft labelings).
    '''
    N = C.shape[0]
    m_at_k = m_probs[k]
    m_k_p1 = next_m_gr_old(m, C, labeled, y, k, 1., gamma**2.)
    m_k_p1 = get_probs_gr(m_k_p1)
    risk = m_at_k*np.sum([min(m_k_p1[i], 1.- m_k_p1[i]) for i in range(N)])
    m_k_m1 = next_m_gr_old(m, C, labeled, y, k, -1., gamma**2.)
    m_k_m1 = get_probs_gr(m_k_m1)
    risk += (1.-m_at_k)*np.sum([min(m_k_m1[i], 1.- m_k_m1[i]) for i in range(N)])
    return risk

def mbr_gr(C, unlabeled, gamma, m):
    '''
    Compute the MBR acquisition choice, adapted to the Gaussian Regression model.
    '''
    m_probs = get_probs_gr(m)
    labeled = list(filter(lambda i: i not in unlabeled, range(C.shape[0])))
    risks = [EE_gr(j, m, C, m_probs, gamma) for j in unlabeled]
    k_gbr = unlabeled[np.argmin(risks)]
    return k_gbr

def modelchange_gr(C, unlabeled, gamma, m):
    '''
    Compute the model change criterion under the probit model (with Gaussian approximations of Probit posterior).
    '''
    mc = [min(np.absolute(1. - m[k])/(gamma**2. + C[k,k]), np.absolute(-1. - m[k])/(gamma**2. + C[k,k])) \
                       * np.linalg.norm(C[k,:]) for k in unlabeled]
    k_mc = unlabeled[np.argmax(mc)]
    return k_mc

def modelchange_gr_multi(C, unlabeled, gamma, m):
    '''
    Compute the model change criterion under the probit model (with Gaussian approximations of Probit posterior).
    '''
    mc = [np.max(np.abs(m[k])) * np.linalg.norm(C[k,:]) / (gamma**2. + C[k,k]) for k in unlabeled]
    k_mc = unlabeled[np.argmax(mc)]
    return k_mc





########################### Probit Active Learning Acquisition Functions ###########

def vopt_p(C, unlabeled, gamma, m, dumb=False, probit_norm=False):
    '''
    Compute V-opt criterion adapted to the Probit model (i.e. the Gaussian Approximation
    of the Probit posterior).
    '''
    ips = np.array([np.inner(C[k,:], C[k,:]) for k in unlabeled]).flatten()
    if dumb:
        v_opt = ips/(gamma**2. + np.diag(C)[unlabeled])
    else:
        if probit_norm:
            # take the "best case" labeling -- MAYBE do weighted average?
            v_opt = ips * np.array([hess_calc(m[k], np.sign(m[k]), gamma)/(hess_calc(m[k], np.sign(m[k]), gamma)*C[k,k] + 1.) for k in unlabeled])
        else:
            # take the "best case" labeling -- MAYBE do weighted average?
            v_opt = ips * np.array([hess_calc2(m[k], np.sign(m[k]), gamma)/(hess_calc2(m[k], np.sign(m[k]), gamma)*C[k,k] + 1.) for k in unlabeled])
    k_max = unlabeled[np.argmax(v_opt)]
    return k_max


def vopt_p_new(C, unlabeled, gamma, m, dumb=False, probit_norm=False):
    '''
    Compute V-opt criterion adapted to the Probit model (i.e. the Gaussian Approximation
    of the Probit posterior).

    This differs from vopt_p in that we use the NA for the k^th entries to plug into the posterior covariance calculation of columns,
    whereas vopt_p is using the previous MAP estimator's corresponding values. Empirically, doesn't seem to change the overall
    performance, though different AL choices are made.
    '''
    ips = np.array([np.inner(C[k,:], C[k,:]) for k in unlabeled]).flatten()
    if dumb:
        v_opt = ips/(gamma**2. + np.diag(C)[unlabeled])
    else:
        if probit_norm:
            mNA = m - np.array([jac_calc(m[i], np.sign(m[i]), gamma)*C[i,i]/(1. + C[i,i]*hess_calc(m[i],np.sign(m[i]), gamma)) for i in range(m.shape[0])])
            # take the "best case" labeling -- MAYBE do weighted average?
            v_opt = ips * np.array([hess_calc(mNA[k], np.sign(mNA[k]), gamma)/(hess_calc(mNA[k], np.sign(mNA[k]), gamma)*C[k,k] + 1.) for k in unlabeled])
        else:
            # take the "best case" labeling -- MAYBE do weighted average?
            mNA = m - np.array([jac_calc2(m[i], np.sign(m[i]), gamma)*C[i,i]/(1. + C[i,i]*hess_calc2(m[i],np.sign(m[i]), gamma)) for i in range(m.shape[0])])
            v_opt = ips * np.array([hess_calc2(mNA[k], np.sign(mNA[k]), gamma)/(hess_calc2(mNA[k], np.sign(mNA[k]), gamma)*C[k,k] + 1.) for k in unlabeled])
    k_max = unlabeled[np.argmax(v_opt)]
    return k_max


def modelchange_p(C, unlabeled, gamma, m, probit_norm=False, debug=False):
    '''
    Compute the model change criterion under the probit model (with Gaussian approximations of Probit posterior).
    '''
    if probit_norm:
        mc = [min(np.absolute(jac_calc(m[k], -1, gamma)/(1. + C[k,k]*hess_calc(m[k], -1, gamma ))), \
           np.absolute(jac_calc(m[k], 1, gamma)/(1. + C[k,k]*hess_calc(m[k], 1, gamma )))) \
                       * np.linalg.norm(C[k,:]) for k in unlabeled]
    else:
        mc = [min(np.absolute(jac_calc2(m[k], -1, gamma)/(1. + C[k,k]*hess_calc2(m[k], -1, gamma ))), \
           np.absolute(jac_calc2(m[k], 1, gamma)/(1. + C[k,k]*hess_calc2(m[k], 1, gamma )))) \
                       * np.linalg.norm(C[k,:]) for k in unlabeled]
    k_mc = unlabeled[np.argmax(mc)]
    if debug:
        return k_mc, mc
    return k_mc

def modelchange_p_other(C, unlabeled, gamma, m, probit_norm=False, debug=False):
    '''
    Compute the model change criterion under the probit model (with Gaussian approximations of Probit posterior).
    '''
    if probit_norm:
        mc = [np.absolute(jac_calc(m[k], np.sign(m[k]), gamma)/(1. + C[k,k]*hess_calc(m[k], np.sign(m[k]), gamma ))) \
                       * np.linalg.norm(C[k,:]) for k in unlabeled]
    else:
        mc = [np.absolute(jac_calc2(m[k], np.sign(m[k]), gamma)/(1. + C[k,k]*hess_calc2(m[k], np.sign(m[k]), gamma )))\
                       * np.linalg.norm(C[k,:]) for k in unlabeled]
    k_mc = unlabeled[np.argmax(mc)]
    if debug:
        return k_mc, mc
    return k_mc



# Helper functions for MBR probit adaptation
def logistic_cdf(t, scale):
    return 1.0/(1.0 + np.exp(-t/scale))


def EE_p(k, m, C, gamma, probit_norm=False):
    '''
    Calculate the EE (expected error) in the Gaussian Regression model; that is,
    adapted the MBR criterion from Harmonic functions to fit the Gaussian Regression
    model (in which we have +1, -1 labels, and soft labelings).
    '''
    if probit_norm:
        cdf_func = norm.cdf
        jac_func = jac_calc
        hess_func = hess_calc
    else:
        cdf_func = logistic_cdf
        jac_func = jac_calc2
        hess_func = hess_calc2




    ck = C[k,:]
    ckk = ck[k]
    # Timing comparion in testing
    # tic = time.clock()
    # val = jac_func(m[k], 1., gamma)/(1. + ckk*hess_func(m[k], 1., gamma))
    # toc = time.clock()
    # print('NA probit val took %1.5f seconds' % (toc - tic))
    #
    # tic = time.clock()
    # val = (1. - m[k])/(gamma**2. + ckk)
    # toc = time.clock()
    # print('GR val took %1.5f seconds' % (toc - tic))



    # Get Newton Approximation of plus k, +1 optimizer
    m_k_p1 = m - jac_func(m[k], 1., gamma)/(1. + ckk*hess_func(m[k], 1., gamma))*ck
    risk = cdf_func(m[k], scale=gamma)*np.sum([cdf_func(-abs(m_k_p1[i]), scale=gamma) for i in range(m.shape[0])])

    # Get Newton Approximation of plus k, -1 optimizer
    m_k_m1 = m - jac_func(m[k], -1., gamma)/(1. + ckk*hess_func(m[k], -1., gamma ))*ck
    risk += cdf_func(-m[k], scale=gamma)*np.sum([cdf_func(-abs(m_k_m1[i]), scale=gamma) for i in range(m.shape[0])])
    return risk

def mbr_p(C, unlabeled, gamma, m, probit_norm=False):
    '''
    I think we can do MBR (EEM) in the Probit case, maybe this is useful?
    '''
    # Timing comparison
    # k = unlabeled[20]
    # tic = time.clock()
    # m_probs = get_probs_gr(m)
    # eegr = EE_gr(k, m, C, m_probs, gamma**2.)
    # toc = time.clock()
    # print('EE_gr took %1.5f seconds' % (toc - tic))
    #
    # tic = time.clock()
    # eegr = EE_p(k, m, C, gamma, probit_norm)
    # toc = time.clock()
    # print('EE_p took %1.5f seconds' % (toc - tic))


    mbr = [EE_p(k, m, C, gamma, probit_norm) for k in unlabeled]
    k_mbr = unlabeled[np.argmin(mbr)]
    return k_mbr

def uncertainty(unlabeled, m):
    # m[i] in -1, 1
    return unlabeled[np.argmin(np.abs(m[unlabeled]))]

def uncertainty_hf(m):
    return np.argmin(np.abs(m - 0.5))



######################### Spectral Truncation MC Acquisition #################
def modelchange_p_st(C_alpha, alpha, v_c, gamma, debug=False):
    '''
    This assumes that you give only the rows of the eigenvectors corresponding to the CANDIDATE set that you want
    to evaluate the criterion on. the index returned is the index in the Candidate set list, not in the original
    indices.
    '''
    uks = v_c @ alpha
    C_a_vk = C_alpha @ (v_c.T)
    mc = np.array([np.absolute(jac_calc2(uks[i], np.sign(uks[i]),gamma))/(1. + \
        np.inner(v_c[i,:], C_a_vk[:,i])*hess_calc2(uks[i], np.sign(uks[i]),gamma))* np.linalg.norm(C_a_vk[:,i]) \
                    for i in range(v_c.shape[0])])
    k = np.argmax(mc)
    if debug:
        return k, mc
    return k
