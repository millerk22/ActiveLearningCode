# author: Kevin Miller
import numpy as np
from scipy.stats import norm


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



################ Harmonic Functions Active Learning Acquisition Functions ##########

def vopt_hf(C, **kwargs):
    '''
    Compute the original V-opt criterion under the Harmonic Functions model.
        ** NOTE : this returns the index IN the submatrix that C is, not in the
        original index {1, 2, ..., N}
    '''
    v_opt = [np.inner(C[i,:], C[i,:])/C[i,i] for i in range(C.shape[0])]
    k_max = np.argmax(v_opt)
    return k_max

def sopt_hf(C, **kwargs):
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

def mbr_hf(C, m, **kwargs):
    '''
    Compute the original MBR (AKA EEM) criterion under the Harmonic Functions model.
        ** NOTE : NEED 0-1 LABELING FOR THIS CRITERION.
        Also, this returns the index IN the submatrix that C is, not in the
        original index {1, 2, ..., N}
    '''
    if np.max(m) > 1. or np.min(m) < 0.:
        raise ValueError("Harmonic Function must have be between 0 and 1. Either your  \
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
    sums = np.sum(C[np.ix_(unlabeled,unlabeled)], axis=1)
    sums = np.asarray(sums).flatten()**2.
    s_opt = sums/(gamma**2. + np.diag(C)[unlabeled])
    k_max = unlabeled[np.argmax(s_opt)]
    return k_max



''' Helper functions for EEM function in Gaussian Regression model'''

def next_m_gr(m, C, y, lab, k, y_k, gamma2):
    '''
    Calculate the "plus k, y_k" posterior mean update in the Gaussian Regression
    model.
    '''
    ck = C[k,:]
    ckk = ck[k]
    ip = np.dot(ck[lab], y[lab])
    val = ((gamma2)*y_k -ip )/(gamma2*(gamma2 + ckk))
    m_k = m + val*ck
    return m_k


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


def EE_gr(k, m, C, y, labeled, unlabeled, m_probs, gamma):
    '''
    Calculate the EE (expected error) in the Gaussian Regression model; that is,
    adapted the MBR criterion from Harmonic functions to fit the Gaussian Regression
    model (in which we have +1, -1 labels, and soft labelings).
    '''
    N = C.shape[0]
    m_at_k = m_probs[k]
    m_k_p1 = next_m_gr(m, C, y, labeled, k, 1., gamma**2.)
    m_k_p1 = get_probs_gr(m_k_p1)
    risk = m_at_k*np.sum([min(m_k_p1[i], 1.- m_k_p1[i]) for i in range(N)])
    m_k_m1 = next_m_gr(m, C, y, labeled, k, -1., gamma**2.)
    m_k_m1 = get_probs_gr(m_k_m1)
    risk += (1.-m_at_k)*np.sum([min(m_k_m1[i], 1.- m_k_m1[i]) for i in range(N)])
    return risk


def mbr_gr(C, unlabeled, gamma, m, y):
    '''
    Compute the MBR acquisition choice, adapted to the Gaussian Regression model.
    '''
    m_probs = get_probs_gr(m)
    labeled = list(filter(lambda i: i not in unlabeled, range(C.shape[0])))
    risks = [EE_gr(j, m, C, y, labeled, unlabeled, m_probs, gamma**2) for j in unlabeled]
    k_gbr = unlabeled[np.argmin(risks)]
    return k_gbr





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


def modelchange_p(C, unlabeled, gamma, m, probit_norm=False):
    if probit_norm:
        mc = [min(np.absolute(jac_calc(m[k], -1, gamma)/(1. + C[k,k]*hess_calc(m[k], -1, gamma ))), \
           np.absolute(jac_calc(m[k], 1, gamma)/(1. + C[k,k]*hess_calc(m[k], 1, gamma )))) \
                       * np.linalg.norm(C[k,:]) for k in unlabeled]
    else:
        mc = [min(np.absolute(jac_calc2(m[k], -1, gamma)/(1. + C[k,k]*hess_calc2(m[k], -1, gamma ))), \
           np.absolute(jac_calc2(m[k], 1, gamma)/(1. + C[k,k]*hess_calc2(m[k], 1, gamma )))) \
                       * np.linalg.norm(C[k,:]) for k in unlabeled]
    k_mc = np.argmax(mc)
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

    # Get Newton Approximation of plus k, +1 optimizer
    m_k_p1 = m - jac_func(m[k], 1, gamma)/(1. + C[k,k]*hess_func(m[k], 1, gamma))*C[k,:]
    risk = cdf_func(m[k], scale=gamma)*np.sum([cdf_func(-abs(m_k_p1[i]), scale=gamma) for i in range(m.shape[0])])

    # Get Newton Approximation of plus k, -1 optimizer
    m_k_m1 = m - jac_func(m[k], -1, gamma)/(1. + C[k,k]*hess_func(m[k], -1, gamma ))*C[k,:]
    risk += cdf_func(-m[k], scale=gamma)*np.sum([cdf_func(-abs(m_k_m1[i]), scale=gamma) for i in range(m.shape[0])])
    return risk

def mbr_p(C, unlabeled, gamma, m, probit_norm=False):
    '''
    I think we can do MBR (EEM) in the Probit case, maybe this is useful?
    '''
    mbr = [EE_p(k, m, C, gamma, probit_norm) for k in unlabeled]
    k_mbr = np.argmin(mbr)
    return k_mbr
