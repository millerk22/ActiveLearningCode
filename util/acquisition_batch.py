# author: Kevin Miller
import numpy as np
from scipy.stats import norm
from util.al_util import *
import time


'''
Graph Based SSL Model Considered:
    + Probit : Adapt the Bayesian model to follow a probit likelihood potential.
        That is, observations are assumed to follow the noise model:
            y_j = sgn( u_j + eta_j ), where eta_j ~ N(0, gamma^2) or Logistic(0, gamma)

        We approximate this intractable posterior via a Laplace (Gaussian) approximation
        so that our estimated probit posterior follows:
            P(u | y) ~ N(m, C), where
                m = argmin_u 1/2<u, Lu> - \sum_{j \in labeled} \log Psi_gamma(u_j y_j)  and
                C^{-1} = L + \sum_{j \in labeled} F'(m_j, y_j) e_j e_j^T
'''

def get_k_batch(Lt, y, labeled, B, acquisition, u0=None, q0=None, gamma=0.1, X=None, labels=None, C=None):
    if acquisition == "admm":
        al_choices, al_choices_full = admm_acquisition(Lt, y, labeled, B, u0=u0, q0=q0, gamma=gamma,X=X, labels=labels)
    elif acquisition == "admm_rand":
        al_choices, al_choices_full = admm_acquisition_rand(Lt, y, labeled, B, u0=u0, q0=q0, gamma=gamma,X=X, labels=labels)
    elif acquisition == "bcd_exact":
        al_choices, al_choices_full = bcd_exactq_acquisition(Lt, y, labeled, B, u0=u0, q0=q0, gamma=gamma,X=X, labels=labels)
    elif acquisition == "bcd_prox":
        al_choices, al_choices_full = bcd_proxq_acquisition(Lt, y, labeled, B, u0=u0, q0=q0, gamma=gamma,X=X, labels=labels)
    elif acquisition == "alt_min":
        al_choices, al_choices_full = alt_exact_acquisition(Lt, y, labeled, B, u0=u0, q0=q0, gamma=gamma,X=X, labels=labels)
    elif acquisition == "modelchange_batch":
        al_choices, al_choices_full = modelchange_batch_acquisition(Lt, y, labeled, B, u0=u0, gamma=gamma, C=C, X=X, labels=labels)
    elif acquisition == "modelchange_batch_exact":
        al_choices, al_choices_full = modelchange_batch_exact_acquisition(Lt, y, labeled, B, u0=u0, gamma=gamma, C=C, X=X, labels=labels)
    else:
        print("Did not find the acquisition function, %s, in our list of batch acquisition functions:" % acquisition)
        print("Possible batch acquisition functions = \n\t%s" % str(["admm", "bcd_exact", "bcd_prox", "alt_min"]))
        return
    # if u0 is not None and X is not None and labels is not None:
    #     plot_iter(u0, X, labels, labeled + al_choices_full, subplot=True)
    #     plt.scatter(X[al_choices_full,0], X[al_choices_full,1], marker='s', c='g', alpha=1.0, label='query choices')
    #     plt.title(r"Nonzero entries of $\mathbf{q}$")
    #     plt.legend()
    #     plt.show()
    #
    #     plot_iter(u0, X, labels, labeled + al_choices, subplot=True)
    #     plt.scatter(X[al_choices,0], X[al_choices,1], marker='s', c='g', alpha=1.0, label='query choices')
    #     plt.title(r"Top B entries of $\mathbf{q}$")
    #     plt.legend()
    #     plt.show()
    return list(al_choices)


# Objective function evaluation
def J(u, labeled, unlabeled, y, q, Lt, gamma=0.1):
    return 0.5*np.inner(u, Lt @ u) + np.sum([np.log(1. + np.exp(-u[j]*y[i])/gamma) for i,j in enumerate(labeled)])  \
                + np.sum([q[i]*np.log(1. + np.exp(abs(u[k])/gamma)) for i,k in enumerate(unlabeled)])

################################################################################
##################### Proximal Calculations for Probit Loss functions ############
####################################################################################
def proxg_labeled(v, y, t, gamma=0.1, max_iter=10, tol=1e-7):
    '''
    v and y need to be same size
    '''
    u = np.zeros_like(v)
    it, diff = 0, 1.
    while it < max_iter and diff > tol:
        # take Newton's step
        a, b = t*np.exp(u*y/gamma), gamma*(1. + np.exp(u*y/gamma))
        up1 = (u*a + v*b**2. + t*y*b)/(a + b**2.)

        diff = np.linalg.norm(up1 - u, ord=np.inf)
        u = up1
        it += 1

    if it == max_iter:
        print("\tproxg_labeled did not converge in %d iterations.. diff = %1.8f" % (max_iter, diff))

    return u


def proxg_unlabeled_qconst(v, q, t, gamma=0.1, max_iter=10, tol=1e-7, debug=False):
    '''
    We assume that v and q are of the same size (# of unlabeled nodes).
    '''
    if v.shape != q.shape:
        raise ValueError("Inputs v %s and q %s should be of the same size, but are not." % (str(v.shape), str(q.shape)))
    u = np.zeros_like(v)
    not_converge = []
    for k in range(q.shape[0]):
        if np.allclose(q[k], 0.0):# if the corresponding entry of q is 0, we just input v[k]
            u[k] = v[k]
        else:
            if not np.absolute(v[k]) <= t*q[k]/(2.*gamma):
                it, diff = 0, 1.
                uk = u[k]
                # u[k] > 0 case
                if v[k] > t*q[k]/(2.*gamma):
                    while it < max_iter and diff > tol:
                        a, b = t*q[k]*np.exp(-uk/gamma), gamma*(1. + np.exp(-uk/gamma))
                        uk1 = (uk*a + v[k]*b**2. - t*q[k]*b)/(b**2. + a)
                        diff = abs(uk1 - uk)
                        uk = uk1
                        it += 1
                # u[k] < 0 case
                else:
                    while it < max_iter and diff > tol:
                        a, b = t*q[k]*np.exp(uk/gamma), gamma*(1. + np.exp(uk/gamma))
                        uk1 = (uk*a + v[k]*b**2. + t*q[k]*b)/(b**2. + a)
                        diff = abs(uk1 - uk)
                        uk = uk1
                        it += 1

                u[k] = uk

                if it == max_iter and debug:
                    not_converge.append((k, diff))

    if debug:
        if len(not_converge) > 0:
            print("%d entries did not converge to tol=%1.10f" %(len(not_converge), tol))
        return u, not_converge

    return u


def proxg_unlabeled_both(v, p, t, gamma=0.1, max_iter=10, tol=1e-7, debug=False):
    '''
    We assume that v and p are of the same size (# of unlabeled nodes).
    ##### NOTE: problem becomes bad when have t >= gamma. No check of this is performed here
    '''
    u = np.zeros_like(v)
    not_converge = []
    flipped = []
    for k in range(p.shape[0]):
        if not abs(v[k]) <= abs(t*p[k] - t**2.*np.log(2.))/(2.*gamma):
            #print("case 1")
            it, diff = 0, 1.
            uk = u[k]
            # Case 1 : v[k] > 0 ==> u_k >0
            if p[k] > t*np.log(2):
                # u[k] > 0 case
                if v[k] > abs(t*p[k] - t**2.*np.log(2.))/(2.*gamma):
                    while it < max_iter and diff > tol:
                        a, b = t*p[k]*np.exp(-uk/gamma), gamma*(1. + np.exp(-uk/gamma))
                        uk1 = (uk*a + v[k]*b**2. - t*p[k]*b)/(b**2. + a)
                        diff = abs(uk1 - uk)
                        uk = uk1
                        it += 1
                # u[k] < 0 case
                else:
                    while it < max_iter and diff > tol:
                        a, b = t*p[k]*np.exp(uk/gamma), gamma*(1. + np.exp(uk/gamma))
                        uk1 = (uk*a + v[k]*b**2. + t*p[k]*b)/(b**2. + a)
                        diff = abs(uk1 - uk)
                        uk = uk1
                        it += 1
            else: # Now the signs are guaranteed flip ?
                #print("case 2")
                if debug:
                    flipped.append(k)
                # u[k] < 0 case
                if v[k] < -abs(t*p[k] - t**2.*np.log(2.))/(2.*gamma):
                    while it < max_iter and diff > tol:
                        a, b = t*p[k]*np.exp(-uk/gamma), gamma*(1. + np.exp(-uk/gamma))
                        uk1 = (uk*a + v[k]*b**2. - t*p[k]*b)/(b**2. + a)
                        diff = abs(uk1 - uk)
                        uk = uk1
                        it += 1
                # u[k] > 0 case
                else:
                    while it < max_iter and diff > tol:
                        a, b = t*p[k]*np.exp(uk/gamma), gamma*(1. + np.exp(uk/gamma))
                        uk1 = (uk*a + v[k]*b**2. + t*p[k]*b)/(b**2. + a)
                        diff = abs(uk1 - uk)
                        uk = uk1
                        it += 1


            u[k] = uk

            if it == max_iter and debug:
                not_converge.append((k, diff))

    q = p - t*np.log(1. + np.exp(np.absolute(u)/gamma))

    if debug:
        if len(not_converge) > 0:
            print("%d entries did not converge to tol=%1.10f" %(len(not_converge), tol))
        if len(flipped) > 0:
            print("%d entries flipped " % (len(flipped)))
        return u, q, not_converge, flipped

    return u, q


##################################################################################
################ Code for the projection onto the set \mathcal{B} ################
#################################################################################
def F(X, lam):
    proj = np.array([x - lam if lam <= x and x <= lam + 1 else ( 0 if lam > x  else 1.) for x in X])
    return np.sum(proj)

def P_B(x, B, maxiter=20, verbose=False):
    xfs = np.sort(np.concatenate((x, x-1.))) # x "full" sorted

    # Conduct the binary search for optimal lambda
    lamstar = None
    l, r = 0, 2*x.shape[0]-1
    sl, sr = F(x, xfs[l]), F(x, xfs[r])

    # we are certain sl = n != B and sr = 0 != B
    it = 0
    while r - l > 1 and it < maxiter:
        it += 1
        m = (l+r)//2
        sm = F(x, xfs[m])
        if verbose:
            print("iter = %d, sl=%1.4f, sm=%1.4f, sr=%1.4f" %(it, sl, sm, sr))

        if sm < B:
            r = m
            sr = sm
        elif sm > B:
            l = m
            sl = sm
        else:
            lamstar = xfs[m]
            break

    if it == maxiter:
        print('Projection onto B did not converge in %d iterations..' % maxiter)
        print("sl=%1.4f, sm=%1.4f, sr=%1.4f" %(sl, sm, sr))

    if lamstar is None:
        lamstar = xfs[l] + (xfs[r] - xfs[l])*(sl - float(B))/(sl - sr)


    # Calculate our projection
    y = np.array([xk - lamstar if lamstar <= xk and xk <= lamstar + 1 else ( 0 if lamstar > xk  else 1.) for xk in x])
    return y





def admm_acquisition(Lt, y, labeled, B, u0=None, q0=None, gamma=0.1, rho=50., \
            rho_delta=1.1, debug=True, tol=1e-7, max_iter=100, mu=10., X=None, labels=None):
    '''
    ADMM Batch Mode Method for Probit Model

    # ASSUMING Lt is a sparse matrix for sparse linear solve

    '''
    # Print ADMM hyperparameters
    print("ADMM hyperparameters:")
    print("B = %d" % B)
    print("rho = %1.6f" % rho)
    print("rho-delta const = %1.4f" % rho_delta)
    print("mu = %1.6f" % mu)
    print("tol = %1.9f" % tol)
    print("max_iter = %d" % max_iter)
    if u0 is not None:
        print("u0 was given")
    else:
        print("u0 not given, giving random starting point")
    if q0 is not None:
        print("q0 was given")
    else:
        print("q0 not given, giving random starting point")
    print("debug = %s" % str(debug))

    N, Nu = Lt.shape[0], Lt.shape[0] - len(labeled)
    unlabeled = list(filter(lambda x: x not in labeled, range(N)))

    if u0 is None:
        np.random.seed(67)
        u0 = np.random.randn(N)
        # Ensure that u0 agrees with labeled nodes on at least the sign
        for i,j in enumerate(labeled):
            if u0[j]*y[i] < 0:
                u0[j] *= -1.

    if q0 is None:
        # np.random.seed(12)
        # q0 = np.random.rand(Nu)
        # q0 *= B/np.sum(q0)
        q0 = np.ones(Nu)*B/float(Nu)

    # instantiate optimization variables and hyperparameters
    #np.random.seed(48)
    un, qn, vn, pn, zn, wn = u0.copy(), q0.copy(), u0.copy(), q0.copy(), np.random.randn(N), np.random.rand(Nu)
    it = 0
    rho_change = True
    print_it = max_iter // 10
    JJ = []
    # U, Q, V, P, Z, W = np.empty((N, max_iter)), np.empty((Nu, max_iter)), np.empty((N, max_iter)), \
    #                 np.empty((Nu, max_iter)), np.empty((N, max_iter)), np.empty((Nu, max_iter))
    tic = time.clock()

    while it < max_iter:
        JJ.append(J(un, labeled, unlabeled, y, qn, Lt, gamma=gamma ))
        # U[:,it] = un
        # Q[:,it] = qn
        # V[:,it] = vn
        # P[:,it] = pn
        # Z[:,it] = zn
        # W[:,it] = wn
        if it % print_it == 0:
            print("iteration %d" % (it + 1))

        # Update in u and q
        if rho_change:
            Lt_rho = Lt + sp.sparse.diags(N*[rho])
            rho_change = False
        #print("1/rho = %1.7f" % (1./rho))


        unp1 = sp.sparse.linalg.spsolve(Lt_rho, rho*vn - zn)
        qnp1 = P_B(pn - wn/rho, B)

        # Update in v and p
        vtmp, ptmp = unp1 + zn/rho, qnp1 + wn/rho
        vnp1 = vn.copy()
        vnp1[labeled] = proxg_labeled(vtmp[labeled], y, 1./rho, gamma=gamma, max_iter=10)
        if debug:
            vnp1[unlabeled], pnp1, not_converge, flipped = proxg_unlabeled_both(vtmp[unlabeled], ptmp, 1./rho, gamma=gamma, debug=True)
        else:
            vnp1[unlabeled], pnp1 = proxg_unlabeled_both(vtmp[unlabeled], ptmp, 1./rho, gamma=gamma, debug=False)

        # Update dual variables z and w
        znp1, wnp1 = zn + rho*(unp1 - vnp1), wn + rho*(qnp1 - pnp1)


        # Compute the primal and dual residuals to adjust the rho parameter
        r, s = np.linalg.norm(np.hstack((unp1 - vnp1, qnp1 - pnp1))), rho*np.linalg.norm(np.hstack((vnp1 - vn, pnp1 - pn)))
        # if r > mu*s:
        #     rho *= rho_delta
        #     rho_change = True
        # elif s > mu*r:
        #     rho /= rho_delta
        #     rho_change = True


        # Stopping criterion
        if r < tol and s < tol:
            print("Converged at iteration %d" % (it + 1))
            break

        # Update optimization variables
        un, vn, qn, pn, zn, wn = unp1, vnp1, qnp1, pnp1, znp1, wnp1

        if it % (max_iter//2) == 0 and X is not None and it != 0:
            # plt.hist(qn[qn > 0.], bins = 30)
            # plt.title("Histogram of qn")
            # plt.show()
            new_choices = list(np.array(unlabeled)[(-qn).argsort()[:B]])
            full_new_choices = list(np.array(unlabeled)[np.where(qn > 0.002)[0]])
            plot_iter(u0, X, labels, labeled + full_new_choices, subplot=True)
            plt.scatter(X[full_new_choices,0], X[full_new_choices,1], marker='s', c='g', alpha=1.0, label='nnz in q')
            plt.title(r"Nonzero entries of $\mathbf{q}$")
            plt.legend()
            plt.show()
            plot_iter(u0, X, labels, labeled + new_choices, subplot=True)
            plt.scatter(X[new_choices,0], X[new_choices,1], marker='s', c='g', alpha=1.0, label='query choices')
            plt.title(r"Top B entries of $\mathbf{q}$")
            plt.legend()
            plt.show()

            # print(np.linalg.norm(Q[:,it] - Q[:,it-1]), np.linalg.norm(P[:,it] - P[:,it-1]))
            # plt.subplot(1,2,1)
            # plt.scatter(range(Nu), Q[:,it], marker='.', label='curr')
            # plt.scatter(range(Nu), Q[:,it-1], marker='.', label='prev')
            # plt.legend()
            # plt.subplot(1,2,2)
            # plt.scatter(range(Nu), P[:,it], marker='.', label='curr')
            # plt.scatter(range(Nu), P[:,it-1], marker='.', label='prev')
            # plt.legend()
            # plt.show()
            # print()

        it += 1


    toc = time.clock()
    print("ADMM took %1.6f seconds" % (toc - tic))
    # print("Saving npz file")
    # np.savez('./admm_batch_atom.npz', U=U, V=V, Q=Q, P=P, Z=Z, W=W)
    al_choices = list(np.array(unlabeled)[(-qn).argsort()[:B]])
    al_choices_full = list(np.array(unlabeled)[np.where(qn > 0.002)[0]])


    print("ADMM J(u,q) = %1.6f" % J(un, labeled, unlabeled, y, qn, Lt, gamma=gamma ))
    print("\tmin J val = %1.6f" % np.min(JJ))
    plt.scatter(range(len(JJ)), JJ, marker='x', c='b')
    plt.plot(range(len(JJ)), JJ,'b')
    plt.xlabel('iter')
    plt.ylabel('J(u,q) at iter')
    plt.title("J values over iterations")
    plt.show()

    plot_iter(u0, X, labels, labeled + al_choices, subplot=True)
    plt.scatter(X[al_choices_full,0], X[al_choices_full,1], marker='s', c='g', alpha=1.0, label='query choices')
    plt.title(r"Nonzero entries of $\mathbf{q}$")
    plt.legend()
    plt.show()
    plot_iter(u0, X, labels, labeled + al_choices, subplot=True)
    plt.scatter(X[al_choices,0], X[al_choices,1], marker='s', c='g', alpha=1.0, label='query choices')
    plt.title(r"Top B entries of $\mathbf{q}$")
    plt.legend()
    plt.show()


    return al_choices, al_choices_full

def admm_acquisition_rand(Lt, y, labeled, B, u0=None, q0=None, gamma=0.1, rho=50., \
            rho_delta=1.1, debug=True, tol=1e-7, max_iter=100, mu=10., X=None, labels=None):
    '''
    ADMM Batch Mode Method for Probit Model

    # ASSUMING Lt is a sparse matrix for sparse linear solve

    '''
    # Print ADMM hyperparameters
    print("ADMM-rand hyperparameters:")
    print("B = %d" % B)
    print("rho = %1.6f" % rho)
    print("rho-delta const = %1.4f" % rho_delta)
    print("mu = %1.6f" % mu)
    print("tol = %1.9f" % tol)
    print("max_iter = %d" % max_iter)
    if u0 is not None:
        print("u0 was given")
    else:
        print("u0 not given, giving random starting point")
    if q0 is not None:
        print("q0 was given")
    else:
        print("q0 not given, giving random starting point")
    print("debug = %s" % str(debug))

    N, Nu = Lt.shape[0], Lt.shape[0] - len(labeled)
    unlabeled = list(filter(lambda x: x not in labeled, range(N)))

    if u0 is None:
        np.random.seed(67)
        u0 = np.random.randn(N)
        # Ensure that u0 agrees with labeled nodes on at least the sign
        for i,j in enumerate(labeled):
            if u0[j]*y[i] < 0:
                u0[j] *= -1.

    if q0 is None:
        #np.random.seed(12)
        np.random.seed(36)
        q0 = np.random.rand(Nu)
        q0 *= B/np.sum(q0)
        #q0 = np.ones(Nu)*B/float(Nu)
        plt.scatter(range(Nu), q0)
        plt.show()

    # instantiate optimization variables and hyperparameters
    #np.random.seed(48)
    un, qn, vn, pn, zn, wn = u0.copy(), q0.copy(), u0.copy(), q0.copy(), np.random.randn(N), np.random.rand(Nu)
    it = 0
    rho_change = True
    print_it = max_iter // 10
    JJ = []
    # U, Q, V, P, Z, W = np.empty((N, max_iter)), np.empty((Nu, max_iter)), np.empty((N, max_iter)), \
    #                 np.empty((Nu, max_iter)), np.empty((N, max_iter)), np.empty((Nu, max_iter))
    tic = time.clock()

    while it < max_iter:
        JJ.append(J(un, labeled, unlabeled, y, qn, Lt, gamma=gamma ))
        # U[:,it] = un
        # Q[:,it] = qn
        # V[:,it] = vn
        # P[:,it] = pn
        # Z[:,it] = zn
        # W[:,it] = wn
        if it % print_it == 0:
            print("iteration %d" % (it + 1))

        # Update in u and q
        if rho_change:
            Lt_rho = Lt + sp.sparse.diags(N*[rho])
            rho_change = False
        #print("1/rho = %1.7f" % (1./rho))


        unp1 = sp.sparse.linalg.spsolve(Lt_rho, rho*vn - zn)
        qnp1 = P_B(pn - wn/rho, B)

        # Update in v and p
        vtmp, ptmp = unp1 + zn/rho, qnp1 + wn/rho
        vnp1 = vn.copy()
        vnp1[labeled] = proxg_labeled(vtmp[labeled], y, 1./rho, gamma=gamma, max_iter=10)
        if debug:
            vnp1[unlabeled], pnp1, not_converge, flipped = proxg_unlabeled_both(vtmp[unlabeled], ptmp, 1./rho, gamma=gamma, debug=True)
        else:
            vnp1[unlabeled], pnp1 = proxg_unlabeled_both(vtmp[unlabeled], ptmp, 1./rho, gamma=gamma, debug=False)

        # Update dual variables z and w
        znp1, wnp1 = zn + rho*(unp1 - vnp1), wn + rho*(qnp1 - pnp1)


        # Compute the primal and dual residuals to adjust the rho parameter
        r, s = np.linalg.norm(np.hstack((unp1 - vnp1, qnp1 - pnp1))), rho*np.linalg.norm(np.hstack((vnp1 - vn, pnp1 - pn)))
        # if r > mu*s:
        #     rho *= rho_delta
        #     rho_change = True
        # elif s > mu*r:
        #     rho /= rho_delta
        #     rho_change = True


        # Stopping criterion
        if r < tol and s < tol:
            print("Converged at iteration %d" % (it + 1))
            break

        # Update optimization variables
        un, vn, qn, pn, zn, wn = unp1, vnp1, qnp1, pnp1, znp1, wnp1

        if it % (max_iter//2) == 0 and X is not None and it != 0:
            # plt.hist(qn[qn > 0.], bins = 30)
            # plt.title("Histogram of qn")
            # plt.show()
            new_choices = list(np.array(unlabeled)[(-qn).argsort()[:B]])
            full_new_choices = list(np.array(unlabeled)[np.where(qn > 0.002)[0]])
            plot_iter(u0, X, labels, labeled + full_new_choices, subplot=True)
            plt.scatter(X[full_new_choices,0], X[full_new_choices,1], marker='s', c='g', alpha=1.0, label='query choices')
            plt.title(r"Nonzero entries of $\mathbf{q}$")
            plt.legend()
            plt.show()
            plot_iter(u0, X, labels, labeled + new_choices, subplot=True)
            plt.scatter(X[new_choices,0], X[new_choices,1], marker='s', c='g', alpha=1.0, label='query choices')
            plt.title(r"Top B entries of $\mathbf{q}$")
            plt.legend()
            plt.show()

            # print(np.linalg.norm(Q[:,it] - Q[:,it-1]), np.linalg.norm(P[:,it] - P[:,it-1]))
            # plt.subplot(1,2,1)
            # plt.scatter(range(Nu), Q[:,it], marker='.', label='curr')
            # plt.scatter(range(Nu), Q[:,it-1], marker='.', label='prev')
            # plt.legend()
            # plt.subplot(1,2,2)
            # plt.scatter(range(Nu), P[:,it], marker='.', label='curr')
            # plt.scatter(range(Nu), P[:,it-1], marker='.', label='prev')
            # plt.legend()
            # plt.show()
            # print()

        it += 1


    toc = time.clock()
    print("ADMM-rand took %1.6f seconds" % (toc - tic))

    # hist, bin_edges = np.histogram(qn[qn>0.], bins=25)
    # right_peak = (-hist).argsort()[1]
    # val = bin_edges[right_peak + 1]
    val = np.max(qn)/2. + 0.001
    print(val)
    plt.hist(qn[qn > 0.], bins=50)
    plt.show()
    #al_choices = list(np.array(unlabeled)[(-qn).argsort()[:B]])
    qn_nnz_norm = qn[np.where(qn > val)[0]]
    qn_nnz_norm = np.exp(3.*qn_nnz_norm)
    qn_nnz_norm /= np.sum(qn_nnz_norm)

    al_choices_nu = np.random.choice(np.where(qn > val)[0], B, p=qn_nnz_norm, replace=False )
    plt.scatter(range(Nu), qn, marker='.')
    plt.scatter(al_choices_nu, qn[al_choices_nu], c='b', marker='x')
    plt.show()
    al_choices = list(np.array(unlabeled)[al_choices_nu])
    al_choices_full = list(np.array(unlabeled)[np.where(qn > 0.002)[0]])


    print("ADMM rand J(u,q) = %1.6f" % J(un, labeled, unlabeled, y, qn, Lt, gamma=gamma ))
    print("\tmin J val = %1.6f" % np.min(JJ))
    plt.scatter(range(len(JJ)), JJ, marker='x', c='b')
    plt.plot(range(len(JJ)), JJ,'b')
    plt.xlabel('iter')
    plt.ylabel('J(u,q) at iter')
    plt.title("J values over iterations")
    plt.show()

    plot_iter(u0, X, labels, labeled + al_choices, subplot=True)
    plt.scatter(X[al_choices_full,0], X[al_choices_full,1], marker='s', c='g', alpha=1.0, label='query choices')
    plt.title(r"Nonzero entries of $\mathbf{q}$")
    plt.legend()
    plt.show()
    plot_iter(u0, X, labels, labeled + al_choices, subplot=True)
    plt.scatter(X[al_choices,0], X[al_choices,1], marker='s', c='g', alpha=1.0, label='query choices')
    plt.title(r"Top B entries of $\mathbf{q}$")
    plt.legend()
    plt.show()


    return al_choices, al_choices_full


def bcd_exactq_acquisition(Lt, y, labeled, B, u0=None, q0=None, gamma=0.1, alpha0=1.0, \
        alpha_delta=2., debug=True, tol=1e-7, max_iter=100, X=None, labels=None):
    '''

    It does look like once you choose your "q_ind", the indices where f_vec is smallest, then it will stick with those values
    thereafter. Theory to show that this is always the case? the objective makes sure that unp1 goes down at the entries of
    q_ind, because all the other ones zeroed out...
    '''
    # Print hyperparameters
    print("BCD \"Exact\" hyperparameters:")
    print("B = %d" % B)
    print("alpha0 = %1.6f" % alpha0)
    print("alpha_delta = %1.4f" % alpha_delta)
    print("tol = %1.9f" % tol)
    print("max_iter = %d" % max_iter)
    if u0 is not None:
        print("u0 was given")
    else:
        print("u0 not given, giving random starting point")
    if q0 is not None:
        print("q0 was given")
    else:
        print("q0 not given, giving random starting point")
    print("debug = %s" % str(debug))

    N, Nu = Lt.shape[0], Lt.shape[0] - len(labeled)
    unlabeled = list(filter(lambda x: x not in labeled, range(N)))

    if u0 is None:
        np.random.seed(67)
        u0 = np.random.randn(N)
        # Ensure that u0 agrees with labeled nodes on at least the sign
        for i,j in enumerate(labeled):
            if u0[j]*y[i] < 0:
                u0[j] *= -1.

    if q0 is None:
        # np.random.seed(12)
        # q0 = np.random.rand(Nu)
        # q0 *= B/np.sum(q0)
        q0 = np.ones(Nu)*B/float(Nu)


    vn, un, qn = u0.copy(), u0.copy(), q0.copy()
    it = 0
    alpha_n = alpha0
    alpha_change = True
    q_ind_prev = {-i for i in range(B)}
    print_it = max_iter // 10
    JJ = []
    tic = time.clock()

    while it < max_iter:
        JJ.append(J(un, labeled, unlabeled, y, qn, Lt, gamma=gamma ))
        if it % print_it == 0:
            print("iteration %d" % (it + 1))

        if alpha_change:
            L_a = Lt + sp.sparse.diags(N*[alpha_n])
            alpha_change = False

        # Update in v - sparse linear solve
        vnp1 = sp.sparse.linalg.spsolve(L_a, alpha_n*un)

        # Update in u - proxg calculations, q const on unlabeled
        unp1 = un.copy()
        unp1[labeled] = proxg_labeled(vnp1[labeled], y, 1./alpha_n, gamma=gamma)
        if debug:
            unp1[unlabeled], not_converge = proxg_unlabeled_qconst(vnp1[unlabeled], qn, 1./alpha_n, gamma=gamma, debug=True)
        else:
            unp1[unlabeled] = proxg_unlabeled_qconst(vnp1[unlabeled], qn, 1./alpha_n, gamma=gamma, debug=False)
        # if it % 5 == 0:
        #     plt.scatter(range(N), vn, marker='.', label='vn')
        #     plt.scatter(range(N), vnp1, marker='.', label='vnp1')
        #     plt.legend()
        #     plt.title('v')
        #     plt.show()
        #
        #     plt.scatter(range(N), un, marker='.', label='un')
        #     plt.scatter(range(N), unp1, marker='.', label='unp1')
        #     plt.legend()
        #     plt.title('u')
        #     plt.show()

        # Update in q - Exact B entries
        f_vec = np.log(1. + np.exp(np.absolute(unp1[unlabeled])/gamma)) # Should be equivalent to look just at absolute(unp1[unlabeled])-- i.e. the decision boundary...
        #q_ind = (np.absolute(unp1[unlabeled])).argsort()[:B]
        #f_vec = np.absolute(unp1[unlabeled])
        mm = np.min(f_vec)
        #print(len(np.where(f_vec == mm)[0]))
        if len(np.where(f_vec == mm)[0]) > B:
            q_ind = np.where(f_vec == mm)[0]
            q_ind = list(np.random.choice(q_ind, B, replace=False))
        else:
            q_ind = (f_vec).argsort()[:B]
        qnp1 = np.zeros(Nu)
        qnp1[q_ind] = 1.
        # plt.subplot(1,2,1)
        # plt.scatter(range(Nu), f_vec1, marker='.')
        # plt.scatter(q_ind, f_vec1[q_ind], marker='x', c='b')
        # mmm = np.min(f_vec1[np.where(f_vec1 > 0.)[0]])
        # plt.ylim(mmm-0.001,mmm+0.001)
        # plt.title('f1')
        # plt.subplot(1,2,2)
        # plt.scatter(range(Nu), f_vec, marker='.')
        # plt.scatter(q_ind, f_vec[q_ind], marker='x', c='b')
        # mmm = np.min(f_vec[np.where(f_vec > 0.)[0]])
        # plt.ylim(mmm-0.001, mmm+0.002)
        # plt.title("iter %d" % it)
        # plt.show()



        # Check stopping criterion
        vdiff, udiff, qdiff = np.linalg.norm(vnp1 - vn, ord=np.inf), np.linalg.norm(unp1 - un, ord=np.inf), len(set(q_ind).difference(q_ind_prev))
        if vdiff < tol and udiff < tol and qdiff == 0:
            break

        # Increase alpha_n by factor
        if np.linalg.norm(unp1 - vnp1) > tol:
            alpha_n *= alpha_delta
            alpha_change = True

        vn, un, qn = vnp1, unp1, qnp1
        q_ind_prev = set(q_ind)


        if it % (max_iter//2) == 0 and X is not None and it != 0:
            new_choices = list(np.array(unlabeled)[(-qn).argsort()[:B]])
            full_new_choices = list(np.array(unlabeled)[np.where(qn > 0.002)[0]])
            plot_iter(u0, X, labels, labeled + full_new_choices, subplot=True)
            plt.scatter(X[full_new_choices,0], X[full_new_choices,1], marker='s', c='g', alpha=1.0, label='query choices')
            plt.title(r"Nonzero entries of $\mathbf{q}$")
            plt.legend()
            plt.show()
            plot_iter(u0, X, labels, labeled + new_choices, subplot=True)
            plt.scatter(X[new_choices,0], X[new_choices,1], marker='s', c='g', alpha=1.0, label='query choices')
            plt.title(r"Top B entries of $\mathbf{q}$")
            plt.legend()
            plt.show()
        #     print("iter %d" %(it+1))
        #     print(q_ind_prev)
        #     new_choices = list(np.array(unlabeled)[(-qn).argsort()[:B]])
        #     full_new_choices = list(np.array(unlabeled)[np.where(qn > 0.)[0]])
        #     plot_iter(u0, X, labels, labeled + full_new_choices, subplot=True)
        #     plt.scatter(X[full_new_choices,0], X[full_new_choices,1], marker='s', c='g', alpha=1.0, label='query choices')
        #     plt.title(r"Nonzero entries of $\mathbf{q}$")
        #     plt.legend()
        #     plt.show()
        #     plot_iter(u0, X, labels, labeled + new_choices, subplot=True)
        #     plt.scatter(X[new_choices,0], X[new_choices,1], marker='s', c='g', alpha=1.0, label='query choices')
        #     plt.title(r"Top B entries of $\mathbf{q}$")
        #     plt.legend()
        #     plt.show()

        it += 1



    print("BCD-Exact converged in %d iterations" % (it +1))
    toc = time.clock()
    print("BCD-Exact took %1.6f seconds" % (toc - tic))
    al_choices = list(np.array(unlabeled)[(-qn).argsort()[:B]])
    al_choices_full = list(np.array(unlabeled)[np.where(qn > 0.002)[0]])


    print("BCD-Exact J(u,q) = %1.6f" % J(un, labeled, unlabeled, y, qn, Lt, gamma=gamma ))
    print("\tmin J val = %1.6f" % np.min(JJ))
    plt.scatter(range(len(JJ)), JJ, marker='x', c='b')
    plt.plot(range(len(JJ)), JJ,'b')
    plt.xlabel('iter')
    plt.ylabel('J(u,q) at iter')
    plt.title("J values over iterations")
    plt.show()

    plot_iter(u0, X, labels, labeled + al_choices, subplot=True)
    plt.scatter(X[al_choices_full,0], X[al_choices_full,1], marker='s', c='g', alpha=1.0, label='query choices')
    plt.title(r"Nonzero entries of $\mathbf{q}$")
    plt.legend()
    plt.show()
    plot_iter(u0, X, labels, labeled + al_choices, subplot=True)
    plt.scatter(X[al_choices,0], X[al_choices,1], marker='s', c='g', alpha=1.0, label='query choices')
    plt.title(r"Top B entries of $\mathbf{q}$")
    plt.legend()
    plt.show()

    return al_choices, al_choices_full


def bcd_proxq_acquisition(Lt, y, labeled, B, u0=None, q0=None, gamma=0.1, alpha0=1.0, alpha_delta=2., beta0 = 1.0, \
                    beta_delta=1., debug=True, tol=1e-7, max_iter=100, X=None, labels=None):
    # Print hyperparameters
    print("BCD \"Prox\" hyperparameters:")
    print("B = %d" % B)
    print("alpha0 = %1.6f" % alpha0)
    print("alpha_delta = %1.4f" % alpha_delta)
    print("beta0 = %1.6f" % beta0)
    print("beta_delta = %1.4f" % beta_delta)
    print("tol = %1.9f" % tol)
    print("max_iter = %d" % max_iter)
    if u0 is not None:
        print("u0 was given")
    else:
        print("u0 not given, giving random starting point")
    if q0 is not None:
        print("q0 was given")
    else:
        print("q0 not given, giving random starting point")
    print("debug = %s" % str(debug))

    N, Nu = Lt.shape[0], Lt.shape[0] - len(labeled)
    unlabeled = list(filter(lambda x: x not in labeled, range(N)))

    if u0 is None:
        np.random.seed(67)
        u0 = np.random.randn(N)
        # Ensure that u0 agrees with labeled nodes on at least the sign
        for i,j in enumerate(labeled):
            if u0[j]*y[i] < 0:
                u0[j] *= -1.

    if q0 is None:
        # np.random.seed(12)
        # q0 = np.random.rand(Nu)
        # q0 *= B/np.sum(q0)
        q0 = np.ones(Nu)*B/float(Nu)


    vn, un, qn = u0.copy(), u0.copy(), q0.copy()
    it = 0
    alpha_n, beta_n = alpha0, beta0
    alpha_change, beta_change = True, True
    print_it = max_iter // 10

    old_choices = [i for i in range(B)]
    JJ = []
    tic = time.clock()
    while it < max_iter:
        JJ.append(J(un, labeled, unlabeled, y, qn, Lt, gamma=gamma ))
        if it % print_it == 0:
            print("iteration %d" % (it + 1))

        if alpha_change or beta_change:
            L_ab = Lt + sp.sparse.diags(N*[alpha_n + beta_n])
            alpha_change, beta_change = False, False

        # Update in v - sparse linear solve
        vnp1 = sp.sparse.linalg.spsolve(L_ab, (alpha_n*un + beta_n*vn))

        # Update in u - proxg calculations, q const on unlabeled
        unp1, utmp = un.copy(), (alpha_n*vnp1 + beta_n*un)/(alpha_n + beta_n)
        unp1[labeled] = proxg_labeled(utmp[labeled], y, 1./(alpha_n+beta_n), gamma=gamma)
        if debug:
            unp1[unlabeled], not_converge = proxg_unlabeled_qconst(utmp[unlabeled], qn, 1./(alpha_n + beta_n), gamma=gamma, debug=True)

        else:
            unp1[unlabeled] = proxg_unlabeled_qconst(utmp[unlabeled], qn, 1./(alpha_n + beta_n), gamma=gamma, debug=False)

        # Update in q - Proximal onto \mathcal{B}
        f_vec = np.log(1. + np.exp(np.absolute(unp1[unlabeled])/gamma))
        qnp1 = P_B(qn - f_vec/beta_n, B)

        # if it % 50 == 0:
        #     plt.scatter(range(Nu), qnp1, marker='.')
        #     plt.title('iter %d, qnp1' % it)
        #     plt.show()

        # Check stopping criterion
        vdiff, udiff, qdiff = np.linalg.norm(vnp1 - vn, ord=np.inf), np.linalg.norm(unp1 - un, ord=np.inf), np.linalg.norm(qnp1 - qn, ord=np.inf)
        if max(vdiff, udiff, qdiff) < tol:
            break

        # Increase alpha_n by factor
        if np.linalg.norm(unp1 - vnp1) > tol:
            alpha_n *= alpha_delta
            alpha_change = True

        if np.linalg.norm(qnp1 - qn) > tol:
            beta_n *= beta_delta
            beta_change = True


        # # new_choices = np.sort(list((-qnp1).argsort()[:B]))
        # # if len(set(new_choices).difference(set(old_choices))) > 0:
        # #     print("iteration %d, top choices in qn changed" % (it+1))
        # #     print("old = %s, new = %s" % (str(old_choices), str(new_choices)))
        #
        # if it % 10 == 0 and X is not None:
        #     print("iter %d" %(it+1))
        #     print("vdiff=%1.7f, udiff=%1.7f, qdiff=%1.7f" % (vdiff, udiff, qdiff))
        #     # plt.subplot(1,2,1)
        #     # plt.scatter(range(Nu), un[unlabeled], marker='.')
        #     # plt.scatter(old_choices, un[np.array(unlabeled)[old_choices]], marker='x', c='b')
        #     # plt.title("un")
        #     # plt.subplot(1,2,2)
        #     # plt.scatter(range(Nu), qn - f_vec/beta_n, marker='.')
        #     # plt.scatter(range(Nu), qnp1, marker='.', c='g')
        #     # plt.show()
        #     #print("old_choices = %s" % str(old_choices))
        #     new_choices = list((-qnp1).argsort()[:B])
        #     print("new_choices = %s" % str(np.sort(new_choices)))
        #     new_choices_N = list(np.array(unlabeled)[new_choices])
        #     full_new_choices = list(np.array(unlabeled)[np.where(qnp1 > 0.005)[0]])
        #     qn_norm = qnp1[np.where(qnp1 > 0.005)]/np.max(qnp1[np.where(qnp1 > 0.005)])
        #     plot_iter(u0, X, labels, labeled + full_new_choices, subplot=True)
        #     plt.scatter(X[full_new_choices,0], X[full_new_choices,1], marker='s', c=[(v,1.-0.4*(1.-v), 1.-v) for v in qn_norm], alpha=1.0, label='query choices')
        #     plt.title(r"Nonzero entries of $\mathbf{q}$")
        #     plt.legend()
        #     plt.show()
        #     plot_iter(u0, X, labels, labeled + new_choices_N, subplot=True)
        #     plt.scatter(X[new_choices_N,0], X[new_choices_N,1], marker='s', c='g', alpha=1.0, label='query choices')
        #     plt.title(r"Top B entries of $\mathbf{q}$")
        #     plt.legend()
        #     plt.show()
        #
        #     # plt.scatter(range(Nu), qnp1, marker='.')
        #     # plt.scatter(old_choices, qnp1[old_choices], marker='x',c='b', label='old choices')
        #     # plt.scatter(new_choices, qnp1[new_choices], marker='o', c='g', label='new choices')
        #     # plt.legend()
        #     # #plt.ylim(0.0001,np.max(qnp1)+0.001)
        #     # plt.title('Compare entries in qn')
        #     # plt.show()
        #     # plt.hist(qnp1[qnp1 > 0.], bins=30)
        #     # plt.title("Histogram of qn")
        #     # plt.show()
        #
        # # old_choices = np.sort(new_choices)

        vn, un, qn = vnp1, unp1, qnp1

        if it % (max_iter//2) == 0 and X is not None and it != 0:
            new_choices = list(np.array(unlabeled)[(-qn).argsort()[:B]])
            full_new_choices = list(np.array(unlabeled)[np.where(qn > 0.002)[0]])
            plot_iter(u0, X, labels, labeled + full_new_choices, subplot=True)
            plt.scatter(X[full_new_choices,0], X[full_new_choices,1], marker='s', c='g', alpha=1.0, label='query choices')
            plt.title(r"Nonzero entries of $\mathbf{q}$")
            plt.legend()
            plt.show()
            plot_iter(u0, X, labels, labeled + new_choices, subplot=True)
            plt.scatter(X[new_choices,0], X[new_choices,1], marker='s', c='g', alpha=1.0, label='query choices')
            plt.title(r"Top B entries of $\mathbf{q}$")
            plt.legend()
            plt.show()

        it += 1

    print("BCD-Prox converged in %d iterations" %(it+1))
    toc = time.clock()
    print("BCD-Prox took %1.6f seconds" % (toc - tic))
    mm = np.max(qn)
    qn_max_ind = np.where(qn == mm)[0]
    if len(qn_max_ind) > B:
        al_choices = list(np.array(unlabeled)[np.random.choice(qn_max_ind, B, replace=False)])
    else:
        al_choices = list(np.array(unlabeled)[(-qn).argsort()[:B]])
    al_choices_full = list(np.array(unlabeled)[np.where(qn > 0.002)[0]])

    print("BCD-Prox J(u,q) = %1.6f" % J(un, labeled, unlabeled, y, qn, Lt, gamma=gamma ))
    print("\tmin J val = %1.6f" % np.min(JJ))
    plt.scatter(range(len(JJ)), JJ, marker='x', c='b')
    plt.plot(range(len(JJ)), JJ,'b')
    plt.xlabel('iter')
    plt.ylabel('J(u,q) at iter')
    plt.title("J values over iterations")
    plt.show()

    plot_iter(u0, X, labels, labeled + al_choices, subplot=True)
    plt.scatter(X[al_choices_full,0], X[al_choices_full,1], marker='s', c='g', alpha=1.0, label='query choices')
    plt.title(r"Nonzero entries of $\mathbf{q}$")
    plt.legend()
    plt.show()
    plot_iter(u0, X, labels, labeled + al_choices, subplot=True)
    plt.scatter(X[al_choices,0], X[al_choices,1], marker='s', c='g', alpha=1.0, label='query choices')
    plt.title(r"Top B entries of $\mathbf{q}$")
    plt.legend()
    plt.show()

    return al_choices, al_choices_full


def alt_exact_acquisition(Lt, y, labeled, B, u0=None, q0=None, gamma=0.1, rho=50.0, \
        rho_delta=1.2, debug=True, tol=1e-7, max_iter=10, admm_max_iter=100, admm_tol=1e-7, mu=10., X=None, labels=None):
    # Print hyperparameters
    print("Alternating Minimization hyperparameters:")
    print("B = %d" % B)
    print("tol = %1.9f" % tol)
    print("max_iter = %d" % max_iter)
    if u0 is not None:
        print("u0 was given")
    else:
        print("u0 not given, giving random starting point")
    if q0 is not None:
        print("q0 was given")
    else:
        print("q0 not given, giving random starting point")
    print("debug = %s" % str(debug))
    print("ADMM-inner params:")
    print("\trho = %1.6f" % rho)
    print("\trho-delta const = %1.4f" % rho_delta)
    print("\tmu = %1.6f" % mu)
    print("\tmax_iter = %d" % admm_max_iter)
    print("\ttol = %1.9f" % admm_tol)

    N, Nu = Lt.shape[0], Lt.shape[0] - len(labeled)
    unlabeled = list(filter(lambda x: x not in labeled, range(N)))

    if u0 is None:
        np.random.seed(67)
        u0 = np.random.randn(N)
        # Ensure that u0 agrees with labeled nodes on at least the sign
        for i,j in enumerate(labeled):
            if u0[j]*y[i] < 0:
                u0[j] *= -1.

    if q0 is None:
        # np.random.seed(12)
        # q0 = np.random.rand(Nu)
        # q0 *= B/np.sum(q0)
        q0 = np.ones(Nu)*B/float(Nu)

    # instantiate optimization variables and hyperparameters
    un, qn = u0.copy(), q0.copy()
    it = 0
    q_ind_prev = {-i for i in range(B)}
    print_it = max_iter // 10

    old_choices = [i for i in range(B)]
    JJ = []
    tic = time.clock()

    while it < max_iter:
        JJ.append(J(un, labeled, unlabeled, y, qn, Lt, gamma=gamma ))
        if it % print_it == 0:
            print("iteration %d" % (it + 1))

        # Update in q - Exact B entries
        f_vec = np.log(1. + np.exp(np.absolute(un[unlabeled])/gamma)) # Should be equivalent to look just at absolute(unp1[unlabeled])-- i.e. the decision boundary...
        q_ind = (f_vec).argsort()[:B]
        qnp1 = np.zeros(Nu)
        qnp1[q_ind] = 1.

        # Update in u via ADMM -- First instantiate ADMM iteration variables and hyperparameters
        um, vm, zm = un.copy(), un.copy(), np.random.randn(N)
        rho_change = True
        for itt in range(admm_max_iter):
            if itt % 50 == 0:
                print("Inner ADMM iteration %d" % (itt + 1))
            # Update in um, sparse linear solve
            if rho_change:
                L_rho = Lt + sp.sparse.diags(N*[rho])
                rho_change = False

            ump1 = sp.sparse.linalg.spsolve(L_rho, rho*vm - zm)

            # Update in vm, proximal calculations of g, holding q fixed
            vtmp = ump1 + zm/rho
            vmp1 = vm.copy()
            vmp1[labeled] = proxg_labeled(vtmp[labeled], y, 1./rho, gamma=gamma, max_iter=10)
            if debug:
                vmp1[unlabeled], not_converge = proxg_unlabeled_qconst(vtmp[unlabeled], qnp1, 1./rho, gamma=gamma, debug=True)
            else:
                vmp1[unlabeled] = proxg_unlabeled_qconst(vtmp[unlabeled], qnp1, 1./rho, gamma=gamma, debug=False)

            # Update dual variable zm
            zmp1 = zm + rho*(ump1 - vmp1)

            # Compute the primal and dual residuals to adjust the rho parameter
            r, s = np.linalg.norm(ump1 - vmp1), rho*np.linalg.norm(vmp1 - vm)
            if r > mu*s:
                rho *= rho_delta
                rho_change = True
            if s > mu*r:
                rho /= rho_delta
                rho_change = True


            # Stopping criterion for ADMM iters.
            if r < admm_tol and s < admm_tol:
                print("Converged at iteration %d" % (it + 1))
                break

            # Update optimization variables
            um, vm, zm = ump1, vmp1, zmp1


        # Update u from inner ADMM loop
        unp1 = um


        # Stopping Criterion for Alternating "Exact" Minimization
        udiff, qdiff = np.linalg.norm(unp1 - un), len(set(q_ind).difference(q_ind_prev))

        # new_choices = np.sort(list((-qnp1).argsort()[:B]))
        # if len(set(new_choices).difference(set(old_choices))) > 0:
        #     print("iteration %d, top choices in qn changed" % (it+1))
        #     print("old = %s, new = %s" % (str(old_choices), str(new_choices)))
        #
        # if X is not None:
        #     print("iter %d" %(it+1))
        #     print("udiff=%1.7f, qdiff=%1.7f" % (udiff, qdiff))
        #     print("old_choices = %s" % str(old_choices))
        #     new_choices = list((-qnp1).argsort()[:B])
        #     print("new_choices = %s" % str(np.sort(new_choices)))
        #     new_choices_N = list(np.array(unlabeled)[new_choices])
        #     full_new_choices = list(np.array(unlabeled)[np.where(qnp1 > 0.)[0]])
        #     qn_norm = qnp1[np.where(qnp1 > 0.)]/np.max(qnp1[np.where(qnp1 > 0.)])
        #     plot_iter(u0, X, labels, labeled + full_new_choices, subplot=True)
        #     plt.scatter(X[full_new_choices,0], X[full_new_choices,1], marker='s', c=[(v,1.-0.4*(1.-v), 1.-v) for v in qn_norm], alpha=1.0, label='query choices')
        #     plt.title(r"Nonzero entries of $\mathbf{q}$")
        #     plt.legend()
        #     plt.show()
        #     plot_iter(u0, X, labels, labeled + new_choices_N, subplot=True)
        #     plt.scatter(X[new_choices_N,0], X[new_choices_N,1], marker='s', c='g', alpha=1.0, label='query choices')
        #     plt.title(r"Top B entries of $\mathbf{q}$")
        #     plt.legend()
        #     plt.show()
        #
        #     plt.scatter(range(Nu), qnp1, marker='.')
        #     plt.scatter(old_choices, qnp1[old_choices], marker='x',c='b', label='old choices')
        #     plt.scatter(new_choices, qnp1[new_choices], marker='o', c='g', label='new choices')
        #     plt.legend()
        #     plt.ylim(np.max(qnp1)-0.001,np.max(qnp1)+0.001)
        #     plt.title('Compare entries in qn')
        #     plt.show()
        #
        # old_choices = np.sort(new_choices)


        if udiff < tol and qdiff == 0:
            print("Alternating Minimization converged in %d iterations" % (it+1))
            break




        q_ind_prev = set(q_ind)
        qn, un = qnp1, unp1

        if it % (max_iter//2) == 0 and it != 0 and X is not None :
            new_choices = list(np.array(unlabeled)[(-qn).argsort()[:B]])
            full_new_choices = list(np.array(unlabeled)[np.where(qn > 0.002)[0]])
            plot_iter(u0, X, labels, labeled + full_new_choices, subplot=True)
            plt.scatter(X[full_new_choices,0], X[full_new_choices,1], marker='s', c='g', alpha=1.0, label='query choices')
            plt.title(r"Nonzero entries of $\mathbf{q}$")
            plt.legend()
            plt.show()
            plot_iter(u0, X, labels, labeled + new_choices, subplot=True)
            plt.scatter(X[new_choices,0], X[new_choices,1], marker='s', c='g', alpha=1.0, label='query choices')
            plt.title(r"Top B entries of $\mathbf{q}$")
            plt.legend()
            plt.show()
        it += 1


    toc = time.clock()
    print("Alt-Min took %1.6f seconds" % (toc - tic))
    al_choices = list(np.array(unlabeled)[(-qn).argsort()[:B]])
    al_choices_full = list(np.array(unlabeled)[np.where(qn > 0.002)[0]])

    print("Alt-Min J(u,q) = %1.6f" % J(un, labeled, unlabeled, y, qn, Lt, gamma=gamma ))
    print("\tmin J val = %1.6f" % np.min(JJ))
    plt.scatter(range(len(JJ)), JJ, marker='x', c='b')
    plt.plot(range(len(JJ)), JJ,'b')
    plt.xlabel('iter')
    plt.ylabel('J(u,q) at iter')
    plt.title("J values over iterations")
    plt.show()


    plot_iter(u0, X, labels, labeled + al_choices, subplot=True)
    plt.scatter(X[al_choices_full,0], X[al_choices_full,1], marker='s', c='g', alpha=1.0, label='query choices')
    plt.title(r"Nonzero entries of $\mathbf{q}$")
    plt.legend()
    plt.show()
    plot_iter(u0, X, labels, labeled + al_choices, subplot=True)
    plt.scatter(X[al_choices,0], X[al_choices,1], marker='s', c='g', alpha=1.0, label='query choices')
    plt.title(r"Top B entries of $\mathbf{q}$")
    plt.legend()
    plt.show()

    return al_choices, al_choices_full



def modelchange_batch_acquisition(Lt,y, labeled, B, u0=None, gamma=0.1, C=None, X=None, labels=None):
    if C is None or u0 is None:
        raise ValueError("Model Change requires that u0 = previous MAP estimator and C is posterior covariance matrix")
    N, Nu = Lt.shape[0], Lt.shape[0] - len(labeled)
    m = u0
    unlabeled = list(filter(lambda x: x not in labeled, range(N)))
    tic = time.clock()
    mc = np.array([min(np.absolute(jac_calc2(m[k], -1, gamma)/(1. + C[k,k]*hess_calc2(m[k], -1, gamma ))), \
       np.absolute(jac_calc2(m[k], 1, gamma)/(1. + C[k,k]*hess_calc2(m[k], 1, gamma )))) \
                   * np.linalg.norm(C[k,:]) for k in unlabeled])
    mc_p = (mc - np.min(mc))/(np.max(mc) - np.min(mc))
    avg = np.average(mc_p)
    print("average of mc_p = %1.6f" % avg)
    plt.hist(mc_p, bins=30)
    plt.title('Modelchange histogram')
    plt.show()
    mc_p[mc_p < avg] = 0.
    mc_pn = mc_p/np.sum(mc_p)
    mc_p_exp = np.exp(3.*mc_p)
    mc_p_expn = mc_p_exp/np.sum(mc_p_exp)


    plt.subplot(1,2,1)
    plt.scatter(range(Nu), mc_p, marker='.')
    plt.title('Model Change Prob - linear')
    plt.subplot(1,2,2)
    plt.scatter(range(Nu), mc_p_exp, marker='.')
    plt.title("Model Change Prob - exp")
    plt.show()

    al_choices = list(np.random.choice(unlabeled, B, p= mc_pn, replace=False))
    al_choices_full = list(np.array(unlabeled)[np.where(mc_p > avg)])

    toc = time.clock()
    print("MC-Rand took %1.6f seconds" % (toc - tic) )

    plot_iter(u0, X, labels, labeled + al_choices, subplot=True)
    plt.scatter(X[al_choices_full,0], X[al_choices_full,1], marker='s', c='g', alpha=1.0, label='query choices')
    plt.title(r"Nonzero entries of $\mathbf{q}$")
    plt.legend()
    plt.show()
    plot_iter(u0, X, labels, labeled + al_choices, subplot=True)
    plt.scatter(X[al_choices,0], X[al_choices,1], marker='s', c='g', alpha=1.0, label='query choices')
    plt.title(r"Chosen B entries of $\mathbf{q}$")
    plt.legend()
    plt.show()

    return al_choices, al_choices_full


def modelchange_batch_exact_acquisition(Lt,y, labeled, B, u0=None, gamma=0.1, C=None, X=None, labels=None):
    if C is None or u0 is None:
        raise ValueError("Model Change requires that u0 = previous MAP estimator and C is posterior covariance matrix")
    N, Nu = Lt.shape[0], Lt.shape[0] - len(labeled)
    m = u0
    unlabeled = list(filter(lambda x: x not in labeled, range(N)))
    tic = time.clock()
    mc = np.array([min(np.absolute(jac_calc2(m[k], -1, gamma)/(1. + C[k,k]*hess_calc2(m[k], -1, gamma ))), \
       np.absolute(jac_calc2(m[k], 1, gamma)/(1. + C[k,k]*hess_calc2(m[k], 1, gamma )))) \
                   * np.linalg.norm(C[k,:]) for k in unlabeled])

    avg = np.average(mc)
    al_choices = list(np.array(unlabeled)[(-mc).argsort()[:B]])
    al_choices_full = list(np.array(unlabeled)[np.where(mc > avg)])

    toc = time.clock()
    print("MC-Exact took %1.6f seconds" % (toc - tic) )

    plot_iter(u0, X, labels, labeled + al_choices, subplot=True)
    plt.scatter(X[al_choices_full,0], X[al_choices_full,1], marker='s', c='g', alpha=1.0, label='query choices')
    plt.title(r"Nonzero entries of $\mathbf{q}$")
    plt.legend()
    plt.show()
    plot_iter(u0, X, labels, labeled + al_choices, subplot=True)
    plt.scatter(X[al_choices,0], X[al_choices,1], marker='s', c='g', alpha=1.0, label='query choices')
    plt.title(r"Top B entries of $\mathbf{q}$")
    plt.legend()
    plt.show()

    return al_choices, al_choices_full
