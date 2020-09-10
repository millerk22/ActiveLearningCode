# Active Learner object class
# author: Kevin Miller, ksmill327@gmail.com
#
# Need to implement acquisition_values function more efficiently, record values if debug is on.
import numpy as np
from .dijkstra import *
from .al_util import *
from .acquisition_batch_newest import *


ACQUISITIONS = ['mc', 'uncertainty', 'rand']
MODELS = ['gr', 'probit-log', 'probit-norm']
CANDIDATES = ['rand', 'full', 'dijkstra']
SELECTION_METHODS = ['top', 'prop', '']

def acquisition_values(acq, Cand, model):
        '''
        NOTE : NEED TO STANDARDIZE ALL ACQUISITION FUNCTIONS TO BE MAXIMUM FUNCTIONS.
        '''
    if acq == "mc":
        if model.full_storage:
            vals = mc_full(Cand, model.m, model.C, model.modelname, gamma=model.gamma)
        else:
            vals = mc_reduced(model.C_a, model.alpha, model.v[Cand,:], model.modelname, uks=model.m[Cand], gamma=model.gamma)
    elif acq == "uncertainty":
        vals = -np.absolute(model.m[Cand])  # ensuring a "max" formulation for acquisition values
    elif acq == "random":
        vals = np.random.rand(len(Cand))
    else:
        raise ValueError("Acquisition function %s not yet implemented" % str(acq))

    return vals





class ActiveLearner(object):
    def __init__(self, acquisition='mc', candidate='full', modelname='probit-log', candidate_frac=0.1, W=None, r=None):
        if acquisition not in ACQUISITIONS:
            raise ValueError("Acquisition function name %s not valid, must be in %s" % (str(acquisition), str(ACQUISITIONS)))
        self.acq = acquisition
        if candidate not in CANDIDATES:
            raise ValueError("Candidate Set Selection name %s not valid, must be in %s" % (str(candidate), str(CANDIDATES)))
        self.candidate = candidate
        if (candidate_frac < 0. or candidate_frac > 1. ) and self.candidate == 'rand':
            print("WARNING: Candidate fraction must be between 0 and 1 for 'rand' candidate selection, setting to default 0.1")
            self.candidate_frac = 0.1
        else:
            self.candidate_frac = candidate_frac
        # if modelname not in MODELS:
        #     raise ValueError("Model name %s not valid, must be in %s" % (str(modelname), str(MODELS)))
        # self.modelname = modelname

        if self.candidate == 'dijkstra':
            self.W = W
            if self.W is None:
                raise ValueError("Candidate set selection %s requires W to be non-empty" % candidate)
            self.DIST = {}
            if r is None:
                self.dijkstra_r = 5.0
            else:
                self.dijkstra_r = r
        # else:
        #     # If weight matrix is passed to ActiveLearner but not doing Dijkstra, ignore it
        #     if self.W is not None:
        #         self.W = None


    def select_query_points(self, model, B=1, method='top', prop_func=None, debug=False):
        if method not in SELECTION_METHODS:
            raise ValueError("Selection method %s not valid, must be one of %s" % (method, SELECTION_METHODS))

        # Define the candidate set
        if self.candidate is "rand":
            Cand = np.random.choice(model.unlabeled, size=int(self.candidate_frac * len(model.unlabeled)), replace=False)
        elif self.candidate is "dijkstra":
            raise NotImplementedError("Have not implemented the dikstra candidate selection for this class")
        else:
            Cand = model.unlabeled

        # Compute acquisition values
        acq_vals = acquisition_values(self.acquisition, Cand, self.model)


        # based on selection method, choose query points
        if B == 1:
            if method != 'top':
                print("Warning : B = 1 but election method is not 'top'. Overriding selection method and selecting top choice for query point.")
                return [Cand[np.argmax(acq_vals)]]
        else:
            if method == 'top':
                return [Cand[k] for k in (-x).argsort(acq_vals)[:B]]
                # return [Cand[k] for k in (-x).argsort(acq_vals)[:B]]
            elif method == 'prop':
                if prop_func is None:
                    # if not given a customized proportionality sampling function, use this default.
                    # (1) normalize to be 0 to 1
                    acq_vals = (acq_vals - np.min(acq_vals))/(np.max(acq_vals) - np.min(acq_vals))
                    sigma = 3.
                    p = np.exp(acq_vals/sigma)
                    p /= np.sum(p)
                else:
                    p = prop_func(acq_vals)
                return list(np.random.choice(Cand, B, replace=False, p=p))
            else:
                raise ValueError("Have not implemented this selection method, %s. Somehow got passed other parameter checks...")








"""
    def fit_model(self, m, labeled, y, v=None, d=None, C=None):
        self.m = m
        self.labeled = labeled
        self.y = y
        self.class_ordering = sorted(list(np.unique(y)))
        self.N = m.shape[0]
        if self.N == 0:
            self.N = m.shape[1]
        if v is None and C is None:
            raise ValueError("Both v and C are None, must specify one or the other. Input the full covariance matrix C if you have it, otherwise input the eigenvectors v and diagonal d")
        if C is None:
            self.v = v
            if self.d is None:
                raise ValueError("Need to specify values for diagonal d if are doing reduced storage option")
            self.d = d
            self.reduced_storage = True
        else:
            self.C = C
            self.reduced_storage = False

    def select(self, B, cand_size=self.N, select_by='top', dijkstra_r=None, dijkstra_pct=0.95, dijkstra_bw = 0.5):
        # candidate set selection
        if self.cand == 'full':
            Cand = list(filter(lambda x: x not in self.labeled, range(self.N)))
        elif self.cand == 'rand':
            Cand = list(np.random.choice(list(filter(lambda x: x not in self.labeled, range(self.N))), \
                    cand_size, replace=False))
        else:
            if self.dijkstra_r != dijkstra_r:
                print("Gave a different value of dijkstra radius this time, not sure how will perform.... havent debugged this yet.")
            self.dijkstra_r = dijkstr_r

            if self.DIST == {}:
                STARTS = {}
                for c in self.class_ordering:
                    STARTS[c] = [j for i,j in enumerate(self.labeled) if self.y[i] == c]

                self.DIST = dijkstra_for_al_csr_joint(self.W, STARTS, radius=dijkstra_r, class_ordering=self.class_ordering)
            else:
                ADDED = {}
                for c in self.class_ordering:
                    ADDED[c] = [j for i,j in enumerate(self.Q) if self.yQ[i] == c]
                self.DIST = dijkstra_for_al_update_csr_joint(self.W, self.DIST, ADDED, \
                                self.labeled, radius=self.dijkstra_r, class_ordering=self.class_ordering)

            Cand = []
            if len(self.DIST)/self.N < dijkstra_pct:
                phase1 = True
            else:
                phase1 = False
            for k in self.DIST:
                if k not in self.labeled:
                    mm, MM = min(self.DIST[k]), max(self.DIST[k])
                    if phase1 and MM == np.inf:
                        if mm > dijkstra_r - dijkstra_bw:
                            Cand.append(k)
                    elif phase2 and (MM - mm) < dijkstra_bw:
                        Cand.append(k)


        # active learning query choices selection
        if acq == 'rand':
            return list(np.random.choice(Cand, B, replace=False))

        acq_vals = acquisition_values(self.acquisition, Cand, self.m, C=self.C, v=self.v, d=self.d, \
                reduced=self.reduced_storage, model=self.model)

        if self.acquisition == "mc": # since will do max instead of min
            flip = -1
        else:
            flip = 1

        if select_by == "top":
            return [Cand[i] for i in (flip*acq_vals).argsort()[:B]]
        elif select_by == "prop":
            probs = (acq_vals - np.min(acq_vals))/(np.max(acq_vals) - np.min(acq_vals))
            avg = np.average(probs)
            probs[probs < avg] = 0.
            probs[probs >= avg] = np.exp(3.*probs[probs >= avg])
            probs /= np.sum(probs)
            return list(np.random.choice(Cand, B, replace=False, p=probs))
        else:
            print('havent implemented "greedy" select_by method yet...')
            return []



    def update_model(self, Q, yQ, exact=False):
        '''
        Update the model parameters m, C ( or d)
        '''
        self.Q = Q
        self.yQ = yQ
        self.labeled += self.Q
        self.y += self.yQ

        if exact:
            self.calculate_model(self.labeled, self.y)
        else:
            if self.reduced_storage:
                print("haven't implemented updates for reduced storage mode yet")
            else:
                self.m, self.C = update_m_and_C(self.m, self.C, self.Q, self.yQ,  \
                                gamma=self.gamma, model=self.model)

        return
"""
