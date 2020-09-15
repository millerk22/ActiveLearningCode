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
    if acq == "mc":
        if model.full_storage:
            vals = mc_full(Cand, model.m, model.C, model.modelname, gamma=model.gamma)
        else:
            vals = mc_reduced(model.C_a, model.alpha, model.v[Cand,:], model.modelname, uks=model.m[Cand], gamma=model.gamma)
    elif acq == "uncertainty":
        vals = -np.absolute(model.m[Cand])  # ensuring a "max" formulation for acquisition values
    elif acq == "rand":
        vals = np.random.rand(len(Cand))
    else:
        raise ValueError("Acquisition function %s not yet implemented" % str(acq))

    return vals





class ActiveLearner(object):
    def __init__(self, acquisition='mc', candidate='full', candidate_frac=0.1, W=None, r=None):
        if acquisition not in ACQUISITIONS:
            raise ValueError("Acquisition function name %s not valid, must be in %s" % (str(acquisition), str(ACQUISITIONS)))
        self.acquisition = acquisition
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


    def select_query_points(self, model, B=1, method='top', prop_func=None, prop_sigma=0.8, debug=False):
        if method not in SELECTION_METHODS:
            raise ValueError("Selection method %s not valid, must be one of %s" % (method, SELECTION_METHODS))

        print("Active Learner settings:")
        print("\tacquisition function = %s" % self.acquisition)
        print("\tB = %d" % B)
        print("\tcandidate set = %s" % self.candidate)
        print("\tselection method = %s" % method)

        # Define the candidate set
        if self.candidate is "rand":
            Cand = np.random.choice(model.unlabeled, size=int(self.candidate_frac * len(model.unlabeled)), replace=False)
        elif self.candidate is "dijkstra":
            raise NotImplementedError("Have not implemented the dikstra candidate selection for this class")
        else:
            Cand = model.unlabeled

        # Compute acquisition values
        acq_vals = acquisition_values(self.acquisition, Cand, model)
        if len(acq_vals.shape) > 1:
            print("WARNING: acq_vals is of shape %s, should be one-dimensional. MIGHT CAUSE PROBLEM" % str(acq_vals.shape))

        # based on selection method, choose query points
        if B == 1:
            if method != 'top':
                print("Warning : B = 1 but election method is not 'top'. Overriding selection method and selecting top choice for query point.")
                return [Cand[np.argmax(acq_vals)]]
        else:
            if method == 'top':
                return [Cand[k] for k in (-acq_vals).argsort()[:B]]

            elif method == 'prop':
                if prop_func is None:
                    # if not given a customized proportionality sampling function, use this default.
                    # (1) normalize to be 0 to 1
                    acq_vals = (acq_vals - np.min(acq_vals))/(np.max(acq_vals) - np.min(acq_vals))
                    p = np.exp(acq_vals/prop_sigma)
                    p /= np.sum(p)
                else:
                    p = prop_func(acq_vals)
                if debug:
                    return list(np.random.choice(Cand, B, replace=False, p=p)), p, acq_vals, Cand
                return list(np.random.choice(Cand, B, replace=False, p=p))
            else:
                raise ValueError("Have not implemented this selection method, %s. Somehow got passed other parameter checks...")
