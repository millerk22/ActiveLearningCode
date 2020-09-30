# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import sys
sys.path.append("..")
import numpy as np
import scipy as sp
from util.Graph_manager import Graph_manager
import time
import copy
from util.gbssl import *
from scipy.sparse.csgraph import connected_components
from sklearn.preprocessing import OneHotEncoder

from datasets.data_loaders_mlflow import load_citation_networks

dataset = 'cora'
data_path = f'../datasets/CORA/'

load_citation_networks(data_path, dataset)

dataset = 'citeseer'
data_path = f'../datasets/CITESEER/'


def load_citation_networks(data_path, dataset):
    idx_features_labels = np.genfromtxt("{}{}.content".format(data_path, dataset),
                                    dtype=np.dtype(str))
    labels = idx_features_labels[:, -1]
    enc = OneHotEncoder()
    enc.fit(labels.reshape((-1, 1)))
    onehot_labels = enc.transform(labels.reshape((-1, 1))).todense()

    idx = np.array(idx_features_labels[:, 0], dtype=np.dtype(str))
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(data_path, dataset),
                                    dtype=np.dtype(str))
    edges = np.array(list(map(lambda x: idx_map.get(x, -1), edges_unordered.flatten())),
                    dtype=np.int32).reshape(edges_unordered.shape)
    edges = edges[(edges != -1).all(axis = 1)]
    adj = sp.sparse.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    _, component_labels = connected_components(csgraph=adj, directed=False, return_labels=True)
    largest_component = np.where(component_labels == 0)[0]
    adj = adj[np.ix_(largest_component, largest_component)]
    return adj, labels, onehot_labels


load_citation_networks(data_path, dataset)
