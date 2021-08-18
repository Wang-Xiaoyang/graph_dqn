from networkx.classes.function import nodes
import numpy as np
import gym
import torch
from collections import deque
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

import gc

import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
import networkx as nx

# class GCN(nn.Module):
#     def __init__(self, in_feats, hidden_size):
#         super(GCN, self).__init__()
#         self.conv1 = GraphConv(in_feats, hidden_size)
#         self.linear = nn.Linear(hidden_size, 1)

#     def forward(self, g, inputs):
#         h = self.conv1(dgl.add_self_loop(g), inputs)
#         h = torch.relu(h)
#         h = self.linear(h)
#         return h

class GQN(nn.Module):
    def __init__(self, in_feats, hidden_size):
        super(GQN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size, allow_zero_in_degree=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, g, inputs):
        # Apply graph convolution and activation.
        h = F.relu(self.conv1(g, inputs))
        with g.local_scope():
            g.ndata['h'] = h
            # Calculate graph representation by sum readout.
            # note: this is doing graph polling - check the normalization; check data dimension
            # better pooling options? currently, f(sum h_v (nodes)) = Q(s,a)
            hg = dgl.sum_nodes(g, 'h',)
            return self.linear(hg)

def copy_graph(g, fea_name):
    # copy a graph
    with g.local_scope():
        src = g.edges()[0]
        dst = g.edges()[1]
        num_nodes = g.num_nodes()
        g_copy = dgl.graph((src, dst), num_nodes=num_nodes)
        fea = torch.clone(g.ndata[fea_name])
        g_copy.ndata[fea_name] = fea
    # try it out
    del g
    gc.collect()
    return g_copy

def self_loop_graph(g, a):
    list_nodes = list(range(g.num_nodes()))
    g = dgl.add_edges(g, list_nodes[:-a], list_nodes[:-a])
    return g

# replay buffer using deque
class Memory():
    def __init__(self, max_size=2000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
            
    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), 
                               size=batch_size, 
                               replace=False)
        return [self.buffer[ii] for ii in idx]

# get node features (bin_capacity) for graph
def state_to_bin_capacity(env):
    # env.bin_capacity_n
    # if done = True, env.bin_capacity_n could contain negative numbers
    # only use this function when done = False
    bin_capacity_list = []
    for i in range(env.n_node_types):
        bin_capacity_list.extend(env.bin_capacity_n[i].copy())
    return bin_capacity_list

# get source nodes and destination nodes for graph
def adjacency_vbpp_to_graph(adjacency_mat):
    # this function transfers the adjacency matrix of the vector bin packing env, 
    # which has the size [n_items + n_bins, n_bins], to the format of pair of nodes [u, v]
    # for defining a graph
    n_items = adjacency_mat.shape[0] - adjacency_mat.shape[1]
    a = np.sum(adjacency_mat, axis=1)
    u = np.where(a == 1)[0]
    v = np.zeros(len(u), dtype=int)
    idx = 0
    for i in u:
        v[idx] = np.where(adjacency_mat[i, :] == 1)[0][0] + n_items
    return u, v

# functions used for importing nodes files
def get_node_id_types(name, nodes_file):
    list_node = [item for item in list(nodes_file['# id']) if item[0] == name]
    # remove duplication
    list_node = list(dict.fromkeys(list_node))
    return list_node

def data_to_node_features_no_capacity(data_pd, n_nodes, bandwidth_list):
    """
    Get node features from data pandas file (for graph); without bin_capacity
    node feature: [node type(one-hot), UE requirement, northing, easting, bandwith (when connects to other nodes)]
    """
    # one-hot representation for node types
    data = np.asarray([['T'], ['M'], ['O']])
    # define one hot encoding
    encoder = OneHotEncoder(sparse=False)
    # transform data
    onehot = encoder.fit_transform(data)

    # get items_list (UE requirements)
    k = 2e-3
    items_list = np.array(data_pd['Aggregate Residential Customers']) * k
    # get coordinates
    northing_list = list(data_pd['northing'])
    easting_list = list(data_pd['easting'])

    node_fea = []
    for i in range(len(data_pd)):
        if i < n_nodes[0]:
            node_type_onehot = list(onehot[0])
            bandwidth = bandwidth_list[0]
        elif i < n_nodes[0] + n_nodes[1]:
            node_type_onehot = list(onehot[1])
            bandwidth = bandwidth_list[1]
        else:
            node_type_onehot = list(onehot[2])
            bandwidth = bandwidth_list[2]
        cur = node_type_onehot + [items_list[i]] + [northing_list[i]] + [easting_list[i]] + [bandwidth]
        node_fea.append(cur)
    return node_fea

def node_fea_with_capacity(node_fea, bin_capacity):
    """
    node feature: [node type(one-hot), UE requirement, northing, easting, bandwith (when connects to other nodes), remaining capacity]
    """
    node_fea_ = []
    assert len(node_fea) == len(bin_capacity)
    for i in range(len(node_fea)):
        node_fea_.append(np.concatenate((node_fea[i], bin_capacity[i])))
    return node_fea_

def node_fea_normalization(node_fea, bandwidth, C):
    """
    Normalize node features as x_input to graphs, for doing GNN.
    This normalization is specified to this problem - simple scaling normalization.
    """
    nodes_fea = np.array(node_fea.copy())
    # items * 1e-2
    nodes_fea[:, 3] = nodes_fea[:, 3] * 1e-2 
    # northing & easting * 1e-6
    nodes_fea[:, 4] = nodes_fea[:, 4] * 1e-6
    nodes_fea[:, 5] = nodes_fea[:, 5] * 1e-6
    # bandwith / max(bandwidth)
    nodes_fea[:, 6] = nodes_fea[:, 6] / max(bandwidth)
    # capacity / max(C)
    nodes_fea[:, 7] = nodes_fea[:, 7] / max(C)
    return nodes_fea