import numpy as np
import gym
import torch
from collections import deque
import matplotlib.pyplot as plt

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
    src = g.edges()[0]
    dst = g.edges()[1]
    num_nodes = g.num_nodes()
    g_copy = dgl.graph((src, dst), num_nodes=num_nodes)
    fea = torch.clone(g.ndata[fea_name])
    g_copy.ndata[fea_name] = fea
    return g_copy

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

# env state to graph features
def state_to_node_fea(env):

    # node feature for RU: [0(indicates this is an RU), item_value, 1 or 0(connection status)]
    nodes_RU_value = np.transpose(env.items.copy())

    nodes_RU_adjacency = np.sum(env.adjacency_mat, axis=1)
    nodes_RU_adjacency = np.expand_dims(nodes_RU_adjacency, axis=1)

    nodes_RU_flag = np.zeros([env.n_items, 1])
    # nodes_RU_current = np.zeros([env.n_items, 1])
    # nodes_RU_current[env.cur_item] = 1

    # nodes_RU = np.concatenate([nodes_RU_flag, nodes_RU_value, nodes_RU_adjacency, nodes_RU_current], axis=1)
    nodes_RU = np.concatenate([nodes_RU_flag, nodes_RU_value, nodes_RU_adjacency], axis=1)

    # node feature for DU: [1(indicates this is a DU), remaining capacity, load(connected RUs)]
    nodes_DU_flag = np.ones([env.n_bins, 1])
    nodes_DU_value = np.zeros([env.n_bins, env.vector_dim])


    for j in range(env.n_bins):
        for i in range(env.vector_dim):
            items_total = np.dot(env.items[i], env.adjacency_mat[:,j])
            nodes_DU_value[j, i] = env.bin_capacity[i] - items_total

    nodes_DU_load = np.sum(env.adjacency_mat, axis=0)
    nodes_DU_load = np.expand_dims(nodes_DU_load, axis=1)
    nodes_DU = np.concatenate([nodes_DU_flag, nodes_DU_value, nodes_DU_load], axis=1)

    return nodes_RU, nodes_DU

def adjacency_vbpp_to_graph(adjacency_mat):
    # this function transfers the adjacency matrix of the vector bin packing env, 
    # which has the size [n_items, n_bins], to the format of pair of nodes [u, v]
    # for defining a graph
    n_items = adjacency_mat.shape[0]
    a = np.sum(adjacency_mat, axis=1)
    u = np.where(a == 1)[0]
    v = np.zeros(len(u), dtype=int)
    idx = 0
    for i in u:
        v[idx] = np.where(adjacency_mat[i, :] == 1)[0][0] + n_items
    return u, v