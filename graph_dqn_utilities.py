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
    # get node features from env states; normalize
    # 1. one-hot representation for node types
    data = np.asarray([['type 0'], ['type 1']])
    encoder = OneHotEncoder(sparse=False)
    onehot = encoder.fit_transform(data)
    nodes_type_0_t = onehot[0] # type
    nodes_type_0_t = np.reshape(np.tile(nodes_type_0_t, env.n_nodes[0]), (env.n_nodes[0], -1)) # n_nodes_type_0 * feature_len
    nodes_type_1_t = onehot[1]
    nodes_type_1_t = np.reshape(np.tile(nodes_type_1_t, env.n_nodes[1]), (env.n_nodes[1], -1))
    nodes_t = np.concatenate((nodes_type_0_t, nodes_type_1_t))

    # 2. type 0 values; type 1: remaining capacity
    nodes_type_0_v = np.transpose(env.fea.copy())
    nodes_type_1_v = np.transpose(env.remaining_capacity)
    nodes_v = np.concatenate((nodes_type_0_v, nodes_type_1_v))
    nodes_v = np.expand_dims(nodes_v, axis=1)
    nodes_v = nodes_v / env.C

    # concatenate
    nodes_fea = np.concatenate((nodes_t, nodes_v), axis=1)

    # for BBU pooling
    # UEs per site
    num_UEs = np.concatenate((env.UE_per_site, np.zeros(env.n_nodes[1])), axis=0)
    num_UEs = np.expand_dims(num_UEs, axis=1)
    num_UEs = num_UEs / 1e2
    nodes_fea = np.concatenate((nodes_fea, num_UEs), axis=1)

    # 3. current node (in terms of MDP action) is 1, others are 0
    current_node = np.zeros((env.n_nodes_total, 1))
    current_node[env.cur_item] = 1
    nodes_fea = np.concatenate((nodes_fea, current_node), axis=1)

    # 4. location
    nodes_x, nodes_y = env.nodes_x / 10, env.nodes_y / 10
    nodes_fea = np.concatenate((nodes_fea, nodes_x, nodes_y), axis=1)

    return nodes_fea

def adjacency_vbpp_to_graph(adjacency_mat):
    # this function transfers the adjacency matrix of the node association env, 
    # which has the size [n_type_0_nodes, n_type_1_nodes], to the format of pair of nodes [u, v]
    # for defining a graph
    # u: source nodes; v: destination nodes
    n_type_0_nodes = adjacency_mat.shape[0]
    a = np.sum(adjacency_mat, axis=1)
    u = np.where(a == 1)[0]
    v = np.zeros(len(u), dtype=int)
    idx = 0
    for i in u:
        v[idx] = np.where(adjacency_mat[i, :] == 1)[0][0] + n_type_0_nodes
        idx += 1
    return u, v
