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
import copy

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
            hg = dgl.sum_nodes(g, 'h')
            return self.linear(hg)

class GQN_2(nn.Module):
    def __init__(self, in_feats, hidden_size):
        # more layers in this class
        super(GQN_2, self).__init__()
        # apply activation function or not?
        self.conv1 = GraphConv(in_feats, hidden_size, allow_zero_in_degree=False)
        # self.conv2 = GraphConv(hidden_size, hidden_size, allow_zero_in_degree=False)
        self.linear1 = nn.Linear(hidden_size, 8)
        self.linear2 = nn.Linear(8, 1)

    def forward(self, g, inputs):
        # This is the network for the BBU pooling env.
        # env_time is an int between [0, 23] (24 hours; normalised), indicate the current time in the env
        ## env_time is an auxiliary input to the linear layer 
        # Apply graph convolution and activation.
        h = F.relu(self.conv1(g, inputs))
        # h = F.relu(self.conv2(g, h))

        with g.local_scope():
            g.ndata['h'] = h
            # Calculate graph representation by sum readout.
            # note: this is doing graph polling - check the normalization; check data dimension
            # better pooling options? currently, f(sum h_v (nodes)) = Q(s,a)
            hg = dgl.sum_nodes(g, 'h')
            h = F.relu(self.linear1(hg))
            return self.linear2(h)

class GQN_3(nn.Module):
    def __init__(self, in_feats, hidden_size):
        # more layers in this class
        super(GQN_3, self).__init__()
        # apply activation function or not?
        self.conv1 = GraphConv(in_feats, hidden_size, allow_zero_in_degree=False)
        # self.conv2 = GraphConv(hidden_size, hidden_size, allow_zero_in_degree=False)
        self.linear1 = nn.Linear(hidden_size, 8)
        self.linear2 = nn.Linear(8, 4)
        self.linear3 = nn.Linear(4, 1)

    def forward(self, g, inputs):
        # This is the network for the BBU pooling env.
        # env_time is an int between [0, 23] (24 hours; normalised), indicate the current time in the env
        ## env_time is an auxiliary input to the linear layer 
        # Apply graph convolution and activation.
        h = F.relu(self.conv1(g, inputs))
        # h = F.relu(self.conv2(g, h))

        with g.local_scope():
            g.ndata['h'] = h
            # Calculate graph representation by sum readout.
            # note: this is doing graph polling - check the normalization; check data dimension
            # better pooling options? currently, f(sum h_v (nodes)) = Q(s,a)
            hg = dgl.sum_nodes(g, 'h')
            h = F.relu(self.linear1(hg))
            h = F.relu(self.linear2(h))
            return self.linear3(h)


# class GQN_with_time(nn.Module):
#     def __init__(self, in_feats, hidden_size):
#         super(GQN_with_time, self).__init__()
#         self.conv1 = GraphConv(in_feats, hidden_size, allow_zero_in_degree=False)
#         self.conv2 = GraphConv(hidden_size, hidden_size, allow_zero_in_degree=False)
#         self.linear1 = nn.Linear(hidden_size+1, 8)
#         self.linear2 = nn.Linear(8, 1)

#     def forward(self, g, inputs, env_time):
#         # This is the network for the BBU pooling env.
#         # env_time is an int between [0, 23] (24 hours; normalised), indicate the current time in the env
#         ## env_time is an auxiliary input to the linear layer 
#         # Apply graph convolution and activation.
#         h = F.relu(self.conv1(g, inputs))
#         h = F.relu(self.conv2(g, h))
#         with g.local_scope():
#             g.ndata['h'] = h
#             # Calculate graph representation by sum readout.
#             # note: this is doing graph polling - check the normalization; check data dimension
#             # better pooling options? currently, f(sum h_v (nodes)) = Q(s,a)
#             hg = dgl.sum_nodes(g, 'h')
#             hg_ = torch.cat((hg, env_time), dim=1)
#             h = F.relu(self.linear1(hg_))
#             return self.linear2(h)

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
    # nodes_type_1_v = np.transpose(env.remaining_capacity)
    nodes_type_1_v = np.zeros(env.n_nodes[1]) * env.C
    nodes_v = np.concatenate((nodes_type_0_v, nodes_type_1_v))
    nodes_v = np.expand_dims(nodes_v, axis=1)
    nodes_v = nodes_v / env.C

    # for BBU pooling
    # UEs per site
    num_UEs = np.concatenate((env.UE_per_site, np.zeros(env.n_nodes[1])), axis=0)
    num_UEs = np.expand_dims(num_UEs, axis=1)
    num_UEs = num_UEs / 1e2

    # 3. current node (in terms of MDP action) is 1, others are 0
    current_node = np.zeros((env.n_nodes_total, 1))
    current_node[env.cur_item] = 1

    # 4. location
    # location_scale = 10 # for standard node_association env
    location_scale = 1.0 # for bbu pooling env
    nodes_x, nodes_y = env.nodes_x / location_scale, env.nodes_y / location_scale

    # 5. degree (number of connections)
    degrees = np.concatenate([np.sum(env.adjacency_mat, axis=1), np.sum(env.adjacency_mat, axis=0)])
    degrees = np.expand_dims(degrees, axis=1) / 10 # scaling

    # # 5. env_time: the time stamp, in 24 hr format
    # # env_time = env.t_norm * np.ones((env.n_nodes_total, 1))
    # # try one-hot representation
    # data_time = np.asarray(range(24)).reshape(-1, 1)
    # encoder = OneHotEncoder(sparse=False)
    # onehot = encoder.fit_transform(data_time)    

    # concatination
    # nodes_fea = np.concatenate((nodes_t, nodes_v, num_UEs, current_node, nodes_x, nodes_y, env_time), axis=1)
    nodes_fea = np.concatenate((nodes_t, nodes_v, num_UEs, current_node, nodes_x, nodes_y), axis=1)
    # nodes_fea = np.concatenate((nodes_t, nodes_v, num_UEs, current_node, nodes_x, nodes_y, degrees), axis=1) # 8 features, with degrees
    return nodes_fea

def adjacency_vbpp_to_graph(adjacency_mat):
    # this function transfers the adjacency matrix of the node association env, 
    # which has the size [n_type_0_nodes, n_type_1_nodes], to the format of pair of nodes [u, v]
    # for defining a graph
    # u: source nodes; v: destination nodes
    n_type_0_nodes = adjacency_mat.shape[0]
    n_type_1_nodes = adjacency_mat.shape[1]
    a = np.sum(adjacency_mat, axis=1)
    u = np.where(a == 1)[0]
    v = np.zeros(len(u), dtype=int)
    idx = 0
    for i in u:
        v[idx] = np.where(adjacency_mat[i, :] == 1)[0][0] + n_type_0_nodes
        idx += 1
    # type_1_nodes: fully connected to each other (avoid self-loop; it will be added later)
    for j in range(n_type_1_nodes):
        for k in range(j+1, n_type_1_nodes):
            u = np.append(u, [j+n_type_0_nodes])
            v = np.append(v, [k+n_type_0_nodes])

    # # type_0_nodes: fully connected to each other
    ## ! not useful for r_a; might be useful in the future
    # for j in range(n_type_0_nodes):
    #     for k in range(j+1, n_type_0_nodes):
    #         u = np.append(u, [j])
    #         v = np.append(v, [k])
    return u, v

def self_loop_graph(g):
    list_nodes = list(range(g.num_nodes()))
    g = dgl.add_edges(g, list_nodes, list_nodes)
    return g

def percentile_ranked_reward(reward_history, current_r, percentile):
    p = np.percentile(reward_history, percentile)
    if current_r >= p:
        r = 1.0
    else:
        r = 0.0
    return r