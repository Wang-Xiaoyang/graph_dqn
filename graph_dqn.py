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

from graph_dqn_utilities import *

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, g, inputs):
        h = self.conv1(dgl.add_self_loop(g), inputs)
        h = torch.relu(h)
        h = self.linear(h)
        return h

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

def choose_action(env, epsilon, g):
    # choose action using epsilon greedy given the current ob
    # g is the graph
    if np.random.uniform(0,1) >= epsilon:
        nodes_RU, nodes_Du = state_to_node_fea(env)
        with torch.no_grad():
            features = torch.tensor(np.concatenate([nodes_RU, nodes_DU]), dtype=torch.double)
            g.ndata['x'] = features
            Q_values = []
            for i in range(env.n_bins):
                g.add_edges([env.cur_item], i+env.n_items)
                inputs = g.ndata['x']
                value = net(g, inputs)
                Q_values.append(sum(value))
                g = dgl.remove_edges(g, torch.tensor([g.num_edges()-1])) # remove the newly added edge
        a = np.argmax(Q_values)
    else:
        a = env.action_space.sample()
    return a

# make env
env = gym.make("gym_graph_bin_packing:graph_vbpp-v0")
env.reset()

nodes_RU, nodes_DU = state_to_node_fea(env)

# build graph
g = dgl.DGLGraph()
g.add_nodes(env.n_items) # add RUs
g.add_nodes(env.n_bins) # add DUs
nx_g = g.to_networkx()

# nx.draw(nx_g, with_labels=True, node_color=[[.7, .7, .7]])
# plt.show()
features = torch.tensor(np.concatenate([nodes_RU, nodes_DU]), dtype=torch.double)
g.ndata['x'] = features

# graph neural net
net = GCN(3, 5).double()
inputs = g.ndata['x']
value = net(g, inputs)

print(sum(value))

a = choose_action(env, 0.002, g)
print(a)