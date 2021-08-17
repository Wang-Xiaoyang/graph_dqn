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
import pandas as pd

import gc

from graph_dqn_utilities_network import *

import wandb
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#device = "cpu"

# !!!! to check:
# the direction of graph (directed and undirected, what is difference in results)

# wandb log
# wandb.init(entity="xiaoyang",
#             project="graph-q-learning")
wandb.init(entity="xiaoyang",
            project="graph-q-learning-network-trained")

"""
Import dataset to decide env configurations.
Dataset 1: bristol_nodes_1.csv 
"""
df = pd.read_csv('bristol_nodes_1.csv')
# get n_nodes
nodes_types = ['T', 'M', 'O']
n_nodes = []
for item in nodes_types:
    list_nodes = get_node_id_types(item, df)
    n_nodes.append(len(list_nodes))
# bandwidth from this type of nodes to others
bandwidth = [10, 100, 0]

# get items_list
k = 2e-3
items_list = np.array(df['Aggregate Residential Customers']) * k
# get coordinates
northing_list = list(df['northing'])
easting_list = list(df['easting'])

#env config
wandb.config.n_nodes = n_nodes
# nodes capacity 
wandb.config.C = [100, 500, 1000]
wandb.config.model = 'GCN'

wandb.config.N_EPS = 500
wandb.config.BUFFER_LEN = 2000
wandb.config.batch_size = 64
wandb.config.GAMMA = 0.98
wandb.config.LEARNING_RATE = 0.005
wandb.config.UPDATE_STEP = 10
wandb.config.epsilon_max = 1.0
wandb.config.epsilon_min = 0.0005
wandb.config.validation_eps = 1
config = wandb.config

def choose_action(env, epsilon, g):
    # choose action using epsilon greedy given the current ob
    # g is the graph
    g = self_loop_graph(g, n_nodes[2])
    if np.random.uniform(0,1) >= epsilon:
        bin_capacity_list = state_to_bin_capacity(env)
        nodes_fea = node_fea_with_capacity(nodes_fea_no_capacity, bin_capacity_list)
        with torch.no_grad():
            features = torch.tensor(node_fea_normalization(nodes_fea, bandwidth, config.C), dtype=torch.double)
            g.ndata['x'] = features
            Q_values = []
            with g.local_scope():
                for i in range(env.action_space.n):
                    g.add_edges([env.cur_item], i+env.n_nodes[0])
                    g = g.to(device)
                    # make sure inputs and g are on the same device
                    inputs = g.ndata['x']
                    value = Q_net(g, inputs).detach()
                    Q_values.append(sum(value))
                    g = g.to('cpu')
                    g = dgl.remove_edges(g, torch.tensor([g.num_edges()-1])) # remove the newly added edge
        a = np.argmax(Q_values)
    else:
        a = env.action_space.sample()
    return a

def model_validate():
    ep_reward = 0
    ob = env.reset(config.n_nodes, config.C, items_list, northing_list, easting_list)

    # observation to graph
    u, v = adjacency_vbpp_to_graph(ob)
    g = dgl.graph((u, v), num_nodes=sum(n_nodes))
    g = self_loop_graph(g, n_nodes[2])
    # node features
    bin_capacity_list = state_to_bin_capacity(env)
    nodes_fea = node_fea_with_capacity(nodes_fea_no_capacity, bin_capacity_list)
    g.ndata['x'] = torch.tensor(node_fea_normalization(nodes_fea, bandwidth, config.C), dtype=torch.double)

    done = False
    while not done:
        # validate model for current Q network
        a = choose_action(env, 0, g)
        print(a)
        ob_, r, done, _ = env.step(a)
        ep_reward += r
        # new ob to new state
        u, v = adjacency_vbpp_to_graph(ob_)
        g = dgl.graph((u, v), num_nodes=sum(n_nodes))
        g = self_loop_graph(g, n_nodes[2])
        # node features
        bin_capacity_list = state_to_bin_capacity(env)
        nodes_fea = node_fea_with_capacity(nodes_fea_no_capacity, bin_capacity_list)
        g.ndata['x'] = torch.tensor(node_fea_normalization(nodes_fea, bandwidth, config.C), dtype=torch.double)
        state_ = g
    return ep_reward

# parameters
N_EPS = config.N_EPS
BUFFER_LEN = config.BUFFER_LEN
batch_size = config.batch_size
GAMMA = config.GAMMA


# initialize two graph neural nets: Q_net, Q_net_
Q_net = GQN(8, 16).double().to(device=device)


# make env
env = gym.make("gym_graph_bin_packing_network:graph_vbpp_network-v0")
env.reset(config.n_nodes, config.C, items_list, northing_list, easting_list)
n_actions = env.action_space.n

# node features for graph
nodes_fea_no_capacity = data_to_node_features_no_capacity(df, n_nodes, bandwidth)
# bin_capacity_list = state_to_bin_capacity(env)
# nodes_fea = node_fea_with_capacity(nodes_fea, bin_capacity_list)

step = 0
ep_rewards = []


print("++++++++ Model Evaluation +++++++++++++++")
path = './happy_water_64.pt'
Q_net.load_state_dict(torch.load(path))

for ii in range(config.validation_eps):
    ep_score = model_validate()
    wandb.log({"evaluation ep score": ep_score})