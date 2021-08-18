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
# Evaluate the trained model
# Return 'nodes.csv' and 'links.csv'
# nodes.csv, columns are [id, lat/long, northing/easting, user distribution, resource utlisation(%), power consumption]
# links.csv, columns are [[id1, id2], latency]
# total power consumption and total latency can be calculated
# trained model --> model validation --> action sequences --> build two output files

from graph_dqn_utilities_network import *

import wandb
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#device = "cpu"

# wandb log
wandb.init(entity="xiaoyang",
            project="graph-q-learning-network-trained")

"""
Import dataset to decide env configurations.
Dataset 1: bristol_nodes_1.csv 
"""
df = pd.read_csv('bristol_nodes_1.csv')
df.drop(['Unnamed: 0'], axis=1) #remove unused information

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

"""
Set parameters; network structure;
Define choose_action function and model_validate function, return variables for nodes and links.
"""
#env config
wandb.config.n_nodes = n_nodes
# nodes capacity 
wandb.config.C = [100, 500, 1000]
wandb.config.power_par = [0.1, 0.5, 1.0]
wandb.config.power_scaling = 1000 # from percentage to real power consumption factor;
                                # power consumption percentage is for each node; power_par controls the different 'baseline' consumption of different 
                                # types of nodes, e.g., tier 1 nodes consumes 10% the power of tier 3 nodes.
                                # scaling is when we get the regularized comsumption percentage, convert them to real power consumption
wandb.config.model = 'GCN'
# network structure
wandb.config.input_size = 8
wandb.config.hidden_size = 16

wandb.config.validation_eps = 1
config = wandb.config

assert len(config.n_nodes) == len(config.C)

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
    """
    Validate model.
    Return ep_reward; ob_ (links) and nodes_fea(nodes features)
    """
    ep_reward = 0
    ob = env.reset(config.n_nodes, config.C, config.power_par, items_list, northing_list, easting_list)

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
    return ep_reward, ob_, nodes_fea

"""
Initialize network and do the validation.
"""

# initialize Q network
Q_net = GQN(config.input_size, config.hidden_size).double().to(device=device)

# make env
env = gym.make("gym_graph_bin_packing_network:graph_vbpp_network-v0")
env.reset(config.n_nodes, config.C, config.power_par, items_list, northing_list, easting_list)
n_actions = env.action_space.n

# node features for graph
nodes_fea_no_capacity = data_to_node_features_no_capacity(df, n_nodes, bandwidth)


print("++++++++ Model Evaluation +++++++++++++++")
path = './happy_water_64.pt'
Q_net.load_state_dict(torch.load(path))

for ii in range(config.validation_eps):
    ep_score, ob_, nodes_fea = model_validate()
    wandb.log({"evaluation ep score": ep_score})


"""
Get output files.
"""
# nodes.csv, columns are [id, lat/long, northing/easting, user distribution, resource utlisation(%), power consumption]

def resource_utilisation(config, nodes_fea):
    # resource utilisation in %
    # power consumption (percentage * a power consumption scaling factor)
    capacity_list, power_scale = [], []
    for i in range(len(config.n_nodes)):
        capacity_list += [config.C[i]] * config.n_nodes[i]
        power_scale += [config.power_par[i]] * config.n_nodes[i]

    remaining_capcity = np.array(nodes_fea)[:, -1]
    remaining_capcity = list(remaining_capcity)

    remaining_ratio = [i / j for i, j in zip(remaining_capcity, capacity_list)]
    used_ratio = np.array([1 - i for i in remaining_ratio])
    # power consumption: u^2 + 2*u
    power = [(- i**2 + i * 2) * j for i, j in zip(used_ratio, power_scale)]
    return used_ratio, np.array(power) * config.power_scaling

nodes_output = df.copy()
nodes_output.drop([' cpu_capacity', ' memory_capacity', ' storage_capacity'], axis=1)

used_ratio, power_consumption = resource_utilisation(config, nodes_fea)
