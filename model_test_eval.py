"""
# Evaluate the trained model
# Return 'nodes.csv' and 'links.csv'
# nodes.csv, columns are [id, node_type (0, 1 or 2), lat/long, northing/easting, user distribution, resource utlisation(%), power consumption]
# links.csv, columns are [[id1, id2], latency]
# total power consumption and total latency can be calculated
# trained model --> model validation --> action sequences --> build two output files
"""

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
import math

import gc

from graph_dqn_utilities_network import *

import wandb
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# wandb log
wandb.init(entity="xiaoyang",
            project="graph-q-learning-network-trained")

"""
Import dataset to decide env configurations.
Dataset 1: bristol_nodes_1.csv 
"""
folder = './data_1/'
file_name = 'bristol_nodes_1.csv'
df = pd.read_csv(folder + file_name)
df.drop(['Unnamed: 0'], axis=1) #remove unused information
nodes_output = df.copy()

# get n_nodes
nodes_types = ['T', 'M', 'O']
n_nodes = []
for item in nodes_types:
    list_nodes = get_node_id_types(item, df)
    n_nodes.append(len(list_nodes))

# get node type indicator: [0, 1, 2] for ['T', 'M', 'O']
node_type_numeric = list(range(len(nodes_types)))
nodes_types_numeric = np.zeros(len(df))
for i in range(len(df)):
    cur_type = df['# id'][i][0]
    nodes_types_numeric[i] = node_type_numeric[nodes_types.index(cur_type)]

nodes_output['node type'] = nodes_types_numeric

# bandwidth from this type of nodes to others - a pre-defined constant
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
Initialize network; perform model validation.
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
nodes.csv, columns are [id, node_type (0, 1 or 2), lat/long, northing/easting, user distribution, resource utlisation(%), power consumption]
"""
# calculate resource and power utilisation

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
    # power = [(- i**2 + i * 2) * j for i, j in zip(used_ratio, power_scale)]
    power_sites_percentage = [- i**2 + i * 2 for i in used_ratio]
    # return used_ratio, np.array(power) * config.power_scaling
    return np.array(used_ratio) * 100, np.array(power_sites_percentage) * 100

nodes_output = nodes_output.drop(['Unnamed: 0',' cpu_capacity', ' memory_capacity', ' storage_capacity'], axis=1)

used_ratio, power_consumption = resource_utilisation(config, nodes_fea)
nodes_output['Resource utilisation (%)'] = used_ratio
nodes_output['Power consumption (%)'] = power_consumption
# save file
nodes_output.to_csv(folder + 'results_nodes.csv')

# links.csv - from ob_ (adjacency matrix, n_rows = n_tier_1_nodes + n_tier_2_nodes
#                                           n_columns = n_tier_2_nodes + n_tier_3_nodes)

def calculate_latency(easting, northing, a):
    c_light = 299792458e-3
    # a is a tuple: (node1, node2)
    x_1, y_1 = easting[a[0]], northing[a[0]]
    x_2, y_2 = easting[a[1]], northing[a[1]]
    dist = math.hypot(x_2 - x_1, y_2 - y_1)
    # latency
    l = dist / c_light * 1.7# latency in Millisecond (ms)
    return l 

links = []
latency_list = []
easting = df['easting']
northing = df['northing']
for i in range(len(ob_)):
    a = list(ob_[i]).index(1)
    link = (df['# id'][i], df['# id'][a + config.n_nodes[0]])
    links.append(link)
    latency = calculate_latency(easting, northing, (i, a+config.n_nodes[0]))
    latency_list.append(latency)

# save file
df_links = pd.DataFrame()
df_links['links'] = links
df_links['latency(ms)'] = latency_list
df_links.to_csv(folder + 'results_links.csv')

# save nodes in a graph format
# source node, source node type, target node, target node type, edge weight(latency),
# destinatio node power consumption, destination node resource utilisation
link_s, link_d = [], []
link_s_type, link_d_type = [], []

for i in range(len(ob_)):
    a = list(ob_[i]).index(1)
    link_source = nodes_output['# id'][i]
    link_source_type = nodes_output['node type'][list(nodes_output['# id']).index(link_source)]
    link_destination = nodes_output['# id'][a + config.n_nodes[0]]
    link_destination_type = nodes_output['node type'][list(nodes_output['# id']).index(link_destination)]
    link_s.append(link_source)
    link_s_type.append(link_source_type)
    link_d.append(link_destination)
    link_d_type.append(link_destination_type)

# get power consumption and resource utilisation for destination nodes
pow_con_list, resource_util_list = [], []
for i in link_d:
    idx = list(nodes_output['# id']).index(i)
    pow_con = nodes_output['Power consumption (%)'][idx]
    resource_util = nodes_output['Resource utilisation (%)'][idx]
    pow_con_list.append(pow_con)
    resource_util_list.append(resource_util)

# write file
df_links_2 = pd.DataFrame()
df_links_2['source nodes'] = link_s
df_links_2['destination nodes'] = link_d
df_links_2['source nodes type'] = link_s_type
df_links_2['destination nodes type'] = link_d_type
df_links_2['latency(ms)'] = latency_list
df_links_2['destination node power consumption'] = pow_con_list
df_links_2['destination node resource utilisation'] = resource_util_list

df_links_2.to_csv(folder + 'results_links_2.csv')