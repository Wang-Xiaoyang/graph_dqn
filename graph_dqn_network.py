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
            project="graph-q-learning-network")

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
wandb.config.validation_eps = 20
config = wandb.config

def choose_action(env, epsilon, g):
    # choose action using epsilon greedy given the current ob
    # g is the graph
    g = self_loop_graph(g)
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
    g = self_loop_graph(g)
    # node features
    bin_capacity_list = state_to_bin_capacity(env)
    nodes_fea = node_fea_with_capacity(nodes_fea_no_capacity, bin_capacity_list)
    g.ndata['x'] = torch.tensor(node_fea_normalization(nodes_fea, bandwidth, config.C), dtype=torch.double)

    done = False
    while not done:
        # validate model for current Q network
        a = choose_action(env, 0, g)
        ob_, r, done, _ = env.step(a)
        ep_reward += r
        # new ob to new state
        u, v = adjacency_vbpp_to_graph(ob_)
        g = dgl.graph((u, v), num_nodes=sum(n_nodes))
        g = self_loop_graph(g)
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
LEARNING_RATE = config.LEARNING_RATE
UPDATE_STEP = config.UPDATE_STEP
epsilon_max = config.epsilon_max
epsilon_min = config.epsilon_min
epsilon_step = (epsilon_max - epsilon_min) / N_EPS # how to decay epsilon

validation_eps = config.validation_eps

# initialize memory buffer
memory = Memory(max_size=BUFFER_LEN)

# initialize two graph neural nets: Q_net, Q_net_
Q_net = GQN(8, 16).double().to(device=device)
Q_net_ = GQN(8, 16).double().to(device=device)
# copy parameters
Q_net_.load_state_dict(Q_net.state_dict())
# define optimizer
optimizer = torch.optim.Adam(Q_net.parameters(), lr=LEARNING_RATE)
# define loss
loss_MSE = nn.MSELoss().to(device)

# make env
env = gym.make("gym_graph_bin_packing_network:graph_vbpp_network-v0")
env.reset(config.n_nodes, config.C, items_list, northing_list, easting_list)
n_actions = env.action_space.n

epsilon = epsilon_max

# node features for graph
nodes_fea_no_capacity = data_to_node_features_no_capacity(df, n_nodes, bandwidth)
# bin_capacity_list = state_to_bin_capacity(env)
# nodes_fea = node_fea_with_capacity(nodes_fea, bin_capacity_list)

wandb.watch(Q_net)

step = 0
ep_rewards = []
for ii in range(N_EPS):
    ep_reward = 0
    # reset and get the init state
    ob = env.reset(config.n_nodes, config.C, items_list, northing_list, easting_list)
    u, v = adjacency_vbpp_to_graph(ob)
    g = dgl.graph((u, v), num_nodes=sum(n_nodes))
    g = self_loop_graph(g)
    # node features
    bin_capacity_list = state_to_bin_capacity(env)
    nodes_fea = node_fea_with_capacity(nodes_fea_no_capacity, bin_capacity_list)
    g.ndata['x'] = torch.tensor(node_fea_normalization(nodes_fea, bandwidth, config.C), dtype=torch.double)

    done = False
    while not done:        
        a = choose_action(env, epsilon, g)
        a_graph = [env.cur_item, a+env.n_nodes[0]] # action in graph format: u-->v, forms an edge
        g.add_edges([a_graph[0]], a_graph[1])

        state_action = g
        del g
        gc.collect()
        
        ob_, r, done, _ = env.step(a)
        ep_reward += r
        # new ob to new state
        u, v = adjacency_vbpp_to_graph(ob)
        g = dgl.graph((u, v), num_nodes=sum(n_nodes))
        g = self_loop_graph(g)
        # node features
        bin_capacity_list = state_to_bin_capacity(env)
        nodes_fea = node_fea_with_capacity(nodes_fea_no_capacity, bin_capacity_list)
        g.ndata['x'] = torch.tensor(node_fea_normalization(nodes_fea, bandwidth, config.C), dtype=torch.double)
        state_ = g        
        # next state / action pairs (for all possible actions)
        state_action_ = []
        for t in range(env.action_space.n):
            with g.local_scope():
                g_tmp = copy_graph(g, 'x')
                g_tmp.add_edges([env.cur_item], t+n_nodes[0])
                state_action_.append(g_tmp)
            del g_tmp
            gc.collect()

        # add to memory
        memory.add((state_action, a_graph, r, state_action_, done))
        state = state_

        if len(memory.buffer) >= batch_size:
            batch = memory.sample(batch_size)
            # get state, action, reward and the next state
            states_actions = dgl.batch([item[0] for item in batch]).to(device)
            actions = torch.DoubleTensor([item[1] for item in batch]).to(device)
            rewards = torch.DoubleTensor([item[2] for item in batch]).to(device)

            states_actions_batch = []
            for item in batch:
                for k in range(env.action_space.n):
                    states_actions_batch.append(item[3][k])
            states_actions_ = dgl.batch(states_actions_batch).to(device) # the length of this batch is longer than others, for multiple actions

            done_info = [item[4] for item in batch]

            # if next state is the terminal state
            non_final_mask = torch.tensor(list(map(lambda s: s is not True, done_info)), dtype=torch.float32).to(device) #
            # next_q_values = max_a Q_target(s',a)
            next_q_values_all = Q_net_(states_actions_, states_actions_.ndata['x']).detach()
            next_q_values = torch.max(next_q_values_all.reshape(batch_size, n_actions), axis=1).values
            next_q_values = next_q_values.to(device)
            q_targets = rewards.squeeze() + GAMMA * torch.mul(next_q_values, non_final_mask).to(device) 
            q_targets = torch.unsqueeze(q_targets, 1).to(device)

            # q_values for current Q(s,a)
            in_fea = states_actions.ndata['x'].to(device)
            q_values = Q_net(states_actions, in_fea)

            # one step of back prop
            optimizer.zero_grad() # clear gradient
            loss = loss_MSE(q_targets, q_values)
            loss.backward()
            optimizer.step()
            
            if step % UPDATE_STEP == 0:
                Q_net_.load_state_dict(Q_net.state_dict())
            step += 1
    
    epsilon = max((epsilon-epsilon_step), epsilon_min)
    # ep_rewards.append(ep_reward)
    # log results after each game episode
    wandb.log({"ep score": ep_reward, "env step": step})

print("++++++++ Model Evaluation +++++++++++++++")
for ii in range(validation_eps):
    ep_score = model_validate()
    wandb.log({"evaluation ep score": ep_score})


timestr = time.strftime("%Y%m%d_%H%M%S")
name = './' + timestr + '_Q_net.pt'
# save Q_net
torch.save(Q_net.state_dict(), name)