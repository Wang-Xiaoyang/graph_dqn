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

import wandb

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = "cpu"

# !!!! to check:
# the direction of graph (directed and undirected, what is difference in results)

# wandb log
wandb.init(entity="xiaoyang",
            project="graph-q-learning")

#env config
wandb.config.N_RUs = 20
wandb.config.N_DUs = 4

wandb.config.N_EPS = 2000
wandb.config.BUFFER_LEN = 10000
wandb.config.BATCH_SIZE = 500
wandb.config.GAMMA = 0.98
wandb.config.LEARNING_RATE = 0.005
wandb.config.UPDATE_STEP = 20
wandb.config.epsilon_max = 1.0
wandb.config.epsilon_min = 0.0005
wandb.config.validation_eps = 20
config = wandb.config

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
                g = g.to(device)
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
    ob = env.reset()

    # observation to graph
    u, v = adjacency_vbpp_to_graph(ob)
    nodes_RU, nodes_DU = state_to_node_fea(env)
    g = dgl.graph((u, v), num_nodes=n_nodes)
    g.ndata['x'] = torch.tensor(np.concatenate([nodes_RU, nodes_DU]), dtype=torch.double)

    done = False
    while not done:
        # validate model for current Q network
        a = choose_action(env, 0, g)
        ob_, r, done, _ = env.step(a)
        ep_reward += r
        # new ob to new state
        u, v = adjacency_vbpp_to_graph(ob_)
        nodes_RU, nodes_DU = state_to_node_fea(env)
        g = dgl.graph((u, v), num_nodes=n_nodes)
        g.ndata['x'] = torch.tensor(np.concatenate([nodes_RU, nodes_DU]), dtype=torch.double)
        state_ = g
    return ep_reward

# parameters
N_EPS = config.N_EPS
BUFFER_LEN = config.BUFFER_LEN
BATCH_SIZE = config.BATCH_SIZE
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
Q_net = GQN(3, 8).double().to(device=device)
Q_net_ = GQN(3, 8).double().to(device=device)
# copy parameters
Q_net_.load_state_dict(Q_net.state_dict())
# define optimizer
optimizer = torch.optim.Adam(Q_net.parameters(), lr=LEARNING_RATE)
# define loss
loss_MSE = nn.MSELoss().to(device)

# make env
env = gym.make("gym_graph_bin_packing:graph_vbpp-v0")
env.reset()
n_RUs, n_DUs = env.n_items, env.n_bins
n_nodes = n_RUs + n_DUs
n_actions = env.action_space.n

epsilon = epsilon_max

wandb.watch(Q_net)

step = 0
ep_rewards = []
for ii in range(N_EPS):
    ep_reward = 0
    # reset and get the init state
    ob = env.reset()
    u, v = adjacency_vbpp_to_graph(ob)
    nodes_RU, nodes_DU = state_to_node_fea(env)
    g = dgl.graph((u, v), num_nodes=n_nodes)
    g.ndata['x'] = torch.tensor(np.concatenate([nodes_RU, nodes_DU]), dtype=torch.double)

    done = False
    while not done:        
        a = choose_action(env, epsilon, g)
        a_graph = [env.cur_item, a+env.n_items] # action in graph format: u-->v, forms an edge
        g.add_edges([a_graph[0]], a_graph[1])

        state_action = g
        ob_, r, done, _ = env.step(a)
        ep_reward += r
        # new ob to new state
        u, v = adjacency_vbpp_to_graph(ob_)
        nodes_RU, nodes_DU = state_to_node_fea(env)
        g = dgl.graph((u, v), num_nodes=n_nodes)
        g.ndata['x'] = torch.tensor(np.concatenate([nodes_RU, nodes_DU]), dtype=torch.double)
        state_ = g
        # next state / action pairs (for all possible actions)
        state_action_ = []
        for t in range(env.action_space.n):
            g_tmp = copy_graph(g, 'x')
            g_tmp.add_edges([env.cur_item], t+n_RUs)
            state_action_.append(g_tmp)

        # add to memory
        memory.add((state_action, a_graph, r, state_action_, done))
        state = state_

        if len(memory.buffer) >= BATCH_SIZE:
            batch = memory.sample(BATCH_SIZE)
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
            next_q_values = torch.max(next_q_values_all.reshape(BATCH_SIZE, n_actions), axis=1).values
            q_targets = rewards + GAMMA * torch.mul(next_q_values, non_final_mask).to(device) 
            q_targets = torch.unsqueeze(q_targets, 1).to(device)

            # q_values for current Q(s,a)
            q_values = Q_net(states_actions, states_actions.ndata['x'])

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
    wandb.log({"ep score": ep_reward, "env step": step})

for ii in range(validation_eps):
    ep_score = model_validate()
    wandb.log({"evaluation ep score": ep_score})

# save Q_net
torch.save(Q_net.state_dict(), './Q_net.pt')
