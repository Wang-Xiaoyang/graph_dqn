import numpy as np
import gym
import torch
import time
import random

import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
import networkx as nx

from collections import deque

from graph_dqn_utilities import *

import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#device = "cpu"

# BBU pooling
wandb.init(entity='xiaoyang',
            project='graph-q-learning-bbu-pooling-area-b')

#env config
# wandb.config.N_RUs = 20
# wandb.config.N_DUs = 4
wandb.config.model = 'GCN'

wandb.config.input_feats = 7
wandb.config.hidden_size = 16

wandb.config.N_EPS = 1000
wandb.config.BUFFER_LEN = 2000
wandb.config.batch_size = 64
wandb.config.GAMMA = 0.98
wandb.config.LEARNING_RATE = 0.001
wandb.config.UPDATE_STEP = 10
wandb.config.epsilon_max = 0.95
wandb.config.epsilon_min = 0.000
wandb.config.epsilon_decay = 0.992
wandb.config.validation_eps = 200
wandb.config.power_buffer_len = 50
wandb.config.reward_buffer_len = 800
wandb.config.reward_percentile = 90

wandb.config.power_par = 0.5
config = wandb.config

def choose_action(env, epsilon, g):
    # choose action using epsilon greedy given the current ob
    # g is the graph

    # # lawful actions
    a_to_choose = env.available_action()

    if np.random.uniform(0,1) >= epsilon:
        nodes_fea = state_to_node_fea(env)
        with torch.no_grad():
            features = torch.tensor(nodes_fea, dtype=torch.double)
            g.ndata['x'] = features
            Q_values = []
            with g.local_scope():
                for i in range(env.n_nodes[1]):
                    # if i in a_to_choose:
                    g.add_edges([env.cur_item], i+env.n_nodes[0])
                    g = g.to(device)
                    # make sure inputs and g are on the same device
                    inputs = g.ndata['x']
                    value = Q_net(g, inputs).detach()
                    Q_values.append(sum(value))
                    g = g.to('cpu')
                    g = dgl.remove_edges(g, torch.tensor([g.num_edges()-1])) # remove the newly added edge
        # a = a_to_choose[np.argmax(Q_values)]
        a = np.argmax(Q_values)
    else:
        a = env.action_space.sample()
        # a = random.sample(a_to_choose, 1)
        # a = a[0]
    return a

def random_agent(t_set=-1):
    if t_set == -1:
        ep_reward = 0
        ob = env.reset()

        done = False
        while not done:
            a = env.action_space.sample()
            ob_, r, done, _ = env.step(a)
            ep_reward += r
    else:
        ep_reward = 0
        ob = env.reset(t=t_set)

        done = False
        while not done:
            a = env.action_space.sample()
            ob_, r, done, _ = env.step(a)
            ep_reward += r
    return ep_reward, env.t, env.power_c

# parameters
N_EPS = config.N_EPS
BUFFER_LEN = config.BUFFER_LEN
batch_size = config.batch_size
GAMMA = config.GAMMA
LEARNING_RATE = config.LEARNING_RATE
UPDATE_STEP = config.UPDATE_STEP
epsilon_max = config.epsilon_max
epsilon_min = config.epsilon_min
epsilon_decay = config.epsilon_decay
# leave out 5 episodes, without exploration
if N_EPS > 5:
    epsilon_step = (epsilon_max - epsilon_min) / (N_EPS - 5) # how to decay epsilon
else:
    epsilon_step = (epsilon_max - epsilon_min) / N_EPS

validation_eps = config.validation_eps

# initialize memory buffer
memory = Memory(max_size=BUFFER_LEN)

# initialize two graph neural nets: Q_net, Q_net_
Q_net = GQN_2(config.input_feats, config.hidden_size).double().to(device=device)
Q_net_ = GQN_2(config.input_feats, config.hidden_size).double().to(device=device)
# copy parameters
Q_net_.load_state_dict(Q_net.state_dict())
# define optimizer
optimizer = torch.optim.Adam(Q_net.parameters(), lr=LEARNING_RATE)
# define loss
loss_MSE = nn.MSELoss().to(device)

# make env
# env = gym.make("gym_graph_bin_packing:graph_vbpp-v0")
# env.reset(config.N_RUs, config.N_DUs)
# n_RUs, n_DUs = env.n_items, env.n_bins
# n_nodes = n_RUs + n_DUs
# n_actions = env.action_space.n

# env = gym.make("gym_graph_node_association:graph_association-v0")
env = gym.make("gym_graph_bbu_pooling:graph_bbu_pooling-v0")
env.reset()
n_RUs, n_DUs = env.n_nodes[0], env.n_nodes[1]
n_nodes = n_RUs + n_DUs
n_actions = env.action_space.n

epsilon = epsilon_max

# # for use percentile reward
# # 24: 24 hours; for each time, use a buffer with pre-defined length
# reward_buffer = np.zeros((24, config.reward_buffer_len))
# # run random agents, fill in reward buffer
# for i in range(100):
#     ep_r, t = random_agent()
#     idx_r = np.argmin(reward_buffer[t, :])
#     reward_buffer[t, idx_r] = ep_r


# 24: 24 hours; for each time, use a buffer with pre-defined length
# fill in the power buffer with big numbers - this does not affect reward calculation
power_consumption_buffer = np.ones((24, 1)) * 500
# run random agents, fill in reward buffer
random.seed(4)
for env_time in range(24):
    for i in range(20):
        _, t, power_c = random_agent(env_time)
        if power_c != 0:
            if power_c < power_consumption_buffer[env_time]:
                power_consumption_buffer[t] = power_c

random.seed()
# historical min power for all times; size: 1 * 24
power_min = np.min(power_consumption_buffer, axis=1)

# reward buffer
# reward_buffer = np.zeros(config.reward_buffer_len)
reward_buffer = deque(maxlen=config.reward_buffer_len)
# loss buffer
loss_buffer = deque(maxlen=config.reward_buffer_len)

loss = 0.0


def model_validate():
    ep_reward = 0
    # ob = env.reset()
    ob = env.reset(noise_fea=True)

    # observation to graph
    u, v = adjacency_vbpp_to_graph(ob)
    nodes_fea = state_to_node_fea(env)
    g = dgl.graph((u, v), num_nodes=n_nodes)
    g.ndata['x'] = torch.tensor(nodes_fea, dtype=torch.double)
    g = self_loop_graph(g)

    done = False
    pre_power = 0.0
    while not done:
        # validate model for current Q network
        a = choose_action(env, 0, g)
        ob_, r, done, _ = env.step(a)

        cur_power = env.power_consumption(env.adjacency_mat)

        # # design reward for power consumption
        # incremental reward
        cur_min = power_min[env.t]
        # r_c = (env.power_c - cur_min) / cur_min
        r_c = (cur_power - pre_power) / cur_min # use cur_min to normalise power consumption
        
        if r_c != 0:
            ## case 2: balance
            r -= np.log(r_c + config.power_par)
            ## case 1: power only
            # r = -np.log(r_c + config.power_par)

            pre_power = cur_power

        if env.power_c != 0:
            # update power buffer
            if env.power_c < cur_min:
                power_min[env.t] = env.power_c

        ep_reward += r

        # new ob to new state
        u, v = adjacency_vbpp_to_graph(ob_)
        nodes_fea = state_to_node_fea(env)
        g = dgl.graph((u, v), num_nodes=n_nodes)
        g.ndata['x'] = torch.tensor(nodes_fea, dtype=torch.double)
        g = self_loop_graph(g)
        state_ = g
    return ep_reward

print("++++++++ Model Evaluation +++++++++++++++")

# run_name = 'laced-mountain-249' # sim ends: 685
# run_name = 'exalted-fog-250' # sim ends: 885
# run_name = 'jumping-wave-252' # sim ends: 632

# path = './Q_net_' + run_name + '.pt'
# Q_net = GQN_2(7, 16).double().to(device=device)
# Q_net.load_state_dict(torch.load(path))

# for ii in range(validation_eps):
#     ep_score = model_validate()
#     metrics = {"evaluation_ep_score": ep_score}
#     wandb.log(metrics)

"""
Observation: after training 200 episodes with noisy environment, the validation
result in sim environment improves???
"""

print("++++++++ Validation: Real (noise) +++++++++++++++")
# run_name = 'laced-mountain-249_650' # sim ends: 685
# run_name = 'exalted-fog-250_850' # sim ends: 885
run_name = 'jumping-wave-252_600' # sim ends: 632

path = './Q_net_' + run_name + '.pt'
Q_net = GQN_2(7, 16).double().to(device=device)
Q_net.load_state_dict(torch.load(path))

for ii in range(validation_eps):
    ep_score = model_validate()
    metrics = {"evaluation_ep_score": ep_score}
    wandb.log(metrics)