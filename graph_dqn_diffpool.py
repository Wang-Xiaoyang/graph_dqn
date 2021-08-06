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

from model.encoder import DiffPool

import gc

from graph_dqn_utilities import *

import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
if torch.cuda.is_available():
    torch.cuda.set_device(0)

# !!!! to check:
# the direction of graph (directed and undirected, what is difference in results)

# wandb log
# wandb.init(entity="xiaoyang",
#             project="graph-q-learning-diffpool")
wandb.init(entity="xiaoyang",
            project="graph-q-learning-compare")

#env config
wandb.config.N_RUs = 10
wandb.config.N_DUs = 3
wandb.config.input_dim = 3 # feature dimension of nodes
# GCN model config
wandb.config.model = 'DiffPool'
wandb.config.hidden_dim = 8 # hidden dimension of GCN
wandb.config.Q_value_dim = 1 # Q values are 1-d numbers
wandb.config.gc_per_block = 2 # 2 GCN layers in each clustered graph
wandb.config.dropout = 0.0
wandb.config.num_pool = 1
wandb.config.clip = 10.0 # gradient clip

wandb.config.N_EPS = 500
wandb.config.BUFFER_LEN = 2000
wandb.config.batch_size = 128
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
    if np.random.uniform(0,1) >= epsilon:
        nodes_RU, nodes_Du = state_to_node_fea(env)
        with torch.no_grad():
            features = torch.tensor(np.concatenate([nodes_RU, nodes_DU]), dtype=torch.float32)
            g.ndata['feat'] = features
            Q_values = []
            with g.local_scope():
                for i in range(env.n_bins):
                    g.add_edges([env.cur_item], i+env.n_items)
                    g = g.to(device)
                    # # make sure inputs and g are on the same device
                    value = model(g).detach()
                    Q_values.append(sum(value))
                    g = g.to('cpu')
                    g = dgl.remove_edges(g, torch.tensor([g.num_edges()-1])) # remove the newly added edge
        a = np.argmax(Q_values)
    else:
        a = env.action_space.sample()
    return a

def model_validate():
    ep_reward = 0
    ob = env.reset(config.N_RUs, config.N_DUs)

    # observation to graph
    u, v = adjacency_vbpp_to_graph(ob)
    nodes_RU, nodes_DU = state_to_node_fea(env)
    g = dgl.graph((u, v), num_nodes=n_nodes)
    g.ndata['feat'] = torch.tensor(np.concatenate([nodes_RU, nodes_DU]), dtype=torch.float32)

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
        g.ndata['feat'] = torch.tensor(np.concatenate([nodes_RU, nodes_DU]), dtype=torch.float32)
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

# # initialize two graph neural nets: Q_net, Q_net_
# Q_net = GQN(config.input_dim, config.hidden_dim).double().to(device=device)
# Q_net_ = GQN(config.input_dim, config.hidden_dim).double().to(device=device)
# # copy parameters
# Q_net_.load_state_dict(Q_net.state_dict())
# # define optimizer
# optimizer = torch.optim.Adam(Q_net.parameters(), lr=LEARNING_RATE)
# # define loss
# loss_MSE = nn.MSELoss().to(device)


assign_dim = config.N_DUs
pool_ratio = config.N_DUs / (config.N_DUs + config.N_RUs)
print("++++++++++MODEL STATISTICS++++++++")
print("model hidden dim is", config.hidden_dim)
print("model embedding dim for graph instance embedding", config.hidden_dim)
print("initial batched pool graph dim is", assign_dim)
activation = F.relu

# init model
model = DiffPool(config.input_dim,
                config.hidden_dim,
                config.hidden_dim, # embedding dim
                config.Q_value_dim,
                activation,
                config.gc_per_block,
                config.dropout,
                config.num_pool,
                False,
                config.batch_size,
                'meanpool',
                assign_dim,
                pool_ratio)

model_ = DiffPool(config.input_dim,
                config.hidden_dim,
                config.hidden_dim, # embedding dim
                config.Q_value_dim,
                activation,
                config.gc_per_block,
                config.dropout,
                config.num_pool,
                False,
                config.batch_size,
                'meanpool',
                assign_dim,
                pool_ratio)            
# copy parameters
model_.load_state_dict(model.state_dict())

print("model init finished")
print("MODEL:::::::", "DiffPool")
if device.type == "cuda":
    model = model.cuda()
    model_ = model_.cuda()

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                    model.parameters()), lr=0.001)


# make env
env = gym.make("gym_graph_bin_packing:graph_vbpp-v0")
env.reset(config.N_RUs, config.N_DUs)
n_RUs, n_DUs = env.n_items, env.n_bins
n_nodes = n_RUs + n_DUs
n_actions = env.action_space.n

epsilon = epsilon_max

wandb.watch(model)

step = 0
ep_rewards = []
for ii in range(N_EPS):
    ep_reward = 0
    # reset and get the init state
    ob = env.reset(config.N_RUs, config.N_DUs)
    u, v = adjacency_vbpp_to_graph(ob)
    nodes_RU, nodes_DU = state_to_node_fea(env)
    g = dgl.graph((u, v), num_nodes=n_nodes)
    g.ndata['feat'] = torch.tensor(np.concatenate([nodes_RU, nodes_DU]), dtype=torch.float32)

    done = False
    while not done:        
        a = choose_action(env, epsilon, g)
        a_graph = [env.cur_item, a+env.n_items] # action in graph format: u-->v, forms an edge
        g.add_edges([a_graph[0]], a_graph[1])

        state_action = g
        del g
        gc.collect()
        
        ob_, r, done, _ = env.step(a)
        ep_reward += r
        # new ob to new state
        u, v = adjacency_vbpp_to_graph(ob_)
        nodes_RU, nodes_DU = state_to_node_fea(env)
        g = dgl.graph((u, v), num_nodes=n_nodes)
        g.ndata['feat'] = torch.tensor(np.concatenate([nodes_RU, nodes_DU]), dtype=torch.float32)
        state_ = g        
        # next state / action pairs (for all possible actions)
        state_action_ = []
        for t in range(env.action_space.n):
            with g.local_scope():
                g_tmp = copy_graph(g, 'feat')
                g_tmp.add_edges([env.cur_item], t+n_RUs)
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

            
            model.train()
            model.zero_grad()

            # if next state is the terminal state
            non_final_mask = torch.tensor(list(map(lambda s: s is not True, done_info)), dtype=torch.float32).to(device) #
            # next_q_values = max_a Q_target(s',a)
            next_q_values_all = model_(states_actions_).detach()
            next_q_values = torch.max(next_q_values_all.reshape(batch_size, n_actions), axis=1).values
            next_q_values = next_q_values.to(device)
            q_targets = rewards + GAMMA * torch.mul(next_q_values, non_final_mask).to(device) 
            q_targets = torch.unsqueeze(q_targets, 1).to(device)
            q_targets = q_targets.to(torch.float32)

            # q_values for current Q(s,a)
            q_values = model(states_actions)

            # one step of back prop
            loss = model.MSE_loss(q_targets, q_values)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), config.clip)
            optimizer.step()
            
            if step % UPDATE_STEP == 0:
                model_.load_state_dict(model.state_dict())
            step += 1
    
    epsilon = max((epsilon-epsilon_step), epsilon_min)
    # ep_rewards.append(ep_reward)
    wandb.log({"ep score": ep_reward, "env step": step})

print("++++++++ Model Evaluation +++++++++++++++")
for ii in range(validation_eps):
    ep_score = model_validate()
    wandb.log({"evaluation ep score": ep_score})

# save the model
torch.save(model.state_dict(), './model_diffpool.pt')
