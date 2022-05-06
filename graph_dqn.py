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

# !!!! to check:
# the direction of graph (directed and undirected, what is difference in results)

# wandb log
# wandb.init(entity="xiaoyang",
#             project="graph-q-learning")
# wandb.init(entity="xiaoyang",
#             project="graph-q-learning-compare")

# env: gym-graph_node_association
# wandb.init(entity="xiaoyang",
#                 project="graph-q-learning-dynamic")

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
wandb.config.validation_eps = 30
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

def model_validate():
    ep_reward = 0
    ob = env.reset()

    # observation to graph
    u, v = adjacency_vbpp_to_graph(ob)
    nodes_fea = state_to_node_fea(env)
    g = dgl.graph((u, v), num_nodes=n_nodes)
    g.ndata['x'] = torch.tensor(nodes_fea, dtype=torch.double)
    g = self_loop_graph(g)

    done = False
    while not done:
        # validate model for current Q network
        a = choose_action(env, 0, g)
        ob_, r, done, _ = env.step(a)
        ep_reward += r
        # new ob to new state
        u, v = adjacency_vbpp_to_graph(ob_)
        nodes_fea = state_to_node_fea(env)
        g = dgl.graph((u, v), num_nodes=n_nodes)
        g.ndata['x'] = torch.tensor(nodes_fea, dtype=torch.double)
        g = self_loop_graph(g)
        state_ = g
    return ep_reward

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

wandb.watch(Q_net)

step = 0
time_count = 0
for ii in range(N_EPS):
    ep_reward = 0
    r_percentile = 0
    # reset and get the init state
    ob =env.reset(t = time_count)
    time_count += 1
    if time_count >= 24:
        time_count = 0

    u, v = adjacency_vbpp_to_graph(ob)
    nodes_fea = state_to_node_fea(env)
    g = dgl.graph((u, v), num_nodes=n_nodes)
    g = self_loop_graph(g)
    g.ndata['x'] = torch.tensor(nodes_fea, dtype=torch.double)
    state = copy.deepcopy(g)

    done = False
    loss = 0.0
    pre_power = 0.0

    while not done:        
        a = choose_action(env, epsilon, state)
        a_graph = [env.cur_item, a+n_RUs] # action in graph format: u-->v, forms an edge
        state.add_edges([a_graph[0]], a_graph[1]) # state becomes (state, action)
        
        ob_, r, done, _ = env.step(a)
        cur_power = env.power_consumption(env.adjacency_mat)

        # # design reward for power consumption
        # incremental reward
        cur_min = power_min[env.t]
        # r_c = (env.power_c - cur_min) / cur_min
        r_c = (cur_power - pre_power) / cur_min # use cur_min to normalise power consumption
        if r_c != 0:
            # power_min[env.t] = np.min([env.power_c, cur_min])
            # r -= np.log(r_c * 1e3)
            # r -= r_c * 10
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
        state_ = copy.deepcopy(g)
        # next state / action pairs (for all possible actions)
        state_action_ = []
        for t in range(env.action_space.n):
            with g.local_scope():
                g_tmp = copy.deepcopy(g)
                g_tmp.add_edges([env.cur_item], t+n_RUs)
                state_action_.append(g_tmp)

        # # add sample to memory
        memory.add((state, a_graph, r, state_action_, done))

        # if not done:
        #     memory.add((state_action, a_graph, 0.0, state_action_, done))
        # else:
        #     # get percentile value and ranked reward
        #     r_percentile = percentile_ranked_reward(reward_buffer[env.t, :], ep_reward, config.reward_percentile)
        #     memory.add((state_action, a_graph, r_percentile, state_action_, done))
        #     # add ep_reward to buffer (if ep_reward is larger than the minimal reward in the buffer)
        #     if ep_reward > np.min(reward_buffer[env.t, :]):
        #         idx_r = np.argmin(reward_buffer[env.t, :])
        #         reward_buffer[env.t, idx_r] = ep_reward
        
        # update state
        state = state_

        # training
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
            q_targets = rewards + GAMMA * torch.mul(next_q_values, non_final_mask).to(device) 
            q_targets = torch.unsqueeze(q_targets, 1).to(device)

            # q_values for current Q(s,a)
            in_fea = states_actions.ndata['x'].to(device)
            q_values = Q_net(states_actions, in_fea)

            # one step of back prop
            optimizer.zero_grad() # clear gradient
            loss = loss_MSE(q_targets, q_values)
            loss.backward()    # ep_rewards.append(ep_reward)
            optimizer.step()
            
            if step % UPDATE_STEP == 0:
                Q_net_.load_state_dict(Q_net.state_dict())
            
            step += 1
            loss_buffer.append(loss.detach().cpu().numpy())
    
    # epsilon = max((epsilon-epsilon_step), epsilon_min)
    epsilon *= epsilon_decay
    if epsilon < 0.02:
        epsilon = 0.0
    # log results after each game episode
    wandb.log({"ep score": ep_reward, "loss": loss, "env step": step, "time in data": env.t, "epsilon": epsilon})

    # add ep_reward to buffer - only keep good scores
    # if ep_reward > np.min(reward_buffer):
    #     reward_buffer[np.argmin(reward_buffer)] = ep_reward
    reward_buffer.append(ep_reward)

    if ii > 0 and ii % 50 == 0:
        # save Q_net
        torch.save(Q_net.state_dict(), './Q_net_' + wandb.run.name + '_' + str(ii) + '.pt')

    # if the average of last 10 ep_reward is higher than 90 percentile, stop training
    if epsilon < 0.05 and (np.mean(np.array(reward_buffer)[-25:]) > np.percentile(np.array(reward_buffer), config.reward_percentile) or \
        np.std(np.array(reward_buffer)[-40:]) < 0.3):
        # np.mean(np.array(loss_buffer)[-10:]) < 0.5:
        print('Stop criterion is met.')
        break

print("++++++++ Model Evaluation +++++++++++++++")
for ii in range(validation_eps):
    ep_score = model_validate()
    metrics = {"evaluation_ep_score": ep_score}
    wandb.log(metrics)

# save Q_net
torch.save(Q_net.state_dict(), './Q_net_' + wandb.run.name + '.pt')
