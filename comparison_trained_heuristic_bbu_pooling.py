"""
Random agent; baseline.
Randomly choose between availabe actions.
"""

from turtle import color
from unittest import result
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
import random

from graph_dqn_utilities_evaluation import *
from graph_dqn_utilities import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make("gym_graph_bbu_pooling:graph_bbu_pooling-v0")
env.reset()
n_RUs, n_DUs = env.n_nodes[0], env.n_nodes[1]
n_nodes = n_RUs + n_DUs
n_actions = env.action_space.n

date = '2019-06-03'
mode = 'weekdays' # 'weekdays' or 'weekends'

date_list = []
for i in range(3, 10):
    day = '2019-06-' + "{:0>2d}".format(i)
    date_list.append(day)

date_list.pop(1)


def random_agent_to_begin(t_set=-1):
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

# for calculating reward
power_par = 0.5
power_consumption_buffer = np.ones((24, 1)) * 500
# run random agents, fill in reward buffer
random.seed(4)
for env_time in range(24):
    for i in range(20):
        _, t, power_c = random_agent_to_begin(env_time)
        if power_c != 0:
            if power_c < power_consumption_buffer[env_time]:
                power_consumption_buffer[t] = power_c

power_min = np.min(power_consumption_buffer, axis=1)

# """
# Random agent.
# """
random.seed(42) # 40;
def random_agent(env, date, t):
    ep_reward = 0
    ob = env.reset(t=t, date=date)

    done = False
    pre_power = 0.0
    while not done:
        a_to_choose = env.available_action()
        a = random.sample(a_to_choose, 1)
        _, r, done, _ = env.step(a[0])

        cur_power = env.power_consumption(env.adjacency_mat)
        cur_min = power_min[env.t]
        # r_c = (env.power_c - cur_min) / cur_min
        r_c = (cur_power - pre_power) / cur_min # use cur_min to normalise power consumption
        if r_c != 0:
            # # case 2: power only
            # r = -np.log(r_c + power_par)
            # case 1: balance
            r -= np.log(r_c + power_par)
            pre_power = cur_power

        ep_reward += r
    latency, power, load_DI, user_DI = evaluation_metrics(env)
    return ep_reward, latency, power, load_DI, user_DI

# evaluation, one day
ep_reward_day, latency_day, power_day, load_DI_day, user_DI_day = [], [], [], [], []

if mode == 'weekdays':
    days_ = list(range(4))
elif mode == 'weekends':
    days_ = list(range(4, 6))

for t in days_:
    date = date_list[t]
    for i in range(24):
        ep_reward, latency, power, load_DI, user_DI = random_agent(env, t=i, date=date)
        latency_day.append(latency)
        power_day.append(power)
        load_DI_day.append(load_DI)
        user_DI_day.append(user_DI)
        ep_reward_day.append(ep_reward)

print('average for random: ', np.mean(ep_reward_day), np.mean(latency_day), np.mean(power_day), np.mean(load_DI_day), np.mean(user_DI_day))
random.seed()

# """
# User fair agent.
# """
def p_fairness_agent(env, date, t):
    ep_reward = 0
    ob = env.reset(t=t, date=date)

    done = False
    pre_power = 0.0
    while not done:
        a_to_choose = env.available_action()
        connected_list = []
        for i in range(env.n_nodes[1]):
            if i in a_to_choose:
                # for each type_1 node
                connected = np.dot(env.UE_per_site, env.adjacency_mat[:,i])
                connected_list.append(connected)

        a = a_to_choose[np.argmin(connected_list)]
        _, r, done, _ = env.step(a)
        
        cur_power = env.power_consumption(env.adjacency_mat)
        cur_min = power_min[env.t]
        # r_c = (env.power_c - cur_min) / cur_min
        r_c = (cur_power - pre_power) / cur_min # use cur_min to normalise power consumption
        if r_c != 0:
            # # case 2: power only
            # r = -np.log(r_c + power_par)
            # case 1: balance
            r -= np.log(r_c + power_par)
            pre_power = cur_power
        
        ep_reward += r
    latency, power, load_DI, user_DI = evaluation_metrics(env)
    return ep_reward, latency, power, load_DI, user_DI

# evaluation, one day
ep_reward_day, latency_day, power_day, load_DI_day, user_DI_day = [], [], [], [], []

if mode == 'weekdays':
    days_ = list(range(4))
elif mode == 'weekends':
    days_ = list(range(4, 6))

for t in days_:
    date = date_list[t]
    for i in range(24):
        ep_reward, latency, power, load_DI, user_DI = p_fairness_agent(env, t=i, date=date)
        latency_day.append(latency)
        power_day.append(power)
        load_DI_day.append(load_DI)
        user_DI_day.append(user_DI)
        ep_reward_day.append(ep_reward)

print('average for p_fairness_agent agent: ', np.mean(ep_reward_day), np.mean(latency_day), np.mean(power_day), np.mean(load_DI_day), np.mean(user_DI_day))


# """
# Load balanced agent.
# """
def p_load_balance_agent(env, date, t):
    ep_reward = 0
    ob = env.reset(t=t, date=date)

    done = False
    pre_power = 0.0
    while not done:
        a_to_choose = env.available_action()
        used_list = []
        for i in range(env.n_nodes[1]):
            if i in a_to_choose:
                # for each type_1 node
                used = np.dot(env.fea, env.adjacency_mat[:,i])
                used_list.append(used)

        a = a_to_choose[np.argmin(used_list)]
        _, r, done, _ = env.step(a)
        
        cur_power = env.power_consumption(env.adjacency_mat)
        cur_min = power_min[env.t]
        # r_c = (env.power_c - cur_min) / cur_min
        r_c = (cur_power - pre_power) / cur_min # use cur_min to normalise power consumption
        if r_c != 0:
            # # case 2: power only
            # r = -np.log(r_c + power_par)
            # case 1: balance
            r -= np.log(r_c + power_par)
            pre_power = cur_power
        
        ep_reward += r
    latency, power, load_DI, user_DI = evaluation_metrics(env)
    return ep_reward, latency, power, load_DI, user_DI

# evaluation, one day
ep_reward_day, latency_day, power_day, load_DI_day, user_DI_day = [], [], [], [], []

if mode == 'weekdays':
    days_ = list(range(4))
elif mode == 'weekends':
    days_ = list(range(4, 6))

for t in days_:
    date = date_list[t]
    for i in range(24):
        ep_reward, latency, power, load_DI, user_DI = p_load_balance_agent(env, t=i, date=date)
        latency_day.append(latency)
        power_day.append(power)
        load_DI_day.append(load_DI)
        user_DI_day.append(user_DI)
        ep_reward_day.append(ep_reward)

print('average for p_load_balance agent: ', np.mean(ep_reward_day), np.mean(latency_day), np.mean(power_day), np.mean(load_DI_day), np.mean(user_DI_day))

# """
# Latency prioritised agent.
# """
def p_latency_agent(env, date, t):
    ep_reward = 0
    ob = env.reset(t=t, date=date)

    done = False
    pre_power = 0.0
    while not done:
        a_to_choose = env.available_action()
        dist_list = []
        for a_t in a_to_choose:
            dist = env.calculate_distance(a_t)
            dist_list.append(dist)
        
        a = a_to_choose[np.argmin(dist_list)]
        _, r, done, _ = env.step(a)
        
        cur_power = env.power_consumption(env.adjacency_mat)
        cur_min = power_min[env.t]
        # r_c = (env.power_c - cur_min) / cur_min
        r_c = (cur_power - pre_power) / cur_min # use cur_min to normalise power consumption
        if r_c != 0:
            # # case 2: power only
            # r = -np.log(r_c + power_par)
            # case 1: balance
            r -= np.log(r_c + power_par)
            pre_power = cur_power
        
        ep_reward += r
    latency, power, load_DI, user_DI = evaluation_metrics(env)
    return ep_reward, latency, power, load_DI, user_DI

# evaluation, one day
ep_reward_day, latency_day, power_day, load_DI_day, user_DI_day = [], [], [], [], []

if mode == 'weekdays':
    days_ = list(range(4))
elif mode == 'weekends':
    days_ = list(range(4, 6))

for t in days_:
    date = date_list[t]
    for i in range(24):
        ep_reward, latency, power, load_DI, user_DI = p_latency_agent(env, t=i, date=date)
        latency_day.append(latency)
        power_day.append(power)
        load_DI_day.append(load_DI)
        user_DI_day.append(user_DI)
        ep_reward_day.append(ep_reward)

print('average for p_latency agent: ', np.mean(ep_reward_day), np.mean(latency_day), np.mean(power_day), np.mean(load_DI_day), np.mean(user_DI_day))


# """
# Power prioritised agent.
# """
def p_power_agent(env, date, t):
    ep_reward = 0
    ob = env.reset(t=t, date=date)

    done = False
    pre_power = 0.0
    while not done:
        a_to_choose = env.available_action()
        power_list = []
        for a_t in a_to_choose:
            adj_mat_copy = env.adjacency_mat.copy()
            adj_mat_copy[env.cur_item, a_t] = 1
            p_additional = env.power_consumption(adj_mat_copy) - env.power_consumption(env.adjacency_mat)
            power_list.append(p_additional)
        
        a = a_to_choose[np.argmin(power_list)]
        _, r, done, _ = env.step(a)

        cur_power = env.power_consumption(env.adjacency_mat)
        cur_min = power_min[env.t]
        # r_c = (env.power_c - cur_min) / cur_min
        r_c = (cur_power - pre_power) / cur_min # use cur_min to normalise power consumption
        if r_c != 0:
            # # case 2: power only
            # r = -np.log(r_c + power_par)
            # case 1: balance
            r -= np.log(r_c + power_par)
            pre_power = cur_power

        ep_reward += r
    latency, power, load_DI, user_DI = evaluation_metrics(env)
    return ep_reward, latency, power, load_DI, user_DI

# evaluation, one day
ep_reward_day, latency_day, power_day, load_DI_day, user_DI_day = [], [], [], [], []

if mode == 'weekdays':
    days_ = list(range(4))
elif mode == 'weekends':
    days_ = list(range(4, 6))

for t in days_:
    date = date_list[t]
    for i in range(24):
        ep_reward, latency, power, load_DI, user_DI = p_power_agent(env, t=i, date=date)
        latency_day.append(latency)
        power_day.append(power)
        load_DI_day.append(load_DI)
        user_DI_day.append(user_DI)
        ep_reward_day.append(ep_reward)

print('average for p_power agent: ', np.mean(ep_reward_day), np.mean(latency_day), np.mean(power_day), np.mean(load_DI_day), np.mean(user_DI_day))
plt.figure()
plt.plot(power_day)

# """
# # Trained agent
# """
#############################
# balanced agent
## this is the balanced results in the paper
run_name_list = ['super-silence-236']

# power agent
## this is the power result in the paper
run_name_list = ['genial-snow-208_250']


result_all = np.zeros((len(run_name_list), 4))

def choose_action(env, epsilon, g):
    # choose action using epsilon greedy given the current ob
    # g is the graph

    # lawful actions
    a_to_choose = env.available_action()

    if np.random.uniform(0,1) >= epsilon:
        nodes_fea = state_to_node_fea(env)
        with torch.no_grad():
            features = torch.tensor(nodes_fea, dtype=torch.double)
            g.ndata['x'] = features
            Q_values = []
            with g.local_scope():
                for i in range(env.n_nodes[1]):
                    if i in a_to_choose:
                        g.add_edges([env.cur_item], i+env.n_nodes[0])
                        g = g.to(device)
                        # make sure inputs and g are on the same device
                        inputs = g.ndata['x']
                        value = Q_net(g, inputs).detach()
                        Q_values.append(sum(value))
                        g = g.to('cpu')
                        g = dgl.remove_edges(g, torch.tensor([g.num_edges()-1])) # remove the newly added edge
        a = a_to_choose[np.argmax(Q_values)]
        # a = np.argmax(Q_values)
    else:
        # a = env.action_space.sample()
        a = random.sample(a_to_choose, 1)
        a = a[0]
    return a

def model_validate(env, date, t):
    ep_reward = 0
    ob = env.reset(t=t, date=date)

    # observation to graph
    u, v = adjacency_vbpp_to_graph(ob)
    nodes_fea = state_to_node_fea(env)
    g = dgl.graph((u, v), num_nodes=n_nodes)
    g = self_loop_graph(g)
    g.ndata['x'] = torch.tensor(nodes_fea, dtype=torch.double)

    done = False

    pre_power = 0.0
    while not done:
        # validate model for current Q network
        a = choose_action(env, 0, g)
        ob_, r, done, _ = env.step(a)

        cur_power = env.power_consumption(env.adjacency_mat)
        cur_min = power_min[env.t]
        # r_c = (env.power_c - cur_min) / cur_min
        r_c = (cur_power - pre_power) / cur_min # use cur_min to normalise power consumption
        if r_c != 0:
            # # case 2: power only
            # r = -np.log(r_c + power_par)
            # case 1: balance
            r -= np.log(r_c + power_par)
            pre_power = cur_power

        ep_reward += r
        # new ob to new state
        u, v = adjacency_vbpp_to_graph(ob_)
        nodes_fea = state_to_node_fea(env)
        g = dgl.graph((u, v), num_nodes=n_nodes)
        g = self_loop_graph(g)
        g.ndata['x'] = torch.tensor(nodes_fea, dtype=torch.double)
        state_ = g
    latency, power, load_DI, user_DI = evaluation_metrics(env)
    return ep_reward, latency, power, load_DI, user_DI

if mode == 'weekdays':
    days_ = list(range(4))
elif mode == 'weekends':
    days_ = list(range(4, 6))

for cur_model in range(len(run_name_list)):

    run_name = run_name_list[cur_model]
    path = './Q_net_' + run_name + '.pt'
    Q_net = GQN_2(7, 16).double().to(device=device)
    Q_net.load_state_dict(torch.load(path))

    # evaluation, one day
    ep_reward_day, latency_day, power_day, load_DI_day, user_DI_day = [], [], [], [], []

    for t in days_:
        date = date_list[t]
        for i in range(24):
            ep_reward, latency, power, load_DI, user_DI = model_validate(env, t=i, date=date)
            latency_day.append(latency)
            power_day.append(power)
            load_DI_day.append(load_DI)
            user_DI_day.append(user_DI)
            ep_reward_day.append(ep_reward)

    print('average for trained: ', np.mean(ep_reward_day), np.mean(latency_day), np.mean(power_day), np.mean(load_DI_day), np.mean(user_DI_day))
    # plt.plot(power_day, 'bo')
    # plt.show()

    result_all[cur_model, 0] = np.mean(ep_reward_day)
    result_all[cur_model, 1] = np.mean(power_day)
    result_all[cur_model, 2] = np.mean(load_DI_day)
    result_all[cur_model, 3] = np.mean(user_DI_day)

pd.DataFrame(result_all).to_csv("./models_all.csv")


# # write file
# df_links_2 = pd.DataFrame()
# df_links_2['source nodes'] = link_s
# df_links_2['destination nodes'] = link_d
# df_links_2['source nodes type'] = link_s_type
# df_links_2['destination nodes type'] = link_d_type
# df_links_2['latency(ms)'] = latency_list
# df_links_2['destination node power consumption'] = pow_con_list
# df_links_2['destination node resource utilisation'] = resource_util_list

# df_links_2.to_csv(folder + 'heuristic_agent_results_links_2.csv')
