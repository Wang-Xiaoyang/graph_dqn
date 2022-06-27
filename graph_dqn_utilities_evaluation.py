import numpy as np
import gym
import torch
from collections import deque
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

import gc

import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
import networkx as nx
import copy
import math
import pandas as pd

def evaluation_metrics(env):
    # Evaluation of the result, using 4 metrics
    # Average latency, total power consumption
    # Degree of traffic imbalance, degree of user imbalance
    total_latency = 0
    if sum(sum(env.adjacency_mat)) == env.n_nodes[0]: # all type_0 nodes are connected to type_1s
        ## Latency
        for i in range(env.n_nodes[0]):
            a = np.where(env.adjacency_mat[i,:] == 1)[0]
            x_1, x_2 = env.nodes_x_original[i], env.nodes_x_original[env.n_nodes[0]+a]
            y_1, y_2 = env.nodes_y_original[env.cur_item], env.nodes_y_original[env.n_nodes[0]+a]
            dist = math.hypot(x_2 - x_1, y_2 - y_1)
            total_latency += dist / env.c_light * 1.7 * 1e3 # latency in Milliseconds (ms)
        ## total power
        total_power = env.power_c
        ## Degree of traffic imbalance
        load_DI = env.load_balancing(env.adjacency_mat)
        ## Degree of user imbalance
        user_DI = env.fairness_UEs(env.UE_per_site, env.adjacency_mat)
        return total_latency, total_power, load_DI, user_DI
    else:
        return [], [], [], []


def save_csv_results(env, method, date, time):
    # Results - in the format of graphs
    # source nodes, destination nodes, link latency
    # power util, load balancing, num of UEs

    s_nodes, d_nodes = [], []
    link_latency = []
    load_nodes = []
    UEs_nodes = []

    ## Degree of traffic imbalance
    percentage_list = []
    for i in range(env.n_nodes[1]):
        # for each type_1 node
        used = np.dot(env.fea, env.adjacency_mat[:,i])
        percentage = used / env.C
        percentage_list.append(percentage)

    # fairness
    UE_per_DU = []
    for j in range(env.n_nodes[1]):
        # for each type_1 node
        connected = np.dot(env.UE_per_site, env.adjacency_mat[:,j])
        # if connected != 0:
        UE_per_DU.append(connected)

    if sum(sum(env.adjacency_mat)) == env.n_nodes[0]: # all type_0 nodes are connected to type_1s
        ## Latency
        for i in range(env.n_nodes[0]):
            s_nodes.append(i)
            a = np.where(env.adjacency_mat[i,:] == 1)[0]
            d_nodes.append((env.n_nodes[0]+a)[0])
            load_nodes.append(percentage_list[a[0]])
            UEs_nodes.append(UE_per_DU[a[0]])
            
            x_1, x_2 = env.nodes_x_original[i], env.nodes_x_original[env.n_nodes[0]+a]
            y_1, y_2 = env.nodes_y_original[env.cur_item], env.nodes_y_original[env.n_nodes[0]+a]
            dist = math.hypot(x_2 - x_1, y_2 - y_1)
            latency = dist / env.c_light * 1.7 * 1e3 # latency in Milliseconds (ms)
            link_latency.append(latency)
        
        ## total power
        total_power = env.power_c


    source_nodes_type = [0] * len(s_nodes)
    dest_nodes_type = [1] * len(d_nodes)

    # for display only: connections between DUs
    for i in range(env.n_nodes[1]):
        for j in range(i+1, env.n_nodes[1]):
            s_nodes.append(env.n_nodes[0]+i)
            source_nodes_type.append(1)
            d_nodes.append(env.n_nodes[0]+j)
            dest_nodes_type.append(1)
            # for display only; no real meaning
            link_latency.append(0)
            load_nodes.append(percentage_list[j])
            UEs_nodes.append(UE_per_DU[j])

    # # write file
    df_links = pd.DataFrame()
    df_links['source nodes'] = s_nodes
    df_links['destination nodes'] = d_nodes
    df_links['source nodes type'] = source_nodes_type
    df_links['destination nodes type'] = dest_nodes_type
    df_links['latency(ms)'] = link_latency
    df_links['total power consumption'] = [total_power] * len(s_nodes)
    df_links['destination node resource utilisation'] = load_nodes
    df_links['dest nodes no UEs'] = UEs_nodes

    df_links.to_csv(method + '_' + date + '_' + time + '_results_links.csv')