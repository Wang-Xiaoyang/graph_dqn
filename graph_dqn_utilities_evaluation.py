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