import numpy as np
import gym
import torch

# env state to graph features
def state_to_node_fea(env):

    # node feature for RU: [0(indicates this is an RU), item_value, 1 or 0(connection status)]
    nodes_RU_value = np.transpose(env.items.copy())

    nodes_RU_adjacency = np.sum(env.adjacency_mat, axis=1)
    nodes_RU_adjacency = np.expand_dims(nodes_RU_adjacency, axis=1)

    nodes_RU_flag = np.zeros([env.n_items, 1])
    # nodes_RU_current = np.zeros([env.n_items, 1])
    # nodes_RU_current[env.cur_item] = 1

    # nodes_RU = np.concatenate([nodes_RU_flag, nodes_RU_value, nodes_RU_adjacency, nodes_RU_current], axis=1)
    nodes_RU = np.concatenate([nodes_RU_flag, nodes_RU_value, nodes_RU_adjacency], axis=1)

    # node feature for DU: [1(indicates this is a DU), remaining capacity, load(connected RUs)]
    nodes_DU_flag = np.ones([env.n_bins, 1])
    nodes_DU_value = np.zeros([env.n_bins, env.vector_dim])


    for j in range(env.n_bins):
        for i in range(env.vector_dim):
            items_total = np.dot(env.items[i], env.adjacency_mat[:,j])
            nodes_DU_value[j, i] = env.bin_capacity[i] - items_total

    nodes_DU_load = np.sum(env.adjacency_mat, axis=0)
    nodes_DU_load = np.expand_dims(nodes_DU_load, axis=1)
    nodes_DU = np.concatenate([nodes_DU_flag, nodes_DU_value, nodes_DU_load], axis=1)

    return nodes_RU, nodes_DU