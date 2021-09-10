import gym
import numpy as np
import pandas as pd
import pdb

from graph_dqn_utilities_network import *

env = gym.make("gym_graph_bin_packing_network:graph_vbpp_network-v0")

df = pd.read_csv('bristol_nodes_1.csv')
# get n_nodes
nodes_types = ['T', 'M', 'O']
n_nodes = []
for item in nodes_types:
    list_nodes = get_node_id_types(item, df)
    n_nodes.append(len(list_nodes))
# bandwidth from this type of nodes to others
bandwidth = [10, 100, 0]
C = [100, 500, 1000]

# get items_list
k = 2e-3
items_list = np.array(df['Aggregate Residential Customers']) * k
# get coordinates
northing_list = list(df['northing'])
easting_list = list(df['easting'])

pdb.set_trace()

# node features for graph
nodes_fea_no_capacity = data_to_node_features_no_capacity(df, n_nodes, bandwidth)

ep_reward = 0
ob = env.reset(n_nodes, C, items_list, northing_list, easting_list)

# observation to graph
u, v = adjacency_vbpp_to_graph(ob)
g = dgl.graph((u, v), num_nodes=sum(n_nodes))
g = self_loop_graph(g)
# node features
bin_capacity_list = state_to_bin_capacity(env)
nodes_fea = node_fea_with_capacity(nodes_fea_no_capacity, bin_capacity_list)
g.ndata['x'] = torch.tensor(node_fea_normalization(nodes_fea, bandwidth, C), dtype=torch.double)


done = False
while not done:
    # validate model for current Q network
    a = env.action_space.sample()
    ob_, r, done, _ = env.step(a)
    ep_reward += r
    # new ob to new state
    u, v = adjacency_vbpp_to_graph(ob_)
    g = dgl.graph((u, v), num_nodes=sum(n_nodes))
    g = self_loop_graph(g)
    # node features
    bin_capacity_list = state_to_bin_capacity(env)
    nodes_fea = node_fea_with_capacity(nodes_fea_no_capacity, bin_capacity_list)
    g.ndata['x'] = torch.tensor(node_fea_normalization(nodes_fea, bandwidth, C), dtype=torch.double)
    state_ = g

ep_reward
