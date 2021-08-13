import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np

class GraphVbppNetEnv(gym.Env):
    """An environment to play vector bin packing.
    Using network data: three tiers of nodes, tier 1 (edge) --> tier 2 (metro) --> tier 3 (core) nodes
    """

    def __init__(self):
        super(GraphVbppNetEnv, self).__init__()
        # define env constants
        self.n_node_types = 3
        self.n_tier_1, self.n_tier_2, self.n_tier_3 = 20, 1, 1
        self.n_nodes = [self.n_tier_1, self.n_tier_2, self.n_tier_3]
        self.vector_dim = 1 # the dim of each item and bin - [cpu] or [cpu, memory] or ...
        self.C_1, self.C_2, self.C_3 = 100, 500, 1000 # cpu capacity for tier 1, 2 and 3 nodes - we only consider cpu atm
        self.C = [self.C_1, self.C_2, self.C_3]
        # bin capacity = [T, M, O]; for T, it contains all tier 1 nodes; same applies to M and O
        self.bin_capacity = [[self.C[i] * np.ones(self.vector_dim)] * self.n_nodes[i] for i in range(self.n_node_types)]
        # self.bin_load_capacity = 10 # the max number of items to be placed in one bin
        # self.bins = [self.bin_capacity] * self.n_bins # init bins; 
        self.item_value_max, self.item_value_min = [10], [1]
        
        # # init items
        self.items = np.ones(sum(self.n_nodes))
        
        # init adjacency matrix; this is the adjacency mat in terms of bin packing: [n_nodes, n_nodes]
        self.adjacency_mat = np.zeros([sum(self.n_nodes), sum(self.n_nodes)])

        # Define action and observation space
        # action space: connect with metro or core nodes (DUs)
        self.action_space = spaces.Discrete(sum(self.n_nodes[1:])
        self.observation_space = spaces.Box(low=0, high=1, shape=
                        (self.n_bins, self.n_items), dtype=np.float64)

    def step(self, a):
        # Execute one time step within the environment
        # action: put i_th item to j_th bin; action = j, i is an env variable
        done = False

        self.adjacency_mat_pre = self.adjacency_mat.copy()

        # check the number of items already in the j_th bin
        if sum(self.adjacency_mat[:, a]) >= self.bin_load_capacity:
            # unable to place new items
            done = True
            r = self.get_reward()
            return self.adjacency_mat, r, done, {}

        # check the remaining capacity of j_th bin
        for i in range(self.vector_dim):
            items_total = np.dot(self.items[i], self.adjacency_mat[:,a])
            if items_total + self.items[i][self.cur_item] > self.bin_capacity[i]:
                done = True
                r = self.get_reward()
                return self.adjacency_mat, r, done, {}

        # execute action
        self.adjacency_mat[self.cur_item, a] = 1
        r = self.get_reward()
        self.cur_item += 1

        if self.cur_item >= self.n_items:
            self.cur_item = 0
            done = True
        
        return self.adjacency_mat, r, done, {}

    def get_reward(self):
        # get the reward from current adjacency matrix
        r = sum(sum(self.adjacency_mat)) - sum(sum(self.adjacency_mat_pre))
        # normalized by N_RUs
        r = r / self.n_items
        # all items places, ep finished - bonus reward for occupied bins
        if sum(sum(self.adjacency_mat)) == self.n_items:
            occupied_bins = sum(self.adjacency_mat) > 0
            # the lower bound for occupied bins, if all items places
            lb_bins = np.ceil(self.n_items / self.bin_load_capacity)
            # r += self.n_items * (self.n_bins - sum(occupied_bins))
            r += lb_bins / (sum(occupied_bins)) * 2
            # r += (self.n_bins - sum(occupied_bins))
        return r


    def reset(self, n_nodes, items_list):
        self.n_nodes = n_nodes
        self.n_tier_1, self.n_tier_2, self.n_tier_3 = self.n_nodes[0], self.n_nodes[1], self.n_nodes[2]
        self.bin_capacity = [[self.C[i] * np.ones(self.vector_dim)] * self.n_nodes[i] for i in range(self.n_node_types)]
        
        # reset items
        self.items = items_list
        
        self.cur_item = 0
        # reset adjacency matrix
        self.adjacency_mat = np.zeros([sum(self.n_nodes), sum(self.n_nodes)])
        # observation and action space
        self.action_space = spaces.Discrete(sum(self.n_nodes[1:])
        self.observation_space = spaces.Box(low=0, high=1, shape=
                        (self.n_bins, self.n_items), dtype=np.float64)
        return self.adjacency_mat # return s_0


    def render(self, mode='human', close=False):
        pass
