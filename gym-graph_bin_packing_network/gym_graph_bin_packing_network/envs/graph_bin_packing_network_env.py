import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import copy
import math

class GraphVbppNetEnv(gym.Env):
    """An environment to play vector bin packing.
    Using network data: three tiers of nodes, tier 1 (edge) --> tier 2 (metro) --> tier 3 (core) nodes
    """

    def __init__(self):
        super(GraphVbppNetEnv, self).__init__()
        # define env constants
        self.c_light = 299792458e-3 # so the latency is in \mu s
        self.n_node_types = 3
        self.n_nodes = [20, 1, 1]
        self.n_tier_1, self.n_tier_2, self.n_tier_3 = self.n_nodes[0], self.n_nodes[1], self.n_nodes[2]
        self.vector_dim = 1 # the dim of each item and bin - [cpu] or [cpu, memory] or ...
        self.C = [100, 500, 1000]
        self.C_1, self.C_2, self.C_3 = self.C[0], self.C[1], self.C[2]
        # bin capacity = [T, M, O]; for T, it contains all tier 1 nodes; same applies to M and O
        self.bin_capacity = [[self.C[i] * np.ones(self.vector_dim)] * self.n_nodes[i] for i in range(self.n_node_types)]
        # self.bin_load_capacity = 10 # the max number of items to be placed in one bin
        # self.bins = [self.bin_capacity] * self.n_bins # init bins; 
        self.item_value_max, self.item_value_min = [10], [1]
        
        # # init items
        self.items = np.ones(sum(self.n_nodes))
        
        # init adjacency matrix; this is the adjacency mat in terms of bin packing: [n_nodes, n_nodes]
        self.adjacency_mat = np.zeros([sum(self.n_nodes), sum(self.n_nodes[1:])])

        # Define action and observation space
        # action space: connect with metro or core nodes (DUs)
        self.action_space = spaces.Discrete(sum(self.n_nodes[1:]))
        # self.observation_space = spaces.Box(low=0, high=1, shape=
        #                 (self.n_bins, self.n_items), dtype=np.float64)

    def step(self, a):
        # Execute one time step within the environment
        # action: put i_th item to j_th bin; action = j, i is an env variable
        done = False
        r = 0

        self.adjacency_mat_pre = self.adjacency_mat.copy()
        self.bin_capacity_pre = copy.deepcopy(self.bin_capacity_n)

        # execute action
        self.adjacency_mat[self.cur_item, a] = 1
        # get bin capacity after this action
        self.bin_capacity_n = self.adjacent_mat_capacity()
        
        # check if there's negative capacity
        sum_neg = 0
        for i in [1,2]:
            sum_neg += sum(np.array(self.bin_capacity_n[i]) < 0)
        if sum_neg > 0:
            # reverse action
            self.adjacency_mat[self.cur_item, a] = 0
            done = True
        else:
            dist = self.calculate_distance(self.northing, self.easting, a)
            r = self.get_reward(dist)
            self.cur_item += 1

        if self.cur_item >= sum(self.n_nodes[:2]):
            self.cur_item = 0
            done = True
        
        return self.adjacency_mat, r, done, {}

    def get_reward(self, dist):
        # get the reward from current adjacency matrix
        r = sum(sum(self.adjacency_mat)) - sum(sum(self.adjacency_mat_pre))
        # bandwidth cost
        if self.cur_item < self.n_tier_1:
            r_b = 0.01 * 10
        else:
            r_b = 0.01 * 100
        # latency cost
        r_l = dist / self.c_light * 1.7 # latency in Millisecond (ms)
        # power consumption cost
        r_p = self.power_consumption(self.bin_capacity_n) - self.power_consumption(self.bin_capacity_pre)
        # deployment cost
        r_d = self.compute_deployment(self.adjacency_mat) - self.compute_deployment(self.adjacency_mat_pre)

        r = r - r_b - r_l - r_p - r_d
        return r

    def adjacent_mat_capacity(self):
        """
        Calculate remaining capacity for each DU (M and O) node given an adjacency matrix.
        """
        bin_capacity = copy.deepcopy(self.bin_capacity)
        # M nodes
        for i in range(self.n_tier_2):
            if sum(self.adjacency_mat[:,i]) > 0:
                indices = []
                for index, item in enumerate(self.adjacency_mat[:,i]):
                    if item == 1:
                        indices.append(index)
                items_in_du = self.items[indices]
                for item in items_in_du:
                    bin_capacity[1][i] = bin_capacity[1][i] - item

        # O nodes
        for i in range(self.n_tier_3):
            if sum(self.adjacency_mat[:,i+self.n_tier_2]) > 0:
                indices = []
                for index, item in enumerate(self.adjacency_mat[:,i+self.n_tier_2]):
                    if item == 1:
                        indices.append(index)
                items_in_du = self.items[indices]
            for item in items_in_du:
                bin_capacity[2][i] = bin_capacity[2][i] - item

        # Between M and O nodes
        for i in range(self.n_tier_2):
            if sum(self.adjacency_mat[i+self.n_tier_1, sum(self.n_nodes[:2]):]) > 0:
                ind = self.adjacency_mat[i+self.n_tier_1, sum(self.n_nodes[:2]):].index(1)
                r_consumed = self.bin_capacity[1][i] - bin_capacity[1][i]
                bin_capacity[2][ind] = bin_capacity[2][ind] - r_consumed
                bin_capacity[1][i] = self.bin_capacity[1][i]
        
        return bin_capacity

    
    def calculate_distance(self, northing, easting, a):
        x_1, y_1 = easting[self.cur_item], northing[self.cur_item]
        x_2, y_2 = easting[a+self.n_tier_1], northing[a+self.n_tier_1]
        dist = math.hypot(x_2 - x_1, y_2 - y_1)
        return dist

    def power_consumption(self, bin_capacity):
        power = 0
        # for M nodes
        for i in range(self.n_tier_2):
            used = self.bin_capacity[1][i] - bin_capacity[1][i]
            percentage = used / self.C_2
            power = power + (percentage**2 + 2 * percentage) * 0.5
        # for O nodes
        for i in range(self.n_tier_3):
            used = self.bin_capacity[2][i] - bin_capacity[2][i]
            percentage = used / self.C_3
            power = power + (percentage**2 + 2 * percentage) * 1.0
        return power

    def compute_deployment(self, adjacency_mat):
        """
        Calculate the compute deployment cost given adjacency matrix.
        """
        activated = np.sum(adjacency_mat, axis=0) > 0
        deployment_cost = sum(activated[0:self.n_tier_2]) * 0.2 + sum(activated[self.n_tier_2:]) * 0.1
        return deployment_cost


    def reset(self, n_nodes, C, items_list, northing_list, easting_list):
        self.n_nodes = n_nodes
        self.n_tier_1, self.n_tier_2, self.n_tier_3 = self.n_nodes[0], self.n_nodes[1], self.n_nodes[2]

        self.C = C
        self.bin_capacity = [[self.C[i] * np.ones(self.vector_dim)] * self.n_nodes[i] for i in range(self.n_node_types)]
        self.C_1, self.C_2, self.C_3 = self.C[0], self.C[1], self.C[2]
        # reset bin capacity
        self.bin_capacity = [[self.C[i] * np.ones(self.vector_dim)] * self.n_nodes[i] for i in range(self.n_node_types)]

        # reset items
        self.items = items_list
        assert len(self.items) == sum(self.n_nodes)

        # coordinates
        self.northing, self.easting = northing_list, easting_list
        assert len(self.northing) == sum(self.n_nodes)
        assert len(self.easting) == sum(self.n_nodes)
        
        self.cur_item = 0
        # reset adjacency matrix
        self.adjacency_mat = np.zeros([sum(self.n_nodes), sum(self.n_nodes[1:])])
        # Core nodes: adjacent to itself
        for item in list(reversed(range(sum(self.n_nodes))))[:self.n_tier_3]:
            self.adjacency_mat[item, item-self.n_tier_1] = 1

        self.bin_capacity_n = self.adjacent_mat_capacity()
        # observation and action space
        self.action_space = spaces.Discrete(sum(self.n_nodes[1:]))
        # self.observation_space = spaces.Box(low=0, high=1, shape=
        #                 (self.n_bins, self.n_items), dtype=np.float64)
        return self.adjacency_mat # return s_0


    def render(self, mode='human', close=False):
        pass
