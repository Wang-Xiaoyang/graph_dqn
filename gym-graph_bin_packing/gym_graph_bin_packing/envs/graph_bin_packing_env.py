import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np

class GraphVbppEnv(gym.Env):
    """An environment to play vector bin packing."""

    def __init__(self):
        super(GraphVbppEnv, self).__init__()
        # define env constants
        self.n_items, self.n_bins = 10, 2
        self.vector_dim = 1 # the dim of each item and bin
        self.bin_capacity = 100 * np.ones(self.vector_dim)
        self.bin_load_capacity = 10 # the max number of items to be placed in one bin
        self.bins = [self.bin_capacity] * self.n_bins # init bins
        self.item_value_max, self.item_value_min = [10], [1]
        
        # init items
        self.items = []
        for i in range(self.vector_dim):
            # init each dim
            self.items.append(np.random.randint(self.item_value_min[i], self.item_value_max[i], self.n_items))
        # current item to be places
        self.cur_item = 0
        
        # init adjacency matrix; this is the adjacency mat in terms of bin packing: [n_items, n_bins]
        self.adjacency_mat = np.zeros([self.n_items, self.n_bins])

        # Define action and observation space
        self.action_space = spaces.Discrete(self.n_bins)
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
        # all items places, ep finished - bonus reward for occupied bins
        if sum(sum(self.adjacency_mat)) == self.n_items:
            occupied_bins = sum(self.adjacency_mat) > 0
            r += self.n_items * (self.n_bins - sum(occupied_bins))
        return r


    def reset(self):
        # Reset the state of the environment to an initial state
        self.bins = [self.bin_capacity] * self.n_bins # init bins
        # reset items
        self.items = []
        for i in range(self.vector_dim):
            # init each dim
            self.items.append(np.random.randint(self.item_value_min[i], self.item_value_max[i], self.n_items))
        self.cur_item = 0
        # reset adjacency matrix
        self.adjacency_mat = np.zeros([self.n_items, self.n_bins])
        return self.adjacency_mat # return s_0


    def render(self, mode='human', close=False):
        pass
