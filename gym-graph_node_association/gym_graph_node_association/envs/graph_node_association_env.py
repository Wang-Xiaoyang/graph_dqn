import gym
from gym import error, spaces, utils
from gym.utils import seeding

import math
import numpy as np
from numpy.core.fromnumeric import size

class GraphAssociationEnv(gym.Env):
    """
    An environment for node association problems in graph
    considering multiple constraints.
    """

    def __init__(self):
        super(GraphAssociationEnv, self).__init__()
        # types of nodes 
        self.types_nodes = 2
        self.n_nodes = [20, 4] # type 0: 20; type 1: 4
        self.n_nodes_total = np.sum(self.n_nodes)
        self.C = 64 # capacity of type 1 nodes
        self.load_capacity = 100 # the load capacity of type 1 nodes, i.e., max # of type 0 nodes that can be connected 
        
        # consider a 2D map: 10 * 10 (km)
        # p(s0) - randomly sample nodes locations; here we set the seed = 42
        np.random.seed(42)
        self.nodes_x = np.random.randint(0, 10, (self.n_nodes_total, 1))
        self.nodes_y = np.random.randint(0, 10, (self.n_nodes_total, 1))
        # p(s0) - featuers of type 0 nodes - in each episode, randomly sample form a Gaussian distribution
        # simulation of user mobility in this area
        np.random.seed()
        self.mu, self.std = 3.0, 1.0 # most of samples should be larger than 0
        self.fea = np.random.normal(loc=self.mu, scale=self.std, size=self.n_nodes[0])
        self.fea = [np.max([0, i]) for i in self.fea] # non-negative features
        # remaining capacity of type 1 nodes (value capacity)
        self.remaining_capacity = np.ones(self.n_nodes[1]) * self.C # update in get_reward()

        # current item to be placed
        self.cur_item = 0
        
        # init adjacency matrix; this is the adjacency mat in terms of node association: [type_0, type_1]
        self.adjacency_mat = np.zeros([self.n_nodes[0], self.n_nodes[1]])

        # Define action and observation space
        self.action_space = spaces.Discrete(self.n_nodes[1])
        self.observation_space = spaces.Box(low=0, high=1, shape=
                        (self.n_nodes[1], self.n_nodes[0]), dtype=np.float64)

        # constants
        self.c_light = 299792458e-3 # km * s^-1

    def step(self, a):
        # Execute one time step within the environment
        # action: associate the i_th type_0 node to j_th type_1 node; action = j; i is an environment variable
        done = False

        self.adjacency_mat_pre = self.adjacency_mat.copy()

        # check the number of nodes already associated the current type_1 node
        if sum(self.adjacency_mat[:, a]) >= self.load_capacity:
            # unable to perform the current action
            done = True
            r = self.get_reward(a)
            return self.adjacency_mat, r, done, {}

        # check the remaining capacity of the j_th type_1 node
        items_total = np.dot(self.fea, self.adjacency_mat[:,a])
        if items_total + self.fea[self.cur_item] > self.C:
            done = True
            r = self.get_reward(a)
            return self.adjacency_mat, r, done, {}

        # execute action
        self.adjacency_mat[self.cur_item, a] = 1
        r = self.get_reward(a)
        self.cur_item += 1

        if self.cur_item >= self.n_nodes[0]:
            self.cur_item = 0
            done = True
        
        return self.adjacency_mat, r, done, {}

    def calculate_distance(self, a):
        x_1, x_2 = self.nodes_x[self.cur_item], self.nodes_x[self.n_nodes[0]+a]
        y_1, y_2 = self.nodes_y[self.cur_item], self.nodes_y[self.n_nodes[0]+a]
        dist = math.hypot(x_2 - x_1, y_2 - y_1)
        return dist

    def capacity_consumption(self, adjacency_mat):
        # this is to calculate the cost associate with the used capacity
        # given adjacency matrix
        # for type_1 node
        # used capacity (percentage) = a
        # cost = - a^2 + 2 * a
        cost = 0
        for i in range(self.n_nodes[1]):
            # for each type_1 node
            used = np.dot(self.fea, adjacency_mat[:,i])
            percentage = used / self.C
            cost += np.min([2 * percentage - percentage**2 + 0.1, 1])
        return cost

    def get_reward(self, a):
        # association cost
        r_a = sum(sum(self.adjacency_mat)) - sum(sum(self.adjacency_mat_pre))
        # latency cost (distance)
        dist = self.calculate_distance(a)
        r_l = dist / self.c_light * 1.7 * 1e3 # latency in Milliseconds (ms)
        r_l *= 1e1 # scaling parameter
        # used capacity cost
        r_c = self.capacity_consumption(self.adjacency_mat) - self.capacity_consumption(self.adjacency_mat_pre)
        r_c *= 5 # scaling parameter

        # remaining capacity for type_1 nodes; env variable
        for i in range(self.n_nodes[1]):
            # for each type_1 node
            used = np.dot(self.fea, self.adjacency_mat[:,i])
            self.remaining_capacity[i] = self.C - used
        
        return r_a - r_l - r_c

    def reset(self, n_nodes = [20, 4], mu = 3.0, std = 1.0):
        self.n_nodes = n_nodes
        self.n_nodes_total = np.sum(self.n_nodes)

        self.mu, self.std = mu, std # most of samples should be larger than 0
        np.random.seed()
        self.fea = np.random.normal(loc=self.mu, scale=self.std, size=self.n_nodes[0])
        self.fea = [np.max([0, i]) for i in self.fea] # non-negative features

        self.cur_item = 0
        # init adjacency matrix; this is the adjacency mat in terms of node association: [type_0, type_1]
        self.adjacency_mat = np.zeros([self.n_nodes[0], self.n_nodes[1]])

        # Define action and observation space
        self.action_space = spaces.Discrete(self.n_nodes[1])
        self.observation_space = spaces.Box(low=0, high=1, shape=
                        (self.n_nodes[1], self.n_nodes[0]), dtype=np.float64)
        return self.adjacency_mat # return s_0


    def render(self, mode='human', close=False):
        pass
