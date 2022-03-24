import gym
from gym import error, spaces, utils
from gym.utils import seeding

import math
import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd

class GraphBBuPoolingEnv(gym.Env):
    """
    An environment for node association problems in graph
    considering multiple constraints.
    """

    def __init__(self):
        super(GraphBBuPoolingEnv, self).__init__()
        # types of nodes 
        self.types_nodes = 2
        # number of nodes: load from dataset
        # dataset is for one day - to begin with
        self.date = '2019-06-04'
        self.set_RUs = pd.read_csv('./dataset_bbu_pooling/central_sites.csv')
        self.set_DUs = pd.read_csv('./dataset_bbu_pooling/central_dus.csv')
        self.num_RUs, self.num_DUs = len(self.set_RUs), len(self.set_DUs)        
        self.n_nodes = [self.num_RUs, self.num_DUs] # type 0: 20; type 1: 4
        self.n_nodes_total = np.sum(self.n_nodes)
        self.C = 200 # !!! capacity of type 1 nodes
        self.load_capacity = 100 # the load capacity of type 1 nodes, i.e., max # of type 0 nodes that can be connected 
        
        # consider central bristol area: easting [355558 , 359577]; northing [171899 , 176196]
        # note: if easting and northing are 6 digits, the unit is meters (in the UK at least)  
        self.nodes_x = np.concatenate((np.array(self.set_RUs['easting']), np.array(self.set_DUs['easting'])))
        self.nodes_y = np.concatenate((np.array(self.set_RUs['northing']), np.array(self.set_DUs['northing'])))
        self.nodes_x, self.nodes_y = np.expand_dims(self.nodes_x, axis=1), np.expand_dims(self.nodes_y, axis=1)

        # power consumption parameters (dBm)
        self.p_static = 51.0
        self.p_max_DU = 53.0
        self.delta_DU = 0.44
        # Watts
        self.p_cooling = 500

        # for RU features and traffic
        self.slot = 3 # ! slot = 3 or 7 (4G sites slot)
        self.sector = 'A11'

        np.random.seed()
        self.t = np.random.randint(24) 
        # load number of UEs
        self.UE_per_site = self.load_UE_num(self.t)       
        # 1. CPU_max
        self.get_RU_feats(self.t, type='CPU_LOAD_MAX')
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
        self.c_light = 299792458 # m * s^-1

    def dBm2w(self, dBm_value):
        # convert dBm to milliwatts
        # milliwatts to Watts
        p = 10 ** (dBm_value / 10)
        return p * 1e-3

    def get_RU_feats(self, t, type):
        # load RU requirements; get a list containing RU features
        # t: time, between 0 and 23 (hour)
        # type: cpu or traffic, depending on which indicator do we use as RU requirements
        list_RUs = list(self.set_RUs['siteid'])
        self.fea = []
        if type == 'CPU_LOAD_MAX':
            for i in range(len(list_RUs)):
                name = './dataset_bbu_pooling/central_sites_cpu/' + str(i) + '_slot_' + str(self.slot) + '_' + str(list_RUs[i]) + \
                     '_' + self.date + '_cpu.csv'
                data = pd.read_csv(name)
                feat = data['CPU_LOAD_MAX']
                # empty dataset
                if len(feat) == 0:
                    self.fea.append(0.0)
                else:
                    self.fea.append(feat[t])            
        return self.fea

    def step(self, a):
        # Execute one time step within the environment
        # action: associate the i_th type_0 node to j_th type_1 node; action = j; i is an environment variable
        done = False

        self.adjacency_mat_pre = self.adjacency_mat.copy()

        # check the number of nodes already associated the current type_1 node
        if sum(self.adjacency_mat[:, a]) >= self.load_capacity:
            # unable to perform the current action
            done = True
            r = self.get_reward(a, False) - 100 # not counting on r_lb and r_fairness, penalize in another way
            return self.adjacency_mat, r, done, {}

        # check the remaining capacity of the j_th type_1 node
        items_total = np.dot(self.fea, self.adjacency_mat[:,a])
        if items_total + self.fea[self.cur_item] > self.C:
            done = True
            r = self.get_reward(a, False) - 100 # not counting on r_lb and r_fairness, penalize in another way
            return self.adjacency_mat, r, done, {}

        # execute action
        self.adjacency_mat[self.cur_item, a] = 1
        r = self.get_reward(a, done)
        self.cur_item += 1

        if self.cur_item >= self.n_nodes[0]:
            self.cur_item = 0
            done = True
            r = self.get_reward(a, done) # re-calculate reward, since this is the last step
        
        return self.adjacency_mat, r, done, {}

    def calculate_distance(self, a):
        x_1, x_2 = self.nodes_x[self.cur_item], self.nodes_x[self.n_nodes[0]+a]
        y_1, y_2 = self.nodes_y[self.cur_item], self.nodes_y[self.n_nodes[0]+a]
        dist = math.hypot(x_2 - x_1, y_2 - y_1)
        return dist

    def power_consumption(self, adjacency_mat):
        # this is to calculate the power comsumption for type_1 nodes (DUs)
        # given adjacency matrix
        # for type_1 node
        # used capacity (percentage) = a
        cost = 0
        for i in range(self.n_nodes[1]):
            # for each type_1 node
            used = np.dot(self.fea, adjacency_mat[:,i])
            percentage = used / self.C
            cost += self.dBm2w(self.p_static + self.delta_DU * self.p_max_DU * percentage)
        cost += self.p_cooling
        return cost * 1e-3 #kilowatt

    def load_balancing(self, adjacency_mat):
        # calculate load balancing at the end of an episode
        # usisng the degree of imbalance
        percentage_list = []
        for i in range(self.n_nodes[1]):
            # for each type_1 node
            used = np.dot(self.fea, adjacency_mat[:,i])
            percentage = used / self.C
            percentage_list.append(percentage)
        
        degree_imbalance = (max(percentage_list) - min(percentage_list)) / np.mean(percentage_list)
        return degree_imbalance

    def load_UE_num(self, t):
        # load the UEs per site at time t
        list_RUs = list(self.set_RUs['siteid'])
        num_UEs = []
        type_traffic = 'USER_AVG' # can also be 'USER_MAX'
        for i in range(len(list_RUs)):
            name = './dataset_bbu_pooling/central_sites_traffic/' + str(i) + '_' + str(list_RUs[i]) + '_' + self.date + '_' + \
                self.sector + '_traffic.csv'
            data = pd.read_csv(name)
            feat = data[type_traffic]
            # empty dataset
            if len(feat) == 0:
                num_UEs.append(0.0)
            else:
                num_UEs.append(feat[t])
        return num_UEs

    def fairness_UEs(self, num_UEs, adjacency_mat):
        # calculate the fairness of UEs served per DU       
        # go through DUs, find the max number of UE served by one DU
        UE_per_DU = []
        for j in range(self.n_nodes[1]):
            # for each type_1 node
            connected = np.dot(num_UEs, adjacency_mat[:,j])
            UE_per_DU.append(connected)        
        return max(UE_per_DU)

    def get_reward(self, a, last_step):
        # 0. association reward (sucessful connection, for rewarding RL agent)
        r_a = sum(sum(self.adjacency_mat)) - sum(sum(self.adjacency_mat_pre))
        # 1. latency reward (distance) (latency for current RU)
        dist = self.calculate_distance(a)
        r_l = dist / self.c_light * 1.7 * 1e3 # latency in Milliseconds (ms)
        r_l *= 1e1 # scaling parameter
        # 2. power consumption (use kiloWatts) (additional power consumption in this step)
        r_c = self.power_consumption(self.adjacency_mat) - self.power_consumption(self.adjacency_mat_pre)
        r_c *= 1e-1 # scaling parameter
        # 3. load balancing among DUs (only when the episode reaches the end)
        # 4. fairness (only in the last step of an episode)
        if last_step == True:
            r_lb = self.load_balancing(self.adjacency_mat)
            r_fairness = self.fairness_UEs(self.UE_per_site, self.adjacency_mat)
            r_fairness *= 1e-3 # scaling
        else:
            r_lb = .0
            r_fairness = .0     
        return r_a - r_l - r_c - r_lb - r_fairness

    def reset(self):
        np.random.seed()
        self.t = np.random.randint(24)        
        # 1. CPU_max
        # reset self.fea
        self.get_RU_feats(self.t, type='CPU_LOAD_MAX')

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
