import gym
from gym import error, spaces, utils
from gym.utils import seeding

import math
from matplotlib.style import available
import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd

"""
Datasets: 
central_sites_a: The central area of the map (cut into 9 areas, choose the central one)
central_sites_b: The area with the highest sites density (city centre and the redcliffe area), 5 * 5 km^2

Change the path for different areas
"""

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
        self.set_RUs = pd.read_csv('./dataset_bbu_pooling/central_sites_b/central_sites.csv')
        self.set_DUs = pd.read_csv('./dataset_bbu_pooling/central_sites_b/central_dus.csv')
        # self.set_DUs = pd.read_csv('./dataset_bbu_pooling/central_sites_b/central_dus_remove_one.csv')
        self.num_RUs, self.num_DUs = len(self.set_RUs), len(self.set_DUs)        
        self.n_nodes = [self.num_RUs, self.num_DUs] # type 0: 20; type 1: 4
        self.n_nodes_total = np.sum(self.n_nodes)
        self.C = 250 # !!! capacity of type 1 nodes
        self.load_capacity = 100 # the load capacity of type 1 nodes, i.e., max # of type 0 nodes that can be connected 
        
        # consider central bristol area: easting [355558 , 359577]; northing [171899 , 176196]
        # note: if easting and northing are 6 digits, the unit is meters (in the UK at least)  
        self.nodes_x_original = np.concatenate((np.array(self.set_RUs['easting']), np.array(self.set_DUs['easting'])))
        self.nodes_y_original = np.concatenate((np.array(self.set_RUs['northing']), np.array(self.set_DUs['northing'])))
        self.nodes_x_original, self.nodes_y_original = np.expand_dims(self.nodes_x_original, axis=1), np.expand_dims(self.nodes_y_original, axis=1)
        # normalise locations (as features for training)
        self.nodes_x = (self.nodes_x_original - np.min(self.nodes_x_original)) / (np.max(self.nodes_x_original) - np.min(self.nodes_x_original))
        self.nodes_y = (self.nodes_y_original - np.min(self.nodes_y_original)) / (np.max(self.nodes_y_original) - np.min(self.nodes_y_original))

        # for RU features and traffic
        self.slot = 3 # ! slot = 3 or 7 (4G sites slot)
        self.sector = 'A11'

        # read csv files
        self.data_cpu_all = []
        self.data_traffic_all = []
        self.read_cpu_traffic()
        
        # power consumption parameters (dBm)
        self.p_static = 51.0
        self.p_max_DU = 53.0
        self.delta_DU = 0.44
        # Watts
        self.p_cooling = 500
        # record power consumption of the system
        self.power_c = 0.0

        np.random.seed()
        self.t = np.random.randint(24)
        self.t_norm = (self.t + 1) / 24
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
    
    def read_cpu_traffic(self):
        # read csv files, for cpu data
        list_RUs = list(self.set_RUs['siteid'])
        for i in range(len(list_RUs)):
            name = './dataset_bbu_pooling/central_sites_b/central_sites_cpu_a_week/' + str(i) + '_slot_' + str(self.slot) + '_' + str(list_RUs[i]) + \
                    '_' + self.date + '_cpu.csv'
            data = pd.read_csv(name)
            self.data_cpu_all.append(data)
            # traffic data
            name = './dataset_bbu_pooling/central_sites_b/central_sites_traffic_a_week/' + str(i) + '_' + str(list_RUs[i]) + '_' + self.date + '_' + \
                self.sector + '_traffic.csv'
            data = pd.read_csv(name)
            self.data_traffic_all.append(data)
    
    def dBm2w(self, dBm_value):
        # convert dBm to milliwatts
        # milliwatts to Watts
        p = 10 ** (dBm_value / 10)
        return p * 1e-3

    def get_RU_feats(self, t, type, noise = False):
        # load RU requirements; get a list containing RU features
        # t: time, between 0 and 23 (hour)
        # type: cpu or traffic, depending on which indicator do we use as RU requirements
        list_RUs = list(self.set_RUs['siteid'])
        self.fea = []

        if type == 'CPU_LOAD_MAX':
            for i in range(len(list_RUs)):
                data = self.data_cpu_all[i]
                feat = data['CPU_LOAD_MAX']
                # empty dataset - try to approximate using the average
                if len(feat) == 0:
                    # self.fea.append(0.0)
                    to_append = np.mean(self.fea)                  
                elif t >= len(feat):
                    to_append = np.mean(self.fea)
                else:
                    to_append = feat[t]

                if noise == True:
                    # add noise: sim2real
                    n_add = 5.0 + np.random.randn(1)[0]
                    to_append += n_add  

                self.fea.append(to_append)

        return self.fea

    def step(self, a):
        # Execute one time step within the environment
        # action: associate the i_th type_0 node to j_th type_1 node; action = j; i is an environment variable
        done = False
        r = 0.0

        self.adjacency_mat_pre = self.adjacency_mat.copy()

        # check the number of nodes already associated the current type_1 node
        if sum(self.adjacency_mat[:, a]) >= self.load_capacity:
            # unable to perform the current action
            # reward = 0
            done = True
            return self.adjacency_mat, r, done, {}

        # check the remaining capacity of the j_th type_1 node
        items_total = np.dot(self.fea, self.adjacency_mat[:,a])
        if items_total + self.fea[self.cur_item] > self.C:
            # unable to perform the current action
            # reward = 0 
            done = True
            return self.adjacency_mat, r, done, {}

        # execute action
        self.adjacency_mat[self.cur_item, a] = 1
        r = self.get_reward(a, done)
        self.cur_item += 1
        # update remaining_capacity for type_1 nodes
        for i in range(self.n_nodes[1]):
            # for each type_1 node
            used = np.dot(self.fea, self.adjacency_mat[:,i])
            self.remaining_capacity[i] = self.C - used

        if self.cur_item >= self.n_nodes[0]:
            self.cur_item = 0
            done = True
            r = self.get_reward(a, done) # re-calculate reward, since this is the last step
        
        return self.adjacency_mat, r, done, {}

    def calculate_distance(self, a):
        x_1, x_2 = self.nodes_x_original[self.cur_item], self.nodes_x_original[self.n_nodes[0]+a]
        y_1, y_2 = self.nodes_y_original[self.cur_item], self.nodes_y_original[self.n_nodes[0]+a]
        dist = math.hypot(x_2 - x_1, y_2 - y_1)
        return dist

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    
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
            # cost += self.dBm2w(self.p_static + self.delta_DU * self.p_max_DU * (math.sqrt(percentage)))
            # cost += self.dBm2w(self.p_static + 0.1 * self.p_max_DU * (math.tanh(percentage*2))) # 115, 116 model

            # cost += self.dBm2w(self.p_static + 0.5 * self.p_max_DU * (math.tanh(percentage*4))) # Use this model for now
            # train 134 and beyond

            # cost += self.dBm2w(self.p_static + 0.1 * self.p_max_DU * (math.tanh(percentage*2))) # Use this model for now
            # # train 145 and beyond

            cost += self.dBm2w(self.p_static + 0.2 * self.p_max_DU * (math.tanh(percentage*3))) # Use this model for now
            # train 148 and beyond

            # cost += self.dBm2w(self.p_static + 0.05 * self.p_max_DU * (math.tanh(percentage*3))) # 108, 109 (parameter 0.1); 125 126 (.1)
            # cost += self.dBm2w(self.p_static + 0.2 * self.p_max_DU * (math.tanh(percentage*3))) # 101, 102 model; ; 119, 120; 122;
            # cost += self.dBm2w(self.p_static + self.delta_DU * self.p_max_DU * (math.tanh(percentage*5))) # dashing-bird-96 model
            # cost += self.dBm2w(self.p_static + self.delta_DU * self.p_max_DU * (self.sigmoid(percentage*6) - 0.5) * 2) # not trained

            # cost += self.dBm2w(self.p_static + self.delta_DU * self.p_max_DU * min((-percentage**3 + 2*percentage), 1)) #feasible-bird-95 model
            # cost += self.dBm2w(self.p_static + self.delta_DU * self.p_max_DU * (-percentage**2 + 2*percentage)) # dazzling-puddle-92 model
            # cost += self.dBm2w(self.p_static + self.delta_DU * self.p_max_DU * percentage**2) # azure-oath-88 model
            # cost += self.dBm2w(self.p_static + self.delta_DU * self.p_max_DU * percentage)
        cost += self.p_cooling
        return cost * 1e-3 #kilowatt

    def load_balancing(self, adjacency_mat):
        # calculate load balancing at the end of an episode
        # usisng the degree of imbalance
        percentage_list = []
        for i in range(self.n_nodes[1]):
            # for each type_1 node
            used = np.dot(self.fea, adjacency_mat[:,i])
            # only consider active DUs
            # if used != 0:
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
            data = self.data_traffic_all[i]
            feat = data[type_traffic]
            # empty dataset
            if len(feat) == 0:
                num_UEs.append(0.0)
            elif t >= len(feat):
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
            # if connected != 0:
            UE_per_DU.append(connected)
        return (max(UE_per_DU) - min(UE_per_DU)) / np.mean(UE_per_DU)     
        # return max(UE_per_DU) / np.mean(UE_per_DU) - 1.0 # normalize

    def get_reward(self, a, last_step):
        # 0. association reward (sucessful connection, for rewarding RL agent)
        r_a = sum(sum(self.adjacency_mat)) - sum(sum(self.adjacency_mat_pre))

        # 1. latency reward (distance) (latency for current RU)
        # dist = self.calculate_distance(a)
        # r_l = dist / self.c_light * 1.7 * 1e3 # latency in Milliseconds (ms)
        # r_l *= 1e1 # scaling parameter

        # # 2. power consumption (use kiloWatts) (additional power consumption in this step)
        # r_c = self.power_consumption(self.adjacency_mat) - self.power_consumption(self.adjacency_mat_pre)
        # r_c *= 5e-2 # scaling parameter

        # 3. load balancing among DUs (only when the episode reaches the end)
        # 4. fairness (only in the last step of an episode)
        if last_step == True:
            self.power_c = self.power_consumption(self.adjacency_mat)
            # r_c = self.power_consumption(self.adjacency_mat)
            # r_c *= 5e-2 # scaling parameter
            # bonus for finishing all placement (compensation for add r_lb and r_fairness)
            r_bonus = 5.0
            r_lb = self.load_balancing(self.adjacency_mat)
            r_fairness = self.fairness_UEs(self.UE_per_site, self.adjacency_mat)
        else:
            r_bonus = 0.0
            r_lb = .0
            r_fairness = .0    
        # return r_a - r_l + r_bonus - r_lb - r_fairness
        # return -np.log(r_l) + r_bonus - r_lb - r_fairness
        # return r_a * 0.5 + r_bonus - r_lb - r_fairness
        return r_bonus - r_lb - r_fairness

    def available_action(self):
        # auxiliary function: get all available actions for the current node
        available_a = []
        for a in range(self.n_nodes[1]):
            # check the number of nodes already associated the current type_1 node
            if sum(self.adjacency_mat[:, a]) < self.load_capacity:
                # check the remaining capacity of the j_th type_1 node
                items_total = np.dot(self.fea, self.adjacency_mat[:,a])
                if items_total + self.fea[self.cur_item] <= self.C:
                    available_a.append(a)
        return available_a         

    def reset(self, noise_fea=False, t=-1, date=-1):
        np.random.seed()
        if t == -1:
            self.t = np.random.randint(24)
        else:
            self.t = t

        if date != -1:
            # specify the date
            self.date = date
            self.data_cpu_all = []
            self.data_traffic_all = []
            self.read_cpu_traffic()

        self.t_norm = (self.t + 1) / 24
        # load number of UEs
        self.UE_per_site = self.load_UE_num(self.t)       
        # 1. CPU_max
        self.get_RU_feats(self.t, type='CPU_LOAD_MAX', noise=noise_fea)
        # remaining capacity of type 1 nodes (value capacity)
        self.remaining_capacity = np.ones(self.n_nodes[1]) * self.C # update in get_reward()

        self.power_c = 0.0

        # current item to be placed
        self.cur_item = 0
        
        # init adjacency matrix; this is the adjacency mat in terms of node association: [type_0, type_1]
        self.adjacency_mat = np.zeros([self.n_nodes[0], self.n_nodes[1]])
        return self.adjacency_mat # return s_0

    def render(self, mode='human', close=False):
        pass