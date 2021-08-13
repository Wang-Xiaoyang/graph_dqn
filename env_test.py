import gym

env = gym.make("gym_graph_bin_packing_network:graph_vbpp_network-v0")

env.reset()

env.bin_capacity
a = 0