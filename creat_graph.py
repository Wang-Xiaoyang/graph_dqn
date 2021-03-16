import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv

import networkx as nx
import matplotlib.pyplot as plt

g = dgl.DGLGraph()

g.add_nodes(10) # 10 isolated nodes are added; RUs

g.add_nodes(1) # add 1 DU



nx_g = g.to_networkx()

# pos = nx.circular_layout(nx_G)
nx.draw(nx_g, with_labels=True, node_color=[[.7, .7, .7]])
plt.show()

# print(G.number_of_edges)

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, g, inputs):
        h = self.conv1(dgl.add_self_loop(g), inputs)
        h = torch.relu(h)
        h = self.linear(h)
        return h

x = torch.randn(11, 3)
g.ndata['x'] = x


net = GCN(3, 5)

inputs = g.ndata['x']
value = net(g, inputs)

print(sum(value))

g.add_edges([2], 10)
value = net(g, inputs)

print(sum(value))


g.add_edges([2,3,5], 10)
value = net(g, inputs)

print(sum(value))
