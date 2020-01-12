import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx


# (1x1024)x(1024x2)

class gcn(nn.Module):
    def __init__(self, num_output_classes, embedding_dim):
        super(gcn, self).__init__()
        
        self.num_output_classes = num_output_classes
        self.embedding_dim = embedding_dim
        
        self.linear = nn.Linear(embedding_dim, num_output_classes)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)


    def forward(self, graph):
        
        edges = nx.to_dict_of_dicts(graph)
        nodes = graph.nodes
        
        sum_of_node_reps = 0
        for node in nodes:
            sum_of_node_reps += graph.node[node]['representation']
        
        linear_out = self.linear(sum_of_node_reps.view(1, self.embedding_dim))
        activated_out = self.relu(linear_out)
        #softmax_out = self.softmax(activated_out)
        
        return activated_out
