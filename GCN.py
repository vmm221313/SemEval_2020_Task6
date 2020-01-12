import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx


class gcn(nn.Module):
    def __init__(self, num_output_classes, embedding_dim, node_param):
        super(gcn, self).__init__()
        
        self.num_output_classes = num_output_classes
        self.embedding_dim = embedding_dim
        #self.node_param = nn.ParameterDict(node_param)
        
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
