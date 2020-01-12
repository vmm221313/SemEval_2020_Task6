import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def get_message_and_update(i, graphs, word_embeddings, node_param):
    edges = nx.to_dict_of_dicts(graphs[i])
    for edge in edges:
        adj_nodes = edges[edge]

    messages = torch.ones(len(adj_nodes), 1024)

    for j, node in enumerate(adj_nodes):
        messages[j, :] = adj_nodes[node]['weight'][0]*word_embeddings[node]

    final_message = messages.max(dim = 0).values.view(-1, 1024)[0]
    word_embeddings[edge] = final_message - node_param[edge]*final_message + node_param[edge]*word_embeddings[edge]

    for j in range(len(graphs)):
        if graphs[j].has_node(edge):
            graphs[j].node[edge]['representation'] = word_embeddings[edge]
        
    return graphs


