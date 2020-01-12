import os
import torch
import pickle
import networkx as nx
from tqdm import tqdm_notebook

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



def load_graph_params():
    with open('data/weights_dict', 'rb') as weights_file:
        weights_dict = pickle.load(weights_file)
    
    with open('data/word_embeddings_BERT_cleaned_vocab', 'rb') as file:
        word_embeddings = pickle.load(file)
    
    node_param = {}
    for word in word_embeddings:
        node_param[word] = torch.randn(1)
    
    return weights_dict, word_embeddings, node_param


def train(graphs, df, model, loss_function, optimizer, num_epochs):
    weights_dict, word_embeddings, node_param = load_graph_params()
    
    for epoch in range(int(num_epochs)):
        for i in range(len(graphs)):
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

            activated_out = model(graphs[i])
            
            print(activated_out)
            break
            loss = loss_function(activated_out, df['label'][i])
            loss.backward()
            optimizer.step()

            break
        break



