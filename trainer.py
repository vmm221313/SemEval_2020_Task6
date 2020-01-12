import os
import torch
import pickle
import networkx as nx
from tqdm import tqdm_notebook

from mpm import get_message_and_update

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
    
    for epoch in tqdm_notebook(range(int(num_epochs))):
        for i in tqdm_notebook(range(len(graphs))):

            graphs = get_message_and_update(i, graphs, word_embeddings, node_param)
            
            model.zero_grad()
                    
            activated_out = model(graphs[i])

            loss = loss_function(activated_out, df['label'][i])
            loss.backward()
            optimizer.step()

    
    return model, graphs


