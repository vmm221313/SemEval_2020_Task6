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


def train(graphs, df, model, loss_function, optimizer, num_epochs, weights_dict, word_embeddings, node_param):
    model.train()
    for epoch in tqdm_notebook(range(int(num_epochs))):
        for i in tqdm_notebook(range(len(graphs))):

            graphs = get_message_and_update(i, graphs, word_embeddings, node_param)
            
            model.zero_grad()
                    
            activated_out = model(graphs[i])

            loss = loss_function(activated_out, torch.tensor([df['label'][i]]))
            loss.backward(retain_graph=True)
            optimizer.step()

            #print(node_param['#'])
    return model, graphs
