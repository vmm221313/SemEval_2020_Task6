import os
import time
import torch
import pickle
import networkx as nx
from tqdm import tqdm_notebook

from mpm import get_message_and_update

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def train(graphs, df, model, loss_function, optimizer, num_epochs, weights_dict, word_embeddings, node_param, word_to_idx, weights_tuple_to_idx):
    #model.train()
    #i = len(graphs)
    num = 1000
    for epoch in tqdm_notebook(range(int(num_epochs))):
        for i in tqdm_notebook(range(num)):    
            
            graphs = get_message_and_update(i, graphs, word_embeddings, node_param, word_to_idx, weights_tuple_to_idx)
            
            #start = time.process_time()   
            model.zero_grad()
                    
            #print('time - {}'.format(time.process_time() - start))
             
            activated_out = model(graphs[i])
            loss = loss_function(activated_out, torch.tensor([df['label'][i]]))
            loss.backward(retain_graph=True)
            optimizer.step()


            
            print(loss)
            #print(node_param[word_to_idx['link']])
            #print(weights_dict[weights_tuple_to_idx[('trees', 'root')]])
            #print(word_embeddings[word_to_idx['link']])
            
            
    torch.save(model, 'saved_models/model__'+str(num)+'_graphs__'+str(num_epochs)+'_epochs')
            
    return model, graphs
