import os
import torch
import pickle
import networkx as nx
from tqdm import tqdm_notebook

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from GCN import gcn
from trainer import train
#from make_graph import build_graphs
from dataloader import graphs_dict_dataloader
from preprocess import load_and_preprocess
from load_graph_params import load_graph_params

df = load_and_preprocess()

num_adj = 3
num_output_classes = 2
embedding_dim = 1024
num_epochs = 1

weights_dict, word_embeddings, node_param, word_to_idx, weights_tuple_to_idx = load_graph_params()

graphs_dataset = graphs_dict_dataloader(df, num_adj, word_embeddings)

graphs_dataset[8]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Using device - {}'.format(device))

model = gcn(num_output_classes, embedding_dim, node_param, weights_dict, word_embeddings)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

# +
#REINIT GRAPH WEIGHTS
# -

model, graphs = train(graphs_dataset, df, model, loss_function, optimizer, num_epochs, weights_dict, word_embeddings, node_param, word_to_idx, weights_tuple_to_idx)

model



model.eval()

for i in tqdm_notebook(range(1000, 1100)):    
    graphs = get_message_and_update(i, graphs, word_embeddings, node_param, word_to_idx, weights_tuple_to_idx)
    with torch.no_grad():
        activated_out = model(graphs[i])
        print(activated_out)

activated_out


