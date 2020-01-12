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
from make_graph import build_graphs
from preprocess import load_and_preprocess

df = load_and_preprocess()

num_adj = 3
num_output_classes = 2
embedding_dim = 1024
num_epochs = 5

graphs = build_graphs(df, num_adj)

model = gcn(num_output_classes, embedding_dim)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

model, graphs = train(graphs, df, model, loss_function, optimizer, num_epochs)


