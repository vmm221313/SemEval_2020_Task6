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
from load_graph_params import load_graph_params

df = load_and_preprocess()

num_adj = 3
num_output_classes = 2
embedding_dim = 1024
num_epochs = 5

weights_dict, word_embeddings, node_param = load_graph_params()

graphs = build_graphs(df, num_adj, word_embeddings)

model = gcn(num_output_classes, embedding_dim, node_param)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

model, graphs = train(graphs, df, model, loss_function, optimizer, num_epochs, weights_dict, word_embeddings, node_param)













word_embeddings['nu(clear']













df.iloc[4]['text']

# +
import re
for i in tqdm_notebook(range(len(df))):
    l = df.iloc[i]['text']
    l_new = []
    for j in range(len(l)):
        l_new.append(re.sub('nuclear', 'nu(clear', l[j]))

    
    df.iloc[i]['text'] = l_new

df.to_csv('data/task6_data.csv', index = False)

# +
import re
word_embeddings_cleaned = {}
for word in tqdm_notebook(word_embeddings):
    cleaned_word = re.sub('keys', '(keys', word)
    word_embeddings_cleaned[cleaned_word] = word_embeddings[word]

with open('data/word_embeddings_BERT_cleaned_vocab', 'wb') as file:
    pickle.dump(word_embeddings_cleaned, file)
# -



with open('data/word_embeddings_BERT_cleaned_vocab', 'wb') as file:
    pickle.dump(word_embeddings_cleaned, file)

a = nn.ParameterDict(node_param)

node_param['apply']

node_param

dict_t = {'appl_y':torch.nn.Parameter(torch.tensor([3.4]))}

node_param

word_embeddings_cleaned

df.iloc[0]['text']

df.iloc[i]['text'][j]

loss_function(torch.tensor([[2.6932, 3.6407]]), torch.tensor([0]))









# +
messages = torch.ones(len(adj_nodes), 1024)
for j, node in enumerate(adj_nodes):
    messages[j, :] = adj_nodes[node]['weight'][0]*word_embeddings[node]

final_message = messages.max(dim = 0).values.view(-1, 1024)[0]
word_embeddings[edge] = final_message - node_param[edge]*final_message + node_param[edge]*word_embeddings[edge]

for j in range(len(graphs)):
    if graphs[j].has_node(edge):
        graphs[j].node[edge]['representation'] = word_embeddings[edge]
# -

adj_nodes

edges = nx.to_dict_of_dicts(graphs[i])
for edge in edges:
    adj_nodes = edges[edge]

i = 1
