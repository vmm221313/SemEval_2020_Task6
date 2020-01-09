import os
import torch
import pickle
import networkx as nx
from tqdm import tqdm_notebook

from make_graph import build_graphs
from preprocess import load_and_preprocess

df = load_and_preprocess()

num_adj = 3

graphs = build_graphs(df, num_adj)

graphs

graphs[0].edges

type(df['text'][i])

# all_words = []
# for i in tqdm_notebook(range(len(df))):
#     all_words += df['text'][i]

# len(all_words)

# vocab = list(set(all_words))
# len(vocab)

# i = 0
# adj_words = []
# for j in range(len(df['text'][i])):
#     for k in range(1, num_adj):
#         adj_words.append((df['text'][i][j], df['text'][i][(j+k)%len(df['text'][i])]))
#         adj_words.append((df['text'][i][j], df['text'][i][(j-k)%len(df['text'][i])]))

# df['text'][0]

# adj_words

adj_words = []
for i in tqdm_notebook(range(len(df))):
    for j in range(len(df['text'][i])):
        for k in range(1, num_adj):
            adj_words.append((df['text'][i][j], df['text'][i][(j+k)%len(df['text'][i])]))
            adj_words.append((df['text'][i][j], df['text'][i][(j-k)%len(df['text'][i])]))

adj_words[:20]

len(adj_words)

adj_words = list(set(adj_words))

len(adj_words)

weights_dict = {}
for i in tqdm_notebook(range(len(adj_words))):
    weights_dict[adj_words[i]] = torch.randn(1)

weights_dict

len(weights_dict)

with open('data/word_embeddings_BERT_cleaned_vocab', 'rb') as file:
    word_embeddings = pickle.load(file)

# G.add_edges_from([(df['text'][i][j], df['text'][i][(j+k)%len(df['text'][i])], {'weight': torch.randn(1)})])
# G.add_edges_from([(df['text'][i][j], df['text'][i][(j-k)%len(df['text'][i])], {'weight': torch.randn(1)})])

graphs_dict = {}
for i in tqdm_notebook(range(len(df))):
    G = nx.DiGraph()
    for j in range(len(df['text'][i])):
        G.add_node(df['text'][i][j], representation = word_embeddings[df['text'][i][j]])
        
    for j in range(len(df['text'][i])):
        for k in range(1, num_adj):
            word = df['text'][i][j]
            next_word = df['text'][i][(j+k)%len(df['text'][i])]
            prev_word = df['text'][i][(j-k)%len(df['text'][i])]
            G.add_edges_from([(word, next_word, {'weight': weights_dict[word, next_word]})])
            G.add_edges_from([(word, prev_word, {'weight': weights_dict[word, prev_word]})])
    graphs_dict[i] = G

graphs_dict[0].edges

graphs_dict[0]['virtual']['two']


