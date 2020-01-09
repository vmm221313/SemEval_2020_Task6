import os
import torch
import pickle
import networkx as nx
from tqdm import tqdm_notebook


def build_graphs(df, num_adj):
    if os.path.exists('data/graphs_dict'):
        with open('data/graphs_dict', 'rb') as graphs_file:
            graphs_dict = pickle.load(graphs_file)
            
        return graphs_dict
    
    else:
        with open('data/word_embeddings_BERT_cleaned_vocab', 'rb') as file:
            word_embeddings = pickle.load(file)
        
        graphs_dict = {}
        for i in tqdm_notebook(range(len(df))):
            G = nx.Graph()
            for j in range(len(df['text'][i])):
                G.add_node(df['text'][i][j], representation = word_embeddings[df['text'][i][j]])
                for k in range(1, num_adj):
                    G.add_edges_from([(df['text'][i][j], df['text'][i][(j+k)%len(df['text'][i])], {'weight': torch.randn(1)})])
                    G.add_edges_from([(df['text'][i][j], df['text'][i][(j-k)%len(df['text'][i])], {'weight': torch.randn(1)})])
            graphs_dict[i] = G
        
        with open('data/graphs_dict', 'wb') as graphs_file:
            pickle.dump(graphs_dict, graphs_file)
            
        return graphs_dict
        
