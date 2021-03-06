import os
import torch
import pickle
import networkx as nx
from tqdm import tqdm_notebook


def build_graphs(df, num_adj, word_embeddings):
    if os.path.exists('data/graphs_dict__num_adj_'+str(num_adj)):
        with open('data/graphs_dict__num_adj_'+str(num_adj), 'rb') as graphs_file:
            graphs_dict = pickle.load(graphs_file)
            
        return graphs_dict
    
    else:
        adj_words = []
        for i in tqdm_notebook(range(len(df))):
            for j in range(len(df['text'][i])):
                for k in range(1, num_adj):
                    adj_words.append((df['text'][i][j], df['text'][i][(j+k)%len(df['text'][i])]))
                    adj_words.append((df['text'][i][j], df['text'][i][(j-k)%len(df['text'][i])]))
        
        adj_words = list(set(adj_words))
        
        weights_dict = {}
        for i in tqdm_notebook(range(len(adj_words))):
            weights_dict[adj_words[i]] = torch.randn(1)
        
        with open('data/weights_dict', 'wb') as weights_file:
            pickle.dump(weights_dict, weights_file)
        
        graphs_dict = {}
        for i in tqdm_notebook(range(len(df))):
            G = nx.DiGraph()
            for j in range(len(df['text'][i])):
                G.add_node(df['text'][i][j], representation = word_embeddings[word_to_idx[df['text'][i][j]]])

            for j in range(len(df['text'][i])):
                for k in range(1, num_adj):
                    word = df['text'][i][j]
                    next_word = df['text'][i][(j+k)%len(df['text'][i])]
                    prev_word = df['text'][i][(j-k)%len(df['text'][i])]
                    G.add_edges_from([(word, next_word, {'weight': weights_dict[weights_tuple_to_idx[word, next_word]]})])
                    G.add_edges_from([(word, prev_word, {'weight': weights_dict[weights_tuple_to_idx[word, next_word]]})])
            graphs_dict[i] = G
        
        with open('data/graphs_dict__num_adj_'+str(num_adj), 'wb') as graphs_file:
            pickle.dump(graphs_dict, graphs_file)
            
        return graphs_dict

