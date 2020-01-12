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
                G.add_node(df['text'][i][j], representation = word_embeddings[df['text'][i][j]])

            for j in range(len(df['text'][i])):
                for k in range(1, num_adj):
                    word = df['text'][i][j]
                    next_word = df['text'][i][(j+k)%len(df['text'][i])]
                    prev_word = df['text'][i][(j-k)%len(df['text'][i])]
                    G.add_edges_from([(word, next_word, {'weight': weights_dict[word, next_word]})])
                    G.add_edges_from([(word, prev_word, {'weight': weights_dict[word, prev_word]})])
            graphs_dict[i] = G
        
        with open('data/graphs_dict', 'wb') as graphs_file:
            pickle.dump(graphs_dict, graphs_file)
            
        return graphs_dict

