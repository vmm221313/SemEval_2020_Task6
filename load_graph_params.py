import re
import torch
import pickle
import torch.nn as nn


def load_graph_params():
    with open('data/weights_dict', 'rb') as weights_file:
        weights_dict = pickle.load(weights_file)
    
    with open('data/word_embeddings_BERT_cleaned_vocab', 'rb') as file:
        word_embeddings = pickle.load(file)  
        
    word_to_idx = {}
    for i, word in enumerate(word_embeddings):
        word_to_idx[word] = str(i)
    
    weights_tuple_to_idx = {}
    for i, tuple in enumerate(weights_dict):
        weights_tuple_to_idx[tuple] = str(i)
        
    weights_dict_indexed = {}
    for tuple in weights_dict:
        weights_dict_indexed[weights_tuple_to_idx[tuple]] = nn.Parameter(weights_dict[tuple])   
    
    node_param = {}
    for word in word_embeddings:
        node_param[word_to_idx[word]] = nn.Parameter(torch.randn(1, requires_grad = True))
        #node_param[word_to_idx[word]] = torch.randn(1, requires_grad = True)
    
    word_embeddings_indexed = {}
    for word in word_embeddings:
        word_embeddings_indexed[word_to_idx[word]] = nn.Parameter(word_embeddings[word])  
    
    return weights_dict_indexed, word_embeddings_indexed, node_param, word_to_idx, weights_tuple_to_idx
