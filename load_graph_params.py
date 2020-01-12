import re
import torch
import pickle
import torch.nn as nn


def load_graph_params():
    with open('data/weights_dict', 'rb') as weights_file:
        weights_dict = pickle.load(weights_file)
    
    with open('data/word_embeddings_BERT_cleaned_vocab', 'rb') as file:
        word_embeddings = pickle.load(file)
        
    node_param = {}
    for word in word_embeddings:
        node_param[word] = nn.Parameter(torch.randn(1))
    
    return weights_dict, word_embeddings, node_param
