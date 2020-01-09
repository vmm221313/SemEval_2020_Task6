import os
import torch
import pickle
import networkx as nx
from tqdm import tqdm_notebook

from make_graph import build_graphs
from preprocess import load_and_preprocess

df = load_and_preprocess()

num_adj = 3

df

build_graphs(df, num_adj)


