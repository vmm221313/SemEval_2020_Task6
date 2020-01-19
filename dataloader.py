import pandas as po

import torch
from torch.utils.data import Dataset, DataLoader

from make_graph import build_graphs


class graphs_dict_dataloader(Dataset):
    def __init__(self, df, num_adj, word_embeddings, transform=None):
        self.df = df
        self.num_adj = num_adj
        self.word_embeddings = word_embeddings
        self.graphs = build_graphs(df, num_adj, word_embeddings)
        self.transform = transform

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


