

import torch
from torch.utils.data import Dataset
from datasets import load_dataset

import random
import time

from utils import tokenize, partition, filter_empty


class TextDataset(Dataset):
    """
    Args:
        cache_dir (string): The directory to cache the transformer datasets and where to retrieve them
        split (string): train / validation / test
        seq_len (int): Length per block
        block_len (int): Number of blocks to return per batch
        device (string): cuda / cpu
    """

    def __init__(self,
                 name,
                 cache_dir,
                 split,
                 seq_len,
                 block_len,
                 device="cuda",
                 sep_padding=False,
                 max_len=None
                 ):
        super().__init__()

        if name == "pg19":
            self.data = load_dataset("pg19", split=split, cache_dir=cache_dir)
            key = "text"
        elif name == "scientific_papers":
            self.data = load_dataset("scientific_papers", "arxiv", split=split, cache_dir=cache_dir)
            key = "article"

        print("Dataset loaded")
        start = time.time()

        # List[(total_len, seq_len)]
        self.data = filter_empty(partition(tokenize([data[key] for data in self.data]),
                                           max_len=seq_len),
                                 min_len=block_len + 1)
        print("Dataset tokenized and partitioned in ", time.time() - start)

        self.seq_len = seq_len
        self.block_len = block_len
        self.device = device

        if sep_padding:
            self.data = self.add_sep_padding(self.data, w=128)

        if max_len is not None:
            self.data = [x[:max_len] for x in self.data]

        self.size = sum([x.size(0) for x in self.data]) // block_len

    def __getitem__(self, index):
        """
        Index is not used

        Returns:
            output (Tensor): Tensor with shape (block_len, seq_len+1)
        """
        bidx = random.randrange(0, len(self.data))
        tidx = random.randrange(0, self.data[bidx].size(0) - self.block_len)

        last_token = self.data[bidx][tidx+1:tidx+self.block_len+1, 0].unsqueeze(-1).long()
        data = self.data[bidx][tidx:tidx+self.block_len].long()
        data = torch.concat([data, last_token], axis=-1)
        assert data.shape == (self.block_len, self.seq_len+1)

        return data.to(self.device)

    def __len__(self):
        return self.size