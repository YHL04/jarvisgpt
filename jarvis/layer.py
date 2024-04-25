

import torch.nn as nn

from .attention import MultiQueryAttention
from .feedforward import GEGLU
from .norm import RMSNorm


class AttentionLayer(nn.Module):
    """
    Class representing a standard transformer layer. This layer includes self-attention,
    normalization, dropout, and a feed-forward network

    Args:
        dim (int): The dimension of the model
        ffn_dim (int): The size of the hidden layer in the feed forward network
        n_head (int): The number of attention heads
        p (float): The probability of dropout
    """

    def __init__(self, dim, hidden_dim, n_head):
        super(AttentionLayer, self).__init__()
        self.attention = MultiQueryAttention(dim=dim, n_head=n_head)
        self.ffn = GEGLU(dim=dim, hidden_dim=hidden_dim)

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x, mask=None, is_causal=False):
        """Compute the output of the transformer layer"""
        _x = x
        x = self.norm1(x)
        x = self.attention(q=x, kv=x, mask=mask, is_causal=is_causal)
        x = x + _x

        _x = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + _x

        return x
