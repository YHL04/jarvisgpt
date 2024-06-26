

import torch.nn as nn
import torch.nn.functional as F


class FFN(nn.Module):
    """
    A simple feed forward network to be used in transformer layers.

    Architecture:
        Sequential(
            LayerNorm(dim)
            Linear(dim, inner_dim)
            GELU()
            Linear(inner_dim, dim)
        )

    Args:
        dim (int): The dimension of the input and output
        inner_dim (int): The dimension of the hidden layer
    """

    def __init__(self, dim, hidden_dim):
        super(FFN, self).__init__()

        self.ff = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.ff(x)


class GEGLU(nn.Module):
    """
    A GEGLU Module from GLU variants improve transformer paper.

    Architecture:
        FFN_gelu(x, W, V, W_2) = (GELU(xW) * xV)W_2

    Args:
        dim (int): The dimension of the input and output
        inner_dim (int): The dimension of the hidden layer
    """
    def __init__(self, dim, hidden_dim):
        super(GEGLU, self).__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        gate = F.gelu(gate, approximate="tanh")
        up = self.up_proj(x)
        fuse = gate * up
        outputs = self.down_proj(fuse)
        return outputs

