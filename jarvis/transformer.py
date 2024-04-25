

import torch.nn as nn

from .embedding import TokenEmbedding
from .layer import AttentionLayer


class Transformer(nn.Module):
    """
    A standard Transformer module that outputs the unprocessed
    output of the last transformer layer

    Args:
        vocab_size (int): Vocabulary size
        max_len (int): Max length
        n_layers (int): Number of layers
        dim (int): Dimension of transformer
        n_head (int): Number of attention heads
        p (int): Dropout probability

    """

    def __init__(self, vocab_size, max_len, n_layers,
                 dim, hidden_dim, n_head, device, **kwargs):

        super(Transformer, self).__init__()
        self.n_layers = n_layers
        self.dim = dim
        self.device = device

        self.embedding = TokenEmbedding(vocab_size=vocab_size, dim=dim)

        self.layers = nn.ModuleList([AttentionLayer(dim=dim,
                                                    hidden_dim=hidden_dim,
                                                    n_head=n_head)
                                    for _ in range(n_layers)])

        self.reset()

    def reset(self):
        self.state = None

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state

    # @torch.autocast("cuda", dtype=torch.float16)
    def forward(self, ids, is_causal):
        """
        Computes transformer output

        Args:
            ids (Tensor[batch_size, length]): tokens
            state (Tensor[batch_size, state_len, dim]): recurrent state
        Returns:
            x (Tensor[batch_size, length, dim]): output
            state (Tensor[batch_size, length, dim]): next recurrent state

        """
        x = self.embedding(ids)

        for layer in self.layers:
            x = layer(x, is_causal=is_causal)

        return x
