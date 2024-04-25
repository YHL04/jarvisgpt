

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):

    def __init__(self):
        pass

    def forward(self, x):
        pass


class PositionalEncoding(nn.Module):
    """
    Compute sinusoid encoding from original transformer paper.

    Args:
        dim (int): dimension of model
        max_len (int): max length of transformer
    """
    def __init__(self, dim, max_len, device):
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, dim, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, dim, step=2, device=device).float()
        # 'i' means index of dim (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / dim)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / dim)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        """Obtain positional encoding according to input size"""
        batch_size, seq_len, dim = x.size()
        return self.encoding[:seq_len, :]


class LearnedPositionalEncoding(nn.Module):
    """
    Learned Positional Embedding

    Args:
        dim (int): Dimension of model
        max_len (int): Max length of transformer
    """

    def __init__(self, dim, max_len):
        super(LearnedPositionalEncoding, self).__init__()
        self.encoding = nn.Parameter(torch.randn(max_len, dim),
                                     requires_grad=True)

    def forward(self, x):
        """Return learned positional encoding according to input shape"""
        batch_size, seq_len, dim = x.size()
        return self.encoding[:seq_len, :]


class TokenEmbedding(nn.Module):
    """
    Token Embedding for transformer
    """

    def __init__(self, vocab_size, dim):
        super(TokenEmbedding, self).__init__()
        self.emb = nn.Embedding(vocab_size, dim)

    def forward(self, ids):
        """
        Args:
            ids: [batch_size, length]
            token_emb: [batch_size, length, dim]
        """
        token_emb = self.emb(ids)
        return token_emb


class TransformerEmbedding(nn.Module):
    """
    Transformer Embedding, combining positional encoding and token embedding
    """

    def __init__(self, vocab_size, dim, max_len, device):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, dim)
        self.pos_emb = PositionalEncoding(dim, max_len, device)

    def forward(self, x):
        """
        Returns complete transformer embedding for transformer layers

        Args:
            x: [batch_size, length]
        Returns:
            token_emb+pos_emb: [batch_size, length, dim]
        """
        token_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(token_emb)
        return token_emb + pos_emb

