

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiQueryAttention(nn.Module):

    def __init__(
        self,
        dim,
        n_head,
        num_kv_heads: int=1
    ):
        super().__init__()

        head_dim = dim // n_head

        self.n_head = n_head
        self.num_kv_heads = num_kv_heads

        assert self.n_head % self.num_kv_heads == 0
        self.num_queries_per_kv = self.n_head // self.num_kv_heads

        self.dim = dim
        self.head_dim = head_dim

        self.q_size = self.n_head * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.scaling = self.head_dim**-0.5

        self.qkv_proj = nn.Linear(self.dim, (self.n_head + 2 * self.num_kv_heads) * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, self.dim, bias=False)

    def forward(
        self, x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states_shape = x.shape
        assert len(hidden_states_shape) == 3

        batch_size, input_len, _ = hidden_states_shape

        qkv = self.qkv_proj(x)
        xq, xk, xv = qkv.split([self.q_size, self.kv_size, self.kv_size],
                               dim=-1)

        xq = xq.view(batch_size, -1, self.n_head, self.head_dim)
        xk = xk.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, -1, self.num_kv_heads, self.head_dim)

        key = k_cache
        value = v_cache
        if self.num_kv_heads != self.n_head:
            # [batch_size, max_seq_len, n_local_heads, head_dim]
            key = torch.repeat_interleave(key, self.num_queries_per_kv, dim=2)
            value = torch.repeat_interleave(value,
                                            self.num_queries_per_kv,
                                            dim=2)

        # [batch_size, n_local_heads, input_len, head_dim]
        q = xq.transpose(1, 2)
        # [batch_size, n_local_heads, max_seq_len, head_dim]
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        # [batch_size, n_local_heads, input_len, max_seq_len]
        scores = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(q)

        # [batch_size, n_local_heads, input_len, head_dim]
        output = torch.matmul(scores, v)

        # [batch_size, input_len, hidden_dim]
        output = (output.transpose(1, 2).contiguous().view(
            batch_size, input_len, -1))
        output = self.o_proj(output)
        return output

# class MultiQueryAttention(nn.Module):
#     """
#     Attention module for Transformer layers.
#     Based on the One Write Head is all you need paper.
#     Basically uses one kv pair for all the queries.
#     """
#
#     def __init__(self, dim, n_head):
#         super(MultiQueryAttention, self).__init__()
#         self.n_head = n_head
#
#         self.w_q = nn.Linear(dim, dim, bias=False)
#         self.w_kv = nn.Linear(dim, 2 * (dim // n_head), bias=False)
#         self.w_concat = nn.Linear(dim, dim, bias=False)
#
#     def forward(self, q, kv, mask=None, is_causal=False):
#         """
#         Args:
#             q:     [batch_size, length, dim]
#             kv:    [batch_size, length, dim]
#         Returns:
#             out:   [batch_size, length, dim]
#         """
#         q, k, v = self.w_q(q), *self.w_kv(kv).chunk(2, dim=-1)
#         q, k, v = self.split(q), k.unsqueeze(1), v.unsqueeze(1)
#
#         out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal)
#
#         out = self.concat(out)
#         out = self.w_concat(out)
#
#         return out
#
#     def split(self, tensor):
#         """
#         Split tensor into number of head
#
#         Args:
#             tensor: [batch_size, length, dim]
#         Returns:
#             tensor: [batch_size, head, length, d_tensor]
#         """
#         batch_size, length, dim = tensor.shape
#
#         d_tensor = dim // self.n_head
#         tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
#
#         return tensor
#
#     def concat(self, tensor):
#         """
#         Inverse function of self.split(tensor : torch.Tensor)
#
#         Args:
#             tensor: [batch_size, head, length, d_tensor]
#         Returns:
#             tensor: [batch_size, length, dim]
#         """
#         batch_size, head, length, d_tensor = tensor.shape
#         dim = head * d_tensor
#
#         tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, dim)
#         return tensor


class Attention(nn.Module):
    """
    Attention module for Transformer layers.
    Composes of learnable parameters in
    query, key, value and concat linear modules.
    """

    def __init__(self, dim, n_head):
        super(Attention, self).__init__()
        self.n_head = n_head

        self.w_q = nn.Linear(dim, dim, bias=True)
        self.w_k = nn.Linear(dim, dim, bias=True)
        self.w_v = nn.Linear(dim, dim, bias=True)
        self.w_concat = nn.Linear(dim, dim, bias=True)

    def forward(self, q, kv, mask=None, is_causal=False):
        """
        Args:
            q:     [batch_size, length, dim]
            kv:    [batch_size, length, dim]
        Returns:
            out:   [batch_size, length, dim]
        """
        q, k, v = self.w_q(q), self.w_k(kv), self.w_v(kv)
        q, k, v = self.split(q), self.split(k), self.split(v)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal)

        out = self.concat(out)
        out = self.w_concat(out)

        return out

    def split(self, tensor):
        """
        Split tensor into number of head

        Args:
            tensor: [batch_size, length, dim]
        Returns:
            tensor: [batch_size, head, length, d_tensor]
        """
        batch_size, length, dim = tensor.shape

        d_tensor = dim // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)

        return tensor

    def concat(self, tensor):
        """
        Inverse function of self.split(tensor : torch.Tensor)

        Args:
            tensor: [batch_size, head, length, d_tensor]
        Returns:
            tensor: [batch_size, length, dim]
        """
        batch_size, head, length, d_tensor = tensor.shape
        dim = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, dim)
        return tensor

