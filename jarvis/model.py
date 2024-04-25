

import torch.nn as nn


class AutoregressiveLM(nn.Module):
    """
    This class is a wrapper around transformer variants for autoregressive language
    models (aka next word prediction). It uses cross entropy instead of
    negative-log-likelihood and its last layer has no bias with no activation.

    Any kind of transformer cls has parameters:
        vocab_size
        max_len
        n_layers
        dim
        n_head
        p

    and functions:
        load_pretrained()
        reset()
        set_state()
        get_state()
    """

    def __init__(self, cls, vocab_size, max_len, n_layers,
                 dim, n_head, device, **kwargs):

        super(AutoregressiveLM, self).__init__()

        self.vocab_size = vocab_size
        self.max_len = max_len
        self.n_layers = n_layers
        self.dim = dim
        self.n_head = n_head

        self.transformer = cls(
            vocab_size=vocab_size,
            max_len=max_len,
            n_layers=n_layers,
            dim=dim,
            n_head=n_head,
            device=device,
            **kwargs
        )

        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def load_pretrained(self):
        self.transformer.load_pretrained()

    def reset(self):
        return self.transformer.reset()

    def set_state(self, state):
        return self.transformer.set_state(*state)

    def get_state(self):
        return self.transformer.get_state()

    def forward(self, x):
        # no log softmax for cross entropy
        x = self.transformer(x, is_causal=True)
        x = self.lm_head(x)

        return x
