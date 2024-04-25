

from .model import AutoregressiveLM
from .transformer import Transformer


def from_pretrained(model_name):
    """
    currently supports "gemma-2b", "gemma-7b"

    Args:
        model_name (string): name of the pretrained weights from huggingface
    """

    if model_name == "gemma-2b":
        return from_pretrained_gemma_2b()

    if model_name == "gemma-7b":
        return from_pretrained_gemma_7b()

    raise ValueError(f"pretrained weights {model_name} not supported")


def from_pretrained_gemma_2b():
    lm = AutoregressiveLM(
        cls=Transformer,
        vocab_size=256000,
        max_len=8192,
        n_layers=18,
        dim=2048,
        hidden_dim=16384,
        n_head=8,
        device="cpu"
    )

    for param in lm.transformer.layers[0].parameters():
        print(param.name)
        print(param.shape)


def from_pretrained_gemma_7b():
    lm = AutoregressiveLM(
        cls=Transformer,
        vocab_size=256000,
        max_len=8192,
        n_layers=28,
        dim=3072,
        hidden_dim=24576,
        n_head=16,
        device="cpu"
    )

