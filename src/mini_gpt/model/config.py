"""Finite model configurations; experiment settings are not permanent capacity ceilings."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ModelConfig:
    vocab_size: int = 256
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 512
    context_length: int = 256
    dropout: float = 0.1
    bias: bool = True
    tie_word_embeddings: bool = False
    layer_norm_eps: float = 1e-5
    pos_encoding: str = "absolute"

    def __post_init__(self) -> None:
        for name in ("vocab_size", "d_model", "n_layers", "n_heads", "d_ff", "context_length"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0 <= self.dropout < 1 or not self.layer_norm_eps > 0:
            raise ValueError("invalid dropout or layer normalization epsilon")
        if self.pos_encoding not in ("absolute", "rope", "alibi"):
            raise ValueError("unsupported position encoding")
        if self.pos_encoding == "rope" and self.head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        if type(self.bias) is not bool or type(self.tie_word_embeddings) is not bool:
            raise ValueError("bias and tie_word_embeddings must be boolean")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModelConfig":
        return cls(**value)
