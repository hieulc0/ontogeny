"""Canonical dense decoder-only Transformer and shared hidden-state traversal."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mini_gpt.model.attention import CausalSelfAttention
from mini_gpt.model.config import ModelConfig
from mini_gpt.runtime import require_remote_cuda


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        require_remote_cuda()
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        require_remote_cuda()
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.mlp = FeedForward(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class MiniGPT(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        require_remote_cuda()
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.wpe = (
            nn.Embedding(config.context_length, config.d_model)
            if config.pos_encoding == "absolute"
            else None
        )
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(parameter, mean=0.0, std=0.02 / (2 * config.n_layers) ** 0.5)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def hidden_states(self, tokens: Tensor) -> Tensor:
        """One canonical traversal for token logits and task-specific readouts."""
        if self.wpe is not None and tokens.size(1) > self.config.context_length:
            raise ValueError("sequence exceeds configured absolute-position context")
        x = self.wte(tokens)
        if self.wpe is not None:
            positions = torch.arange(tokens.size(1), dtype=torch.long, device=tokens.device)
            x = x + self.wpe(positions)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

    def forward(
        self, tokens: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """Targets are already aligned next-token IDs; -1/-100 mask unsupervised tokens."""
        logits = self.lm_head(self.hidden_states(tokens))
        loss = None
        if targets is not None:
            labels = targets.masked_fill(targets == -1, -100)
            if not (labels != -100).any():
                raise ValueError("batch has no supervised tokens")
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size), labels.reshape(-1), ignore_index=-100
            )
        return logits, loss
