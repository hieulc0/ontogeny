"""Stable execution-facing access to canonical state comparison utilities."""

from mini_gpt.checkpoint.comparison import STATE_GROUPS, fingerprints, value_identity

__all__ = ["STATE_GROUPS", "fingerprints", "value_identity"]
