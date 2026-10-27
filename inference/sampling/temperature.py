"""Module implementation."""
import torch
import torch.nn.functional as F


def apply_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Apply temperature scaling to logits.
    """
    temperature = max(float(temperature), 1e-8)
    return F.softmax(logits / temperature, dim=-1)
