"""
checkpointing.py
Utilities for saving and loading partial checkpoints in RNA_PREDICT.
"""
import torch
from typing import Dict

def save_trainable_checkpoint(model: torch.nn.Module, path: str, filter_substrings=None):
    """
    Save only trainable parameters (requires_grad=True) to a checkpoint file.
    Optionally filter by substrings in parameter names (e.g., 'lora', 'merger').
    Args:
        model: nn.Module
        path: str, path to save checkpoint
        filter_substrings: list of substrings (optional)
    """
    state_dict = model.state_dict()
    filtered = {}
    for k, v in state_dict.items():
        param = dict(model.named_parameters()).get(k, None)
        if param is not None and param.requires_grad:
            if filter_substrings is None or any(s in k for s in filter_substrings):
                filtered[k] = v
    torch.save(filtered, path)

# get_trainable_params utility (for test assertions)
def get_trainable_params(model: torch.nn.Module, filter_substrings=None):
    """
    Return a dict of trainable parameters, optionally filtered by substring in name.
    """
    params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            if filter_substrings is None or any(s in name for s in filter_substrings):
                params[name] = param.detach().clone()
    return params
