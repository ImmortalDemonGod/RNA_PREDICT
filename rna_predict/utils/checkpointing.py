"""
checkpointing.py
Utilities for saving and loading partial checkpoints in RNA_PREDICT.
"""
import torch

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
    print("[DEBUG][save_trainable_checkpoint] All state_dict keys ({}):".format(len(state_dict)))
    for k in list(state_dict.keys()):
        print(f"  [state_dict] {k}")
    named_params = dict(model.named_parameters())
    print("[DEBUG][save_trainable_checkpoint] All named_parameters keys ({}):".format(len(named_params)))
    for k in list(named_params.keys()):
        print(f"  [named_param] {k} (requires_grad={named_params[k].requires_grad})")
    filtered = {}
    for k, v in state_dict.items():
        param = named_params.get(k, None)
        if param is not None and param.requires_grad:
            if filter_substrings is None or any(s in k for s in filter_substrings):
                filtered[k] = v
    print("[DEBUG][save_trainable_checkpoint] Filtered keys to save ({}):".format(len(filtered)))
    for k in list(filtered.keys()):
        print(f"  [filtered] {k}")
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
