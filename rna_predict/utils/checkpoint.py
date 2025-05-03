"""
Utility for partial state dict loading in PyTorch models.
Designed for RNA_PREDICT: loads only matching keys from a checkpoint, logs missing/unexpected keys,
and supports strict/non-strict modes for robust checkpoint management.
"""
import logging
from collections import OrderedDict
from typing import Tuple, List, Dict, Any, Union
import torch

logger = logging.getLogger(__name__)


def partial_load_state_dict(
    model: torch.nn.Module,
    state_dict: Union[Dict[str, Any], OrderedDict[str, Any]],
    strict: bool = False
) -> Tuple[List[str], List[str]]:
    """
    Loads parameters from state_dict into model, skipping mismatched keys
    and logging information about missing/unexpected keys.

    Args:
        model: The PyTorch model to load parameters into.
        state_dict: The dictionary containing parameters to load.
        strict: If True, raise an error for missing or unexpected keys (default: False).

    Returns:
        Tuple[List[str], List[str]]: missing_keys, unexpected_keys
    """
    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    # Convert state_dict to OrderedDict if needed
    if not isinstance(state_dict, OrderedDict):
        state_dict = OrderedDict(state_dict)

    # Handle _metadata attribute which is expected by PyTorch but not part of the type
    metadata = getattr(state_dict, '_metadata', None)
    if metadata is not None:
        # PyTorch's state_dict has a _metadata attribute that's not part of the type
        # but is expected to be present for proper loading
        state_dict = state_dict.copy()
        # Use setattr to avoid mypy error with _metadata attribute
        setattr(state_dict, '_metadata', metadata)

    own_state = model.state_dict()

    # Check for unexpected keys first if strict=False
    if not strict:
        for name in state_dict:
            if name not in own_state:
                unexpected_keys.append(name)

    # Load matching keys
    for name, param in state_dict.items():
        if name in own_state:
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            try:
                own_state[name].copy_(param)
            except Exception as e:
                error_msgs.append(
                    f'While copying the parameter named "{name}", '
                    f'whose dimensions in the model are {own_state[name].size()} and '
                    f'whose dimensions in the checkpoint are {param.size()}: {str(e)}'
                )

    # Check for missing keys
    loaded_keys = set(state_dict.keys())
    for name in own_state:
        if name not in loaded_keys:
            missing_keys.append(name)

    # Handle strict mode errors
    if strict:
        unexpected_keys = [k for k in state_dict if k not in own_state]
        if unexpected_keys:
            error_msgs.insert(
                0, 'Unexpected key(s) in state_dict: {}. '.format(', '.join(f'"{k}"' for k in unexpected_keys)))
        if missing_keys:
            error_msgs.insert(
                0, 'Missing key(s) in state_dict: {}. '.format(', '.join(f'"{k}"' for k in missing_keys)))

    if error_msgs:
        raise RuntimeError(
            'Error(s) in loading state_dict for {}:\n\t{}'.format(
                model.__class__.__name__, "\n\t".join(error_msgs)
            )
        )

    # Log warnings if not strict
    if not strict:
        if unexpected_keys:
            logger.warning(f"Unexpected key(s) in state_dict: {', '.join(unexpected_keys)}")
        if missing_keys:
            logger.warning(f"Missing key(s) in state_dict: {', '.join(missing_keys)}")

    logger.info(f"Loaded {len(own_state) - len(missing_keys)} keys from checkpoint into {model.__class__.__name__}.")
    return missing_keys, unexpected_keys
