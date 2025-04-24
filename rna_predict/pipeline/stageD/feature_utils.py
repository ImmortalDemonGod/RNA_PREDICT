"""
Feature and input initialization utilities for Stage D.
Extracted from run_stageD.py to reduce file size and improve cohesion.
"""

from typing import Any, Dict, Optional
import torch
from omegaconf import DictConfig


def _validate_feature_config(cfg):
    # Check if cfg has a model.stageD section
    if hasattr(cfg, "model") and hasattr(cfg.model, "stageD"):
        stage_cfg = cfg.model.stageD
    # Check if cfg has a cfg attribute with model.stageD section (for test cases)
    elif hasattr(cfg, "cfg") and hasattr(cfg.cfg, "model") and hasattr(cfg.cfg.model, "stageD"):
        stage_cfg = cfg.cfg.model.stageD
    else:
        raise ValueError("Configuration must contain model.stageD section")

    required_params = ["ref_element_size", "ref_atom_name_chars_size"]
    for param in required_params:
        if not hasattr(stage_cfg, param):
            raise ValueError(f"Configuration missing required parameter: {param}")
    return stage_cfg

def _validate_atom_metadata(atom_metadata):
    # First try to get atom_metadata directly
    if atom_metadata is None:
        # If atom_metadata is None, try to get it from input_features
        from inspect import currentframe
        frame = currentframe().f_back
        if frame and 'config' in frame.f_locals:
            config = frame.f_locals['config']
            if hasattr(config, 'input_features') and config.input_features is not None:
                if 'atom_metadata' in config.input_features:
                    atom_metadata = config.input_features['atom_metadata']

    # Now validate the atom_metadata
    if atom_metadata is None or "residue_indices" not in atom_metadata:
        raise ValueError("atom_metadata with 'residue_indices' is required for Stage D. This pipeline does not support fallback to fixed atom counts.")
    residue_indices = atom_metadata["residue_indices"]
    if isinstance(residue_indices, torch.Tensor):
        residue_indices = residue_indices.tolist()
    num_residues = max(residue_indices) + 1
    return residue_indices, num_residues

def _init_feature_tensors(batch_size, num_atoms, device, stage_cfg):
    features = {}
    features["ref_pos"] = torch.zeros(batch_size, num_atoms, 3, device=device)
    features["ref_charge"] = torch.zeros(batch_size, num_atoms, 1, device=device)
    features["ref_mask"] = torch.ones(batch_size, num_atoms, 1, device=device)
    ref_element_dim = getattr(stage_cfg, "ref_element_size", None)
    ref_atom_name_chars_dim = getattr(stage_cfg, "ref_atom_name_chars_size", None)
    features["ref_element"] = torch.zeros(batch_size, num_atoms, ref_element_dim, device=device)
    features["ref_atom_name_chars"] = torch.zeros(batch_size, num_atoms, ref_atom_name_chars_dim, device=device)
    return features

def initialize_features_from_config(
    input_feature_dict: Dict[str, Any],
    atom_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Initialize features from input_feature_dict and atom_metadata.

    Args:
        input_feature_dict: Dictionary containing input features and possibly configuration
        atom_metadata: Optional metadata about atoms, including residue indices

    Returns:
        Dictionary of initialized features
    """
    # Simply return the input_feature_dict if it's already populated
    if input_feature_dict and isinstance(input_feature_dict, dict):
        # Add any default values for missing keys if needed
        return input_feature_dict

    # If input_feature_dict is empty or None, create a minimal set of features
    features = {}

    # If we have atom_metadata, use it to determine dimensions
    if atom_metadata and isinstance(atom_metadata, dict):
        # Get device from any tensor in atom_metadata
        device = next((v.device for v in atom_metadata.values() if isinstance(v, torch.Tensor)), torch.device('cpu'))

        # Extract batch size and number of atoms if possible
        batch_size = 1  # Default batch size
        num_atoms = 0

        if 'residue_indices' in atom_metadata:
            residue_indices = atom_metadata['residue_indices']
            if isinstance(residue_indices, torch.Tensor):
                num_atoms = residue_indices.shape[0]
                # Determine number of residues
                num_residues = int(residue_indices.max().item()) + 1
            else:
                num_atoms = len(residue_indices)
                num_residues = max(residue_indices) + 1

            # Create atom_to_token_idx tensor
            if isinstance(residue_indices, torch.Tensor):
                features['atom_to_token_idx'] = residue_indices.unsqueeze(0)
            else:
                features['atom_to_token_idx'] = torch.tensor(
                    residue_indices, device=device, dtype=torch.long
                ).unsqueeze(0)

        # Add minimal required features with reasonable defaults
        if num_atoms > 0:
            # Default dimensions
            ref_element_size = 128
            ref_atom_name_chars_size = 256
            profile_size = 32

            features['ref_pos'] = torch.zeros(batch_size, num_atoms, 3, device=device)
            features['ref_charge'] = torch.zeros(batch_size, num_atoms, 1, device=device)
            features['ref_mask'] = torch.ones(batch_size, num_atoms, 1, device=device)
            features['ref_element'] = torch.zeros(batch_size, num_atoms, ref_element_size, device=device)
            features['ref_atom_name_chars'] = torch.zeros(batch_size, num_atoms, ref_atom_name_chars_size, device=device)

            if num_residues > 0:
                features['restype'] = torch.zeros(batch_size, num_residues, device=device, dtype=torch.long)
                features['profile'] = torch.zeros(batch_size, num_residues, profile_size, device=device)
                features['deletion_mean'] = torch.zeros(batch_size, num_residues, 1, device=device)

            features['ref_space_uid'] = torch.zeros(batch_size, num_atoms, device=device, dtype=torch.long)

    return features
