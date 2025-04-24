"""
Feature and input initialization utilities for Stage D.
Extracted from run_stageD.py to reduce file size and improve cohesion.
"""

from typing import Any, Dict, Optional, Union
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
    cfg_or_features: Union[Dict[str, Any], DictConfig],
    coords_or_atom_metadata: Optional[Union[torch.Tensor, Dict[str, Any]]] = None,
    atom_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Initialize features from configuration or existing features.
    Supports both calling conventions:
    1. initialize_features_from_config(input_feature_dict, atom_metadata)
    2. initialize_features_from_config(cfg, coords, atom_metadata)

    Args:
        cfg_or_features: Either a configuration object or a dictionary of features
        coords_or_atom_metadata: Either coordinates tensor or atom metadata dictionary
        atom_metadata: Optional metadata about atoms, including residue indices

    Returns:
        Dictionary of initialized features
    """
    # Handle the two different calling conventions
    if isinstance(cfg_or_features, dict) and not isinstance(coords_or_atom_metadata, torch.Tensor):
        # This is the first calling convention: (input_feature_dict, atom_metadata)
        input_feature_dict = cfg_or_features
        atom_metadata = coords_or_atom_metadata

        # Simply return the input_feature_dict if it's already populated
        if input_feature_dict and isinstance(input_feature_dict, dict):
            # Add any default values for missing keys if needed
            return input_feature_dict
    else:
        # This is the second calling convention: (cfg, coords, atom_metadata)
        cfg = cfg_or_features
        coords = coords_or_atom_metadata
        # atom_metadata is already correctly assigned

        # Extract configuration parameters
        if hasattr(cfg, "model") and hasattr(cfg.model, "stageD"):
            stage_cfg = cfg.model.stageD
        else:
            raise ValueError("ERR-STAGED-CONFIG-001: Configuration must contain model.stageD section")

        # Create an empty input_feature_dict
        input_feature_dict = {}

        # If coords is provided, use it to determine batch_size and num_atoms
        if isinstance(coords, torch.Tensor):
            batch_size = coords.shape[0]
            num_atoms = coords.shape[1]
            device = coords.device

            # Add coords to input_feature_dict
            input_feature_dict['ref_pos'] = coords

    # If input_feature_dict is empty or None, create a minimal set of features
    features = {}

    # If we have atom_metadata, use it to determine dimensions
    if atom_metadata and isinstance(atom_metadata, dict):
        # Get device from any tensor in atom_metadata or from coords
        if 'ref_pos' in input_feature_dict and isinstance(input_feature_dict['ref_pos'], torch.Tensor):
            device = input_feature_dict['ref_pos'].device
        else:
            device = next((v.device for v in atom_metadata.values() if isinstance(v, torch.Tensor)), torch.device('cpu'))

        # Extract batch size and number of atoms if possible
        batch_size = 1  # Default batch size
        if 'ref_pos' in input_feature_dict and isinstance(input_feature_dict['ref_pos'], torch.Tensor):
            batch_size = input_feature_dict['ref_pos'].shape[0]
            num_atoms = input_feature_dict['ref_pos'].shape[1]
        else:
            num_atoms = 0

        if 'residue_indices' in atom_metadata:
            residue_indices = atom_metadata['residue_indices']
            if isinstance(residue_indices, torch.Tensor):
                if num_atoms == 0:  # Only set if not already determined from coords
                    num_atoms = residue_indices.shape[0]
                # Determine number of residues
                num_residues = int(residue_indices.max().item()) + 1
            else:
                if num_atoms == 0:  # Only set if not already determined from coords
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
            # Get dimensions from config if available, otherwise use defaults
            ref_element_size = 128
            ref_atom_name_chars_size = 256
            profile_size = 32

            # Try to get dimensions from config if available
            if 'stage_cfg' in locals():
                ref_element_size = getattr(stage_cfg, "ref_element_size", ref_element_size)
                ref_atom_name_chars_size = getattr(stage_cfg, "ref_atom_name_chars_size", ref_atom_name_chars_size)
                profile_size = getattr(stage_cfg, "profile_size", profile_size)

            # Only add features if they don't already exist in input_feature_dict
            if 'ref_pos' not in features and 'ref_pos' not in input_feature_dict:
                features['ref_pos'] = torch.zeros(batch_size, num_atoms, 3, device=device)
            elif 'ref_pos' in input_feature_dict:
                features['ref_pos'] = input_feature_dict['ref_pos']

            if 'ref_charge' not in features and 'ref_charge' not in input_feature_dict:
                features['ref_charge'] = torch.zeros(batch_size, num_atoms, 1, device=device)
            elif 'ref_charge' in input_feature_dict:
                features['ref_charge'] = input_feature_dict['ref_charge']

            if 'ref_mask' not in features and 'ref_mask' not in input_feature_dict:
                features['ref_mask'] = torch.ones(batch_size, num_atoms, 1, device=device)
            elif 'ref_mask' in input_feature_dict:
                features['ref_mask'] = input_feature_dict['ref_mask']

            if 'ref_element' not in features and 'ref_element' not in input_feature_dict:
                features['ref_element'] = torch.zeros(batch_size, num_atoms, ref_element_size, device=device)
            elif 'ref_element' in input_feature_dict:
                features['ref_element'] = input_feature_dict['ref_element']

            if 'ref_atom_name_chars' not in features and 'ref_atom_name_chars' not in input_feature_dict:
                features['ref_atom_name_chars'] = torch.zeros(batch_size, num_atoms, ref_atom_name_chars_size, device=device)
            elif 'ref_atom_name_chars' in input_feature_dict:
                features['ref_atom_name_chars'] = input_feature_dict['ref_atom_name_chars']

            if num_residues > 0:
                if 'restype' not in features and 'restype' not in input_feature_dict:
                    features['restype'] = torch.zeros(batch_size, num_residues, device=device, dtype=torch.long)
                elif 'restype' in input_feature_dict:
                    features['restype'] = input_feature_dict['restype']

                if 'profile' not in features and 'profile' not in input_feature_dict:
                    features['profile'] = torch.zeros(batch_size, num_residues, profile_size, device=device)
                elif 'profile' in input_feature_dict:
                    features['profile'] = input_feature_dict['profile']

                if 'deletion_mean' not in features and 'deletion_mean' not in input_feature_dict:
                    features['deletion_mean'] = torch.zeros(batch_size, num_residues, 1, device=device)
                elif 'deletion_mean' in input_feature_dict:
                    features['deletion_mean'] = input_feature_dict['deletion_mean']

            if 'ref_space_uid' not in features and 'ref_space_uid' not in input_feature_dict:
                features['ref_space_uid'] = torch.zeros(batch_size, num_atoms, device=device, dtype=torch.long)
            elif 'ref_space_uid' in input_feature_dict:
                features['ref_space_uid'] = input_feature_dict['ref_space_uid']

    # Merge any features from input_feature_dict that weren't already added
    for key, value in input_feature_dict.items():
        if key not in features:
            features[key] = value

    return features
