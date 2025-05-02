"""
Feature and input initialization utilities for Stage D.
Extracted from run_stageD.py to reduce file size and improve cohesion.
"""

from typing import Any, Dict, Optional, Union
import torch
from omegaconf import DictConfig

def _validate_feature_config(config, require_atom_metadata: bool = False):
    """
    Ensures all required parameters for atom-level feature initialization are present in config or nested config groups.
    Follows Hydra best practices but supports legacy fallbacks with warnings.
    If require_atom_metadata is True, raises ValueError with [ERR-STAGED-BRIDGE-002] if any are missing.
    Otherwise, raises a generic config error.
    """
    import logging
    logger = logging.getLogger(__name__)
    required_params = [
        "ref_element_size",
        "ref_atom_name_chars_size",
        "profile_size",
    ]
    # Try all Hydra best-practice locations
    locations = [
        config,
        getattr(config, "model", None),
        getattr(getattr(config, "model", None), "stageD", None),
        getattr(getattr(getattr(config, "model", None), "stageD", None), "diffusion", None),
    ]
    for param in required_params:
        found = False
        for loc in locations:
            if loc is not None and hasattr(loc, param):
                found = True
                # Warn if not found in config/model.stageD or config/model.stageD/diffusion
                if loc is not config and loc is not getattr(getattr(config, "model", None), "stageD", None) and loc is not getattr(getattr(getattr(config, "model", None), "stageD", None), "diffusion", None):
                    logger.warning(f"[HYDRA-CONF-WARN] Found {param} at nonstandard config level: {type(loc)}. Please migrate to model.stageD.diffusion.")
                break
        if not found:
            logger.debug(f"[DEBUG][feature_utils] Missing config param: {param} (require_atom_metadata={require_atom_metadata})")
            if require_atom_metadata:
                raise ValueError(f"[ERR-STAGED-BRIDGE-002] Configuration missing required parameter: {param}")
            else:
                raise ValueError(f"Configuration missing required parameter: {param}")
    return config

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
        raise ValueError("[ERR-STAGED-BRIDGE-002] atom_metadata with 'residue_indices' is required for Stage D. This pipeline does not support fallback to fixed atom counts.")
    residue_indices = atom_metadata["residue_indices"]
    if isinstance(residue_indices, torch.Tensor):
        residue_indices = residue_indices.tolist()
    num_residues = max(residue_indices) + 1
    return residue_indices, num_residues

def _init_feature_tensors(batch_size, num_atoms, device, stage_cfg, debug_logging=False):
    features = {}
    features["ref_pos"] = torch.zeros(batch_size, num_atoms, 3, device=device)
    features["ref_charge"] = torch.zeros(batch_size, num_atoms, 1, device=device)
    features["ref_mask"] = torch.ones(batch_size, num_atoms, 1, device=device)
    # Robust config extraction for feature dimensions
    def extract_dim(cfg, key, default=None):
        import logging
        logger = logging.getLogger(__name__)
        locations = [
            cfg,
            getattr(cfg, "model", None),
            getattr(getattr(cfg, "model", None), "stageD", None),
            getattr(getattr(getattr(cfg, "model", None), "stageD", None), "diffusion", None),
        ]
        for loc in locations:
            if loc is None:
                continue
            if isinstance(loc, dict) and key in loc:
                value = loc[key]
            elif hasattr(loc, key):
                value = getattr(loc, key)
            else:
                continue
            if value is not None and debug_logging:
                logger.debug(f"[feature_utils] {key}={value} found in {type(loc)}")
            return value
        if debug_logging:
            logger.debug(f"[feature_utils] {key} not found, using default: {default}")
        return default

    # Get dimensions from config, but enforce minimum sizes required by the model
    import logging
    logger = logging.getLogger(__name__)

    # The model expects these minimum dimensions
    min_ref_element_dim = 128
    min_ref_atom_name_chars_dim = 256
    min_profile_dim = 32

    # Get dimensions with defaults that match the model requirements
    ref_element_dim = extract_dim(stage_cfg, "ref_element_size", min_ref_element_dim)
    ref_atom_name_chars_dim = extract_dim(stage_cfg, "ref_atom_name_chars_size", min_ref_atom_name_chars_dim)
    profile_dim = extract_dim(stage_cfg, "profile_size", min_profile_dim)

    # Ensure dimensions are integers and meet minimum requirements
    if not isinstance(ref_element_dim, int) or ref_element_dim < min_ref_element_dim:
        if not isinstance(ref_element_dim, int):
            logger.warning(f"[HYDRA-CONF-FIX] ref_element_size is not an integer. Using default {min_ref_element_dim}.")
        else:
            logger.warning(f"[HYDRA-CONF-FIX] ref_element_size ({ref_element_dim}) is smaller than required by the model ({min_ref_element_dim}). Using {min_ref_element_dim} instead.")
        ref_element_dim = min_ref_element_dim

    if not isinstance(ref_atom_name_chars_dim, int) or ref_atom_name_chars_dim < min_ref_atom_name_chars_dim:
        if not isinstance(ref_atom_name_chars_dim, int):
            logger.warning(f"[HYDRA-CONF-FIX] ref_atom_name_chars_size is not an integer. Using default {min_ref_atom_name_chars_dim}.")
        else:
            logger.warning(f"[HYDRA-CONF-FIX] ref_atom_name_chars_size ({ref_atom_name_chars_dim}) is smaller than required by the model ({min_ref_atom_name_chars_dim}). Using {min_ref_atom_name_chars_dim} instead.")
        ref_atom_name_chars_dim = min_ref_atom_name_chars_dim

    if not isinstance(profile_dim, int) or profile_dim < min_profile_dim:
        if not isinstance(profile_dim, int):
            logger.warning(f"[HYDRA-CONF-FIX] profile_size is not an integer. Using default {min_profile_dim}.")
        else:
            logger.warning(f"[HYDRA-CONF-FIX] profile_size ({profile_dim}) is smaller than required by the model ({min_profile_dim}). Using {min_profile_dim} instead.")
        profile_dim = min_profile_dim

    features["ref_element"] = torch.zeros(batch_size, num_atoms, ref_element_dim, device=device)
    features["ref_atom_name_chars"] = torch.zeros(batch_size, num_atoms, ref_atom_name_chars_dim, device=device)
    features["profile"] = torch.zeros(batch_size, num_atoms, profile_dim, device=device)
    return features

def initialize_features_from_config(
    cfg_or_features: Union[Dict[str, Any], DictConfig],
    coords_or_atom_metadata: Optional[Union[torch.Tensor, Dict[str, Any]]] = None,
    atom_metadata: Optional[Dict[str, Any]] = None,
    debug_logging=False,
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
    # Defensive: always initialize stage_cfg_from_outer
    stage_cfg_from_outer = None
    # Handle the two different calling conventions
    if isinstance(cfg_or_features, dict) and not isinstance(coords_or_atom_metadata, torch.Tensor):
        # This is the first calling convention: (input_feature_dict, atom_metadata)
        input_feature_dict = cfg_or_features
        atom_metadata = coords_or_atom_metadata

        # Try to extract config if present in input_feature_dict
        if 'model' in input_feature_dict and isinstance(input_feature_dict['model'], dict) and 'stageD' in input_feature_dict['model']:
            stage_cfg_from_outer = input_feature_dict['model']['stageD']
        elif 'stageD' in input_feature_dict:
            stage_cfg_from_outer = input_feature_dict['stageD']
        # Simply return the input_feature_dict if it's already populated
        if input_feature_dict and isinstance(input_feature_dict, dict):
            return input_feature_dict
    else:
        # This is the second calling convention: (cfg, coords, atom_metadata)
        cfg = cfg_or_features
        coords = coords_or_atom_metadata
        # atom_metadata is already correctly assigned

        # Extract configuration parameters
        # Handle both DictConfig and dict types
        if isinstance(cfg, dict):
            # For dictionary configs, look for nested 'model' key
            if 'model' in cfg and isinstance(cfg['model'], dict) and 'stageD' in cfg['model']:
                stage_cfg_from_outer = cfg['model']['stageD']
            else:
                # If no nested structure, use the dict directly
                stage_cfg_from_outer = cfg
        else:
            # For OmegaConf/DictConfig objects
            if hasattr(cfg, "model") and hasattr(cfg.model, "stageD"):
                stage_cfg_from_outer = cfg.model.stageD
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
        num_atoms = 0   # Default number of atoms
        num_residues = 0  # Default number of residues

        if 'ref_pos' in input_feature_dict and isinstance(input_feature_dict['ref_pos'], torch.Tensor):
            batch_size = input_feature_dict['ref_pos'].shape[0]
            num_atoms = input_feature_dict['ref_pos'].shape[1]

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
            # The model expects these minimum dimensions
            min_ref_element_dim = 128
            min_ref_atom_name_chars_dim = 256
            min_profile_dim = 32

            # Default to minimum required dimensions
            ref_element_size = min_ref_element_dim
            ref_atom_name_chars_size = min_ref_atom_name_chars_dim
            profile_size = min_profile_dim

            # Import logging for warnings
            import logging
            logger = logging.getLogger(__name__)

            # stage_cfg_from_outer has already been resolved above – reuse it here

            if stage_cfg_from_outer is not None:
                # Handle both dict and object access patterns
                if isinstance(stage_cfg_from_outer, dict):
                    # Dictionary access
                    if 'ref_element_size' in stage_cfg_from_outer:
                        ref_element_size = stage_cfg_from_outer['ref_element_size']
                    if 'ref_atom_name_chars_size' in stage_cfg_from_outer:
                        ref_atom_name_chars_size = stage_cfg_from_outer['ref_atom_name_chars_size']
                    if 'profile_size' in stage_cfg_from_outer:
                        profile_size = stage_cfg_from_outer['profile_size']
                else:
                    # Object attribute access
                    try:
                        if hasattr(stage_cfg_from_outer, "ref_element_size"):
                            ref_element_size = stage_cfg_from_outer.ref_element_size
                        if hasattr(stage_cfg_from_outer, "ref_atom_name_chars_size"):
                            ref_atom_name_chars_size = stage_cfg_from_outer.ref_atom_name_chars_size
                        if hasattr(stage_cfg_from_outer, "profile_size"):
                            profile_size = stage_cfg_from_outer.profile_size
                    except Exception:
                        pass  # Defensive: ignore config attribute errors

            # Ensure dimensions meet minimum requirements
            if not isinstance(ref_element_size, int) or ref_element_size < min_ref_element_dim:
                logger.warning(f"[HYDRA-CONF-FIX] ref_element_size ({ref_element_size}) is smaller than required by the model ({min_ref_element_dim}). Using {min_ref_element_dim} instead.")
                ref_element_size = min_ref_element_dim

            if not isinstance(ref_atom_name_chars_size, int) or ref_atom_name_chars_size < min_ref_atom_name_chars_dim:
                logger.warning(f"[HYDRA-CONF-FIX] ref_atom_name_chars_size ({ref_atom_name_chars_size}) is smaller than required by the model ({min_ref_atom_name_chars_dim}). Using {min_ref_atom_name_chars_dim} instead.")
                ref_atom_name_chars_size = min_ref_atom_name_chars_dim

            if not isinstance(profile_size, int) or profile_size < min_profile_dim:
                logger.warning(f"[HYDRA-CONF-FIX] profile_size ({profile_size}) is smaller than required by the model ({min_profile_dim}). Using {min_profile_dim} instead.")
                profile_size = min_profile_dim

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

def extract_atom_features(input_feature_dict, encoder_input_feature_config, debug_logging=False):
    features = []
    feature_names = []
    batch_size = None
    num_atoms = None
    # Defensive: Ensure all expected features are present and have correct shape
    for key, expected_dim in encoder_input_feature_config.items():
        if key not in input_feature_dict:
            raise ValueError(f"[ERR-STAGED-FEATURES-001] Missing feature '{key}' in input_feature_dict. Check Hydra config and bridging logic.")
        value = input_feature_dict[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"[ERR-STAGED-FEATURES-002] Feature '{key}' is not a tensor (got {type(value)}). Check bridging and config.")
        if value.ndim != 3:
            raise ValueError(f"[ERR-STAGED-FEATURES-003] Feature '{key}' does not have 3 dims (got {value.shape}). Should be [batch, num_atoms, feature_dim].")
        if value.shape[2] != expected_dim:
            raise ValueError(f"[ERR-STAGED-FEATURES-004] Feature '{key}' last dim {value.shape[2]} != expected {expected_dim} from config. Check Hydra config consistency.")
        if batch_size is None:
            batch_size = value.shape[0]
            num_atoms = value.shape[1]
        else:
            if value.shape[0] != batch_size:
                raise ValueError(f"[ERR-STAGED-FEATURES-005] Feature '{key}' batch size {value.shape[0]} != expected {batch_size}.")
            if value.shape[1] != num_atoms:
                raise ValueError(f"[ERR-STAGED-FEATURES-006] Feature '{key}' num_atoms {value.shape[1]} != expected {num_atoms}.")
        features.append(value)
        feature_names.append(key)
    # Concatenate features
    concat = torch.cat(features, dim=2)
    if debug_logging:
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"[extract_atom_features] Defensive check: concatenated feature shape: {concat.shape}")
    return concat
