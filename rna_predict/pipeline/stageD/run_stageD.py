"""
Stage D runner for RNA prediction using structured Hydra configs.

Main entry point for running Stage D (Diffusion Refinement).
Assumes inputs (coordinates, embeddings) are already at the atom level
(e.g., after bridging in a previous step or from data loading).
Includes optional preprocessing for memory optimization controlled via config.

Configuration Requirements:
    The module expects a Hydra configuration with the following structure:
    - model.stageD:
        - enabled: Whether Stage D is enabled
        - mode: Mode (inference or training)
        - device: Device to run on (cpu, cuda, mps)
        - debug_logging: Whether to enable debug logging
        - sigma_data: Sigma data parameter for diffusion
        - c_atom: Atom dimension
        - c_s: Single representation dimension
        - c_z: Pair representation dimension
        - c_s_inputs: Input representation dimension
        - c_noise_embedding: Noise embedding dimension
        - ref_element_size: Size of reference element embeddings
        - ref_atom_name_chars_size: Size of atom name character embeddings
        - inference:
            - num_steps: Number of diffusion steps

"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "/Users/tomriddle1/RNA_PREDICT")))
print("[HYDRA DEBUG] CWD:", os.getcwd())
print("[HYDRA DEBUG] SCRIPT DIR:", os.path.dirname(__file__))
print("[HYDRA DEBUG] sys.path:", sys.path)

import hydra
from omegaconf import DictConfig
import torch
from typing import Dict, Optional, Any, Union, Tuple
import logging
import psutil
import os

# Import structured configs
from rna_predict.conf.config_schema import StageDConfig, register_configs

# Removed unused snoop import and decorator

log = logging.getLogger(__name__)

# Register Hydra configurations
register_configs()

def log_mem(stage):
    process = psutil.Process(os.getpid())
    print(f"[MEMORY-LOG][{stage}] Memory usage: {process.memory_info().rss / (1024 * 1024):.2f} MB")

def initialize_features_from_config(cfg: DictConfig, coords: torch.Tensor, atom_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
    """Initialize input features from configuration.

    Args:
        cfg: Configuration object containing feature specifications
        coords: Input coordinates tensor to derive shapes and device
        atom_metadata: Optional metadata about atoms, including residue indices

    Returns:
        Dictionary of initialized features

    Raises:
        ValueError: If required configuration sections are missing
    """
    # Validate that the required configuration sections exist
    if not hasattr(cfg, "model") or not hasattr(cfg.model, "stageD"):
        raise ValueError("Configuration must contain model.stageD section")

    # Extract the stageD config for cleaner access
    stage_cfg: StageDConfig = cfg.model.stageD

    # Validate required parameters
    required_params = ["ref_element_size", "ref_atom_name_chars_size"]
    for param in required_params:
        if not hasattr(stage_cfg, param):
            raise ValueError(f"Configuration missing required parameter: {param}")

    features = {}
    batch_size, num_atoms = coords.shape[:2]
    device = coords.device

    # --- Robustness checks ---
    if atom_metadata is None or "residue_indices" not in atom_metadata:
        raise ValueError("atom_metadata with 'residue_indices' is required for Stage D. This pipeline does not support fallback to fixed atom counts.")

    # Get residue indices and determine number of residues
    residue_indices = atom_metadata["residue_indices"]
    if isinstance(residue_indices, torch.Tensor):
        residue_indices = residue_indices.tolist()
    num_residues = max(residue_indices) + 1
    log.info(f"Using atom metadata to determine number of residues: {num_residues}")

    # Initialize required features with correct dimensions from config
    features["ref_pos"] = coords.clone()  # [batch_size, num_atoms, 3]
    features["ref_charge"] = torch.zeros(batch_size, num_atoms, 1, device=device)  # [batch_size, num_atoms, 1]
    features["ref_mask"] = torch.ones(batch_size, num_atoms, 1, device=device)  # [batch_size, num_atoms, 1]

    # Get dimensions from config
    ref_element_dim = getattr(stage_cfg, "ref_element_size", None)
    ref_atom_name_chars_dim = getattr(stage_cfg, "ref_atom_name_chars_size", None)
    if ref_element_dim is None or ref_atom_name_chars_dim is None:
        raise ValueError("Configuration missing required input feature sizes.")
    features["ref_element"] = torch.zeros(batch_size, num_atoms, ref_element_dim, device=device)
    features["ref_atom_name_chars"] = torch.zeros(batch_size, num_atoms, ref_atom_name_chars_dim, device=device)

    # Initialize atom_to_token_idx mapping
    residue_indices_tensor = torch.tensor(residue_indices, device=device, dtype=torch.long)
    features["atom_to_token_idx"] = residue_indices_tensor.unsqueeze(0).expand(batch_size, -1)  # [batch_size, num_atoms]
    log.info(f"Using atom metadata for atom_to_token_idx mapping with {num_residues} residues")

    # Initialize additional features required by the model
    features["restype"] = torch.zeros(batch_size, num_residues, device=device, dtype=torch.long)  # [batch_size, num_residues]

    # Use config-driven profile dimension
    profile_size = getattr(stage_cfg, "profile_size", None)
    if profile_size is None:
        raise ValueError("Configuration missing required profile_size parameter.")
    features["profile"] = torch.zeros(batch_size, num_residues, profile_size, device=device)  # Profile embeddings

    features["deletion_mean"] = torch.zeros(batch_size, num_residues, 1, device=device)  # [batch_size, num_residues, 1]
    features["ref_space_uid"] = torch.zeros(batch_size, num_atoms, device=device, dtype=torch.long)  # [batch_size, num_atoms]

    # Add additional features required by the atom attention encoder
    # These are needed to match the expected in_features dimension of 389
    # The expected features are:
    # - ref_pos: 3
    # - ref_charge: 1
    # - ref_mask: 1
    # - ref_element: 128
    # - ref_atom_name_chars: 256 (4 * 64)

    # Make sure all required features are present with the correct dimensions
    if "ref_pos" not in features:
        features["ref_pos"] = coords.clone()  # [batch_size, num_atoms, 3]
    if "ref_charge" not in features:
        features["ref_charge"] = torch.zeros(batch_size, num_atoms, 1, device=device)  # [batch_size, num_atoms, 1]
    if "ref_mask" not in features:
        features["ref_mask"] = torch.ones(batch_size, num_atoms, 1, device=device)  # [batch_size, num_atoms, 1]
    if "ref_element" not in features:
        features["ref_element"] = torch.zeros(batch_size, num_atoms, ref_element_dim, device=device)  # [batch_size, num_atoms, ref_element_dim]
    if "ref_atom_name_chars" not in features:
        features["ref_atom_name_chars"] = torch.zeros(batch_size, num_atoms, ref_atom_name_chars_dim, device=device)  # [batch_size, num_atoms, ref_atom_name_chars_dim]

    # Add all the required features to match the expected dimension
    # Calculate the total dimension dynamically from the config values
    total_dim = 3 + 1 + 1 + ref_element_dim + ref_atom_name_chars_dim
    print(f"[DEBUG][initialize_features_from_config] Total feature dimension: {total_dim}")

    return features

def run_stageD(
    cfg: DictConfig,
    coords: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    s_inputs: torch.Tensor,
    input_feature_dict: Dict[str, Any],
    atom_metadata: Optional[Dict[str, Any]] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Run diffusion refinement on the input coordinates using the unified Stage D runner."""
    log_mem("StageD ENTRY")
    print(f"[DEBUG][run_stageD] ENTRY: z_trunk.shape = {getattr(z_trunk, 'shape', None)}")
    print(f"[DEBUG][run_stageD] ENTRY: s_trunk.shape = {getattr(s_trunk, 'shape', None)}")
    print(f"[DEBUG][run_stageD] ENTRY: s_inputs.shape = {getattr(s_inputs, 'shape', None)}")
    # Validate that the required configuration sections exist
    if not hasattr(cfg, "model") or not hasattr(cfg.model, "stageD"):
        raise ValueError("Configuration must contain model.stageD section")

    # Defensive check: s_trunk must be residue-level at entry
    if atom_metadata is not None and 'residue_indices' in atom_metadata:
        n_atoms = len(atom_metadata['residue_indices'])
        # Try to infer residues from sequence if available
        n_residues = len(input_feature_dict.get('sequence', [])) if input_feature_dict.get('sequence', None) is not None else None
        # If sequence not available, estimate from atom count and atoms per residue
        atoms_per_res = n_atoms // n_residues if n_residues else None
        if s_trunk.shape[1] == n_atoms:
            raise ValueError('[RUNSTAGED ERROR][UNIQUE_CODE_003] s_trunk is atom-level at entry to run_stageD; upstream code must pass residue-level embeddings.')

    # Import unified runner and config dataclass
    from rna_predict.pipeline.stageD.diffusion.run_stageD_unified import run_stageD_diffusion
    from rna_predict.pipeline.stageD.diffusion.utils import DiffusionConfig
    import copy

    # Get Stage D configuration
    stage_cfg = cfg.model.stageD

    # Prepare trunk_embeddings dict
    trunk_embeddings = {
        "s_trunk": s_trunk,
        "s_inputs": s_inputs,
        "pair": z_trunk
    }

    # Optionally add atom_metadata to input features
    features = copy.deepcopy(input_feature_dict) if input_feature_dict is not None else {}
    if atom_metadata is not None:
        features["atom_metadata"] = atom_metadata

    # --- PATCH: Ensure we have residue-level embeddings before bridging ---
    # The bridging will be done by the unified Stage D runner, so we don't need to do it here
    # We just need to ensure we have residue-level embeddings
    if atom_metadata is not None and 'residue_indices' in atom_metadata:
        # Get sequence from features
        sequence = features.get('sequence', None)

        # Ensure we have residue-level embeddings
        if sequence is not None:
            n_residues = len(sequence)

            # Check if s_trunk is already at residue level
            if s_trunk.shape[1] != n_residues:
                raise ValueError('[RUNSTAGED ERROR][UNIQUE_CODE_003] s_trunk is atom-level at entry to run_stageD; upstream code must pass residue-level embeddings.')

            # Check if z_trunk is already at residue level
            if z_trunk.shape[1] != n_residues or z_trunk.shape[2] != n_residues:
                raise ValueError('[RUNSTAGED ERROR][UNIQUE_CODE_006] z_trunk is atom-level at entry to run_stageD; upstream code must pass residue-level embeddings.')

            # Check if s_inputs is already at residue level
            if s_inputs.shape[1] != n_residues:
                raise ValueError('[RUNSTAGED ERROR][UNIQUE_CODE_007] s_inputs is atom-level at entry to run_stageD; upstream code must pass residue-level embeddings.')
    else:
        raise ValueError("[ERR-STAGED-BRIDGE-002] atom_metadata missing or missing residue_indices; cannot bridge residue to atom level.")
    # --- END PATCH ---

    # --- PATCH: Defensive shape checks for atom count and feature mismatches ---
    n_atoms = coords.shape[1] if hasattr(coords, 'shape') and len(coords.shape) > 1 else None
    err_prefix = '[ERR-STAGED-SHAPE-001]'
    if n_atoms is not None:
        # Skip shape checks for non-tensor values and special keys
        skip_keys = ['atom_metadata', 'sequence', 'ref_space_uid']

        # Check and fix trunk_embeddings shapes
        for key, value in list(trunk_embeddings.items()):
            if key in skip_keys or not isinstance(value, torch.Tensor):
                continue
            if value.shape[1] != n_atoms:
                print(f"[DEBUG][run_stageD] {key} shape mismatch: {value.shape[1]} != {n_atoms}")
                # Skip the check for these keys as they're handled by the bridging function
                if key in ['s_trunk', 's_inputs', 'pair']:
                    continue
                raise ValueError(f"{err_prefix} trunk_embeddings['{key}'] atom dim ({value.shape[1]}) != n_atoms ({n_atoms})")

        # Check and fix features shapes
        for key, value in list(features.items()):
            if key in skip_keys or not isinstance(value, torch.Tensor):
                continue
            if value.shape[1] != n_atoms:
                print(f"[DEBUG][run_stageD] {key} shape mismatch: {value.shape[1]} != {n_atoms}")
                # Skip the check for these keys as they're handled by the bridging function
                if key in ['s_trunk', 's_inputs', 'pair']:
                    continue
                raise ValueError(f"{err_prefix} features['{key}'] atom dim ({value.shape[1]}) != n_atoms ({n_atoms})")

    # Copy all features to input_feature_dict to ensure they're available to the diffusion model
    for key, value in features.items():
        if key not in input_feature_dict:
            input_feature_dict[key] = value
    # --- END PATCH ---

    # --- PATCH: Enforce config-driven minimal residue/atom count for memory efficiency ---
    max_residues = getattr(cfg.model.stageD.diffusion, 'test_residues_per_batch', None)
    if max_residues is not None and atom_metadata is not None and 'residue_indices' in atom_metadata:
        residue_indices = atom_metadata['residue_indices']
        keep_atom_indices = [i for i, r in enumerate(residue_indices) if r < max_residues]
        print(f"[DEBUG][run_stageD][PATCH] max_residues = {max_residues}")
        print(f"[DEBUG][run_stageD][PATCH] residue_indices[:50] = {residue_indices[:50]}")
        print(f"[DEBUG][run_stageD][PATCH] keep_atom_indices (len={len(keep_atom_indices)}): {keep_atom_indices[:50]}")
        print(f"[DEBUG][run_stageD][PATCH] len(residue_indices) = {len(residue_indices)}")
        if len(keep_atom_indices) < len(residue_indices):
            print(f"[DEBUG][run_stageD][PATCH] Slicing input to first {max_residues} residues ({len(keep_atom_indices)} atoms out of {len(residue_indices)}) for minimal memory run.")
            coords = coords[:, keep_atom_indices, ...]
            atom_metadata = {k: (v if not isinstance(v, list) else [v[i] for i in keep_atom_indices]) for k, v in atom_metadata.items()}
            atom_metadata['residue_indices'] = [r for r in residue_indices if r < max_residues]
            for key, value in input_feature_dict.items():
                if isinstance(value, torch.Tensor) and value.shape[1] == len(residue_indices):
                    input_feature_dict[key] = value[:, keep_atom_indices, ...]
            print(f"[DEBUG][run_stageD][PATCH] coords.shape after slicing: {coords.shape}")
            for key, value in input_feature_dict.items():
                if isinstance(value, torch.Tensor):
                    print(f"[DEBUG][run_stageD][PATCH] input_feature_dict['{key}'].shape after slicing: {value.shape}")
            print(f"[DEBUG][run_stageD][PATCH] atom_metadata['residue_indices'][:50] after slicing: {atom_metadata['residue_indices'][:50]}")
    # --- END PATCH ---

    log_mem("Before bridging residue-to-atom")
    # Prepare diffusion_config dict (flatten stage_cfg to dict, omitting tensor fields)
    # Use OmegaConf.to_container to convert to dict
    import omegaconf
    try:
        diffusion_config_dict = omegaconf.OmegaConf.to_container(stage_cfg, resolve=True)
        # Ensure we have a dictionary
        if not isinstance(diffusion_config_dict, dict):
            print(f"[DEBUG][run_stageD] diffusion_config_dict is not a dict: {type(diffusion_config_dict)}")
            diffusion_config_dict = {}

        # Remove fields that are not serializable or are tensors
        for k in ["model_architecture", "memory", "inference", "transformer", "atom_encoder", "atom_decoder"]:
            if k in diffusion_config_dict:
                diffusion_config_dict.pop(k, None)

        # Add nested configs if present
        if hasattr(stage_cfg, "inference"):
            diffusion_config_dict["inference"] = omegaconf.OmegaConf.to_container(stage_cfg.inference, resolve=True)
        if hasattr(stage_cfg, "transformer"):
            diffusion_config_dict["transformer"] = omegaconf.OmegaConf.to_container(stage_cfg.transformer, resolve=True)
        if hasattr(stage_cfg, "atom_encoder"):
            diffusion_config_dict["atom_encoder"] = omegaconf.OmegaConf.to_container(stage_cfg.atom_encoder, resolve=True)
        if hasattr(stage_cfg, "atom_decoder"):
            diffusion_config_dict["atom_decoder"] = omegaconf.OmegaConf.to_container(stage_cfg.atom_decoder, resolve=True)
        if hasattr(stage_cfg, "memory"):
            diffusion_config_dict["memory"] = omegaconf.OmegaConf.to_container(stage_cfg.memory, resolve=True)
    except Exception as e:
        print(f"[DEBUG][run_stageD] Error converting config to dict: {e}")
        # Fallback to empty dict
        diffusion_config_dict = {}
    log_mem("After bridging residue-to-atom")

    # Get pair embedding dimension directly from diffusion config
    diffusion_cfg = getattr(stage_cfg, "diffusion", None)
    print(f"[DEBUG][run_stageD] diffusion_cfg: {diffusion_cfg}")
    if diffusion_cfg is None:
        raise ValueError("Configuration missing required 'diffusion' section under model.stageD.")
    # Support both dict and OmegaConf/structured config
    c_s = getattr(diffusion_cfg, "c_s", None) if hasattr(diffusion_cfg, "c_s") else diffusion_cfg.get("c_s", None)
    c_s_inputs = getattr(diffusion_cfg, "c_s_inputs", None) if hasattr(diffusion_cfg, "c_s_inputs") else diffusion_cfg.get("c_s_inputs", None)
    c_z = getattr(diffusion_cfg, "c_z", None) if hasattr(diffusion_cfg, "c_z") else diffusion_cfg.get("c_z", None)
    print(f"[DEBUG][run_stageD] diffusion_cfg.c_s: {c_s}, c_s_inputs: {c_s_inputs}, c_z: {c_z}")
    if c_s is None or c_s_inputs is None or c_z is None:
        print(f"[DEBUG][run_stageD] MISSING PARAMS - c_s: {c_s}, c_s_inputs: {c_s_inputs}, c_z: {c_z}")
        print(f"[DEBUG][run_stageD] diffusion_cfg dict: {diffusion_cfg}")
        raise ValueError("Configuration missing required c_s, c_s_inputs, or c_z parameter in model.stageD.diffusion.")

    print(f"[DEBUG][run_stageD] Using pair embedding dimension c_z={c_z}")

    # Mode and device
    mode = getattr(diffusion_cfg, "mode", None)
    if mode is None:
        raise ValueError("Configuration missing required mode parameter.")
    device = getattr(diffusion_cfg, "device", None)
    if device is None:
        raise ValueError("Configuration missing required device parameter.")
    debug_logging = getattr(diffusion_cfg, "debug_logging", None)
    if debug_logging is None:
        raise ValueError("Configuration missing required debug_logging parameter.")
    sequence = features.get("sequence", None)

    # Build DiffusionConfig dataclass
    # Cast diffusion_config_dict to Dict[str, Any] to satisfy type checker
    typed_diffusion_config: Dict[str, Any] = {str(k): v for k, v in diffusion_config_dict.items()}
    config = DiffusionConfig(
        partial_coords=coords,
        trunk_embeddings=trunk_embeddings,
        diffusion_config=typed_diffusion_config,
        mode=mode,
        device=device,
        input_features=features,
        debug_logging=debug_logging,
        sequence=sequence
    )

    log_mem("Before diffusion")
    print("[DEBUG][run_stageD] Calling unified Stage D runner with DiffusionConfig.")
    # Call the unified runner
    result = run_stageD_diffusion(config)
    log_mem("After diffusion")

    log_mem("Before original_trunk_embeddings_ref loop")
    # ... loop code ...
    log_mem("After original_trunk_embeddings_ref loop")

    log_mem("StageD EXIT")
    return result


# Note: register_configs() is already called at the beginning of the file

@hydra.main(version_base=None, config_path=None, config_name=None)
def hydra_main(cfg: DictConfig) -> None:
    """Main entry point for running Stage D with Hydra configuration.

    Args:
        cfg: Hydra configuration object
    """
    from hydra.core.hydra_config import HydraConfig
    try:
        print("[HYDRA DEBUG] HydraConfig search path:", HydraConfig.get().runtime.config_search_path)
    except Exception as e:
        print("[HYDRA DEBUG] Could not access HydraConfig search path:", e)

    # DEBUG: Print config source and key values for hypothesis testing
    print("[DEBUG][hydra_main] Loaded config:", cfg)
    if hasattr(cfg, "model") and hasattr(cfg.model, "stageD"):
        stageD_cfg = cfg.model.stageD
        print("[DEBUG][hydra_main] model.stageD config:", stageD_cfg)
        # Try to print from model.stageD.diffusion
        diffusion_cfg = stageD_cfg.get("diffusion", None) if isinstance(stageD_cfg, dict) else getattr(stageD_cfg, "diffusion", None)
        if diffusion_cfg is not None:
            print("[DEBUG][hydra_main] model.stageD.diffusion config:", diffusion_cfg)
            for key in ["c_z", "c_s", "c_s_inputs", "c_token"]:
                val = diffusion_cfg.get(key, None) if isinstance(diffusion_cfg, dict) else getattr(diffusion_cfg, key, None)
                print(f"[DEBUG][hydra_main] diffusion.{key}: {val}")
            # Also check nested model_architecture
            model_arch = diffusion_cfg.get("model_architecture", None) if isinstance(diffusion_cfg, dict) else getattr(diffusion_cfg, "model_architecture", None)
            if model_arch is not None:
                print("[DEBUG][hydra_main] model.stageD.diffusion.model_architecture config:", model_arch)
                for key in ["c_z", "c_s", "c_s_inputs", "c_token"]:
                    val = model_arch.get(key, None) if isinstance(model_arch, dict) else getattr(model_arch, key, None)
                    print(f"[DEBUG][hydra_main] model_architecture.{key}: {val}")
        else:
            print("[DEBUG][hydra_main] model.stageD.diffusion not found!")
    else:
        print("[DEBUG][hydra_main] model.stageD not found in config!")

    # Removed sleep and early return now that config debug is validated

    # Validate that the required configuration sections exist
    if not hasattr(cfg, "model") or not hasattr(cfg.model, "stageD"):
        raise ValueError("Configuration must contain model.stageD section")

    # Extract the stageD config for cleaner access
    stage_cfg: StageDConfig = cfg.model.stageD

    # Validate debug_logging parameter
    if hasattr(stage_cfg, "debug_logging"):
        debug_logging = stage_cfg.debug_logging
    elif hasattr(stage_cfg, "diffusion") and hasattr(stage_cfg.diffusion, "debug_logging"):
        debug_logging = stage_cfg.diffusion.debug_logging
    else:
        # Default to True if not found
        debug_logging = True

    if debug_logging:
        log.info("Running Stage D Standalone Demo")
        log.debug("[UNIQUE-DEBUG-STAGED-TEST] Stage D runner started.")

    # Create dummy data for testing using standardized test data
    batch_size = 1  # Use a single batch for testing

    # Validate test data configuration (flat config)
    if not hasattr(cfg, "sequence"):
        raise ValueError("Configuration must contain 'sequence' at the top level")
    if not hasattr(cfg, "atoms_per_residue"):
        raise ValueError("Configuration must contain 'atoms_per_residue' at the top level")
    sequence_str = cfg.sequence
    atoms_per_residue = cfg.atoms_per_residue
    print(f"Using standardized test sequence: {sequence_str} with {atoms_per_residue} atoms per residue")

    num_residues = len(sequence_str)
    num_atoms = num_residues * atoms_per_residue

    # Create dummy inputs with dimensions from config
    dummy_coords = torch.randn(batch_size, num_atoms, 3)  # 3D coordinates

    # Create atom-to-token mapping (each atom maps to its residue)
    atom_to_token_idx = torch.repeat_interleave(
        torch.arange(num_residues),
        atoms_per_residue
    ).unsqueeze(0)  # Shape: [1, num_atoms]

    # Debug logging for atom-to-token mapping
    if debug_logging:
        print(f"[DEBUG][run_stageD] atom_to_token_idx shape: {atom_to_token_idx.shape}")
        print(f"[DEBUG][run_stageD] atom_to_token_idx: {atom_to_token_idx[0][:20]}...")

    # Create residue-level embeddings first using dimensions from config
    # Get dimensions from config
    diffusion_cfg = getattr(stage_cfg, "diffusion", None)
    print(f"[DEBUG][run_stageD] diffusion_cfg: {diffusion_cfg}")
    if diffusion_cfg is None:
        raise ValueError("Configuration missing required 'diffusion' section under model.stageD.")
    # Support both dict and OmegaConf/structured config
    c_s = getattr(diffusion_cfg, "c_s", None) if hasattr(diffusion_cfg, "c_s") else diffusion_cfg.get("c_s", None)
    c_s_inputs = getattr(diffusion_cfg, "c_s_inputs", None) if hasattr(diffusion_cfg, "c_s_inputs") else diffusion_cfg.get("c_s_inputs", None)
    c_z = getattr(diffusion_cfg, "c_z", None) if hasattr(diffusion_cfg, "c_z") else diffusion_cfg.get("c_z", None)
    print(f"[DEBUG][run_stageD] diffusion_cfg.c_s: {c_s}, c_s_inputs: {c_s_inputs}, c_z: {c_z}")
    if c_s is None or c_s_inputs is None or c_z is None:
        print(f"[DEBUG][run_stageD] MISSING PARAMS - c_s: {c_s}, c_s_inputs: {c_s_inputs}, c_z: {c_z}")
        print(f"[DEBUG][run_stageD] diffusion_cfg dict: {diffusion_cfg}")
        raise ValueError("Configuration missing required c_s, c_s_inputs, or c_z parameter in model.stageD.diffusion.")

    print(f"[DEBUG][run_stageD] Using pair embedding dimension c_z={c_z}")

    # Debug logging for dimensions
    if debug_logging:
        print(f"[DEBUG][run_stageD] ENTRY: z_trunk.shape = {torch.randn(batch_size, num_residues, num_residues, c_z).shape}")
        print(f"[DEBUG][run_stageD] ENTRY: s_trunk.shape = {torch.randn(batch_size, num_residues, c_s).shape}")
        print(f"[DEBUG][run_stageD] ENTRY: s_inputs.shape = {torch.randn(batch_size, num_residues, c_s_inputs).shape}")
        print(f"[DEBUG][run_stageD] num_residues = {num_residues}, num_atoms = {num_atoms}")

    # Create residue-level embeddings
    dummy_embeddings = {
        "s_trunk": torch.randn(batch_size, num_residues, c_s),
        "s_inputs": torch.randn(batch_size, num_residues, c_s_inputs),
        "pair": torch.randn(batch_size, num_residues, num_residues, c_z),
        "atom_metadata": {
            "residue_indices": atom_to_token_idx.squeeze(0),  # Map each atom to its residue index
            "atom_type": torch.arange(atoms_per_residue).repeat(num_residues),  # Types for all residues
            "is_backbone": torch.ones(num_atoms, dtype=torch.bool),  # All atoms
        },
        "sequence": sequence_str,  # Add sequence for proper bridging
        "atom_to_token_idx": atom_to_token_idx,  # Add required mapping
        "ref_space_uid": torch.zeros(batch_size, num_atoms, 3),  # Required by encoder
    }

    # Run Stage D
    try:
        # Use atom_metadata from config if present
        atom_metadata = getattr(stage_cfg, "atom_metadata", None)
        if atom_metadata is not None:
            dummy_embeddings["atom_metadata"] = atom_metadata
        # Ensure we have proper tensor types
        # Use type assertions to help the type checker
        s_trunk_tensor = dummy_embeddings["s_trunk"]
        z_trunk_tensor = dummy_embeddings["pair"]
        s_inputs_tensor = dummy_embeddings["s_inputs"]

        # Verify tensor types
        if not isinstance(s_trunk_tensor, torch.Tensor):
            raise TypeError(f"s_trunk must be a torch.Tensor, got {type(s_trunk_tensor)}")
        if not isinstance(z_trunk_tensor, torch.Tensor):
            raise TypeError(f"z_trunk must be a torch.Tensor, got {type(z_trunk_tensor)}")
        if not isinstance(s_inputs_tensor, torch.Tensor):
            raise TypeError(f"s_inputs must be a torch.Tensor, got {type(s_inputs_tensor)}")
        atom_metadata_dict = dummy_embeddings["atom_metadata"] if isinstance(dummy_embeddings["atom_metadata"], dict) else {}

        # Debug logging for shape verification
        if debug_logging:
            print(f"[DEBUG][run_stageD] s_trunk shape: {s_trunk_tensor.shape}")
            print(f"[DEBUG][run_stageD] z_trunk shape: {z_trunk_tensor.shape}")
            print(f"[DEBUG][run_stageD] s_inputs shape: {s_inputs_tensor.shape}")
            print(f"[DEBUG][run_stageD] num_residues: {num_residues}")
            print(f"[DEBUG][run_stageD] num_atoms: {num_atoms}")

            # Check for shape mismatches
            if s_trunk_tensor.shape[1] != num_residues:
                print(f"[DEBUG][run_stageD] s_trunk shape mismatch: {s_trunk_tensor.shape[1]} != {num_residues}")
            if s_inputs_tensor.shape[1] != num_residues:
                print(f"[DEBUG][run_stageD] s_inputs shape mismatch: {s_inputs_tensor.shape[1]} != {num_residues}")
            if z_trunk_tensor.shape[1] != num_residues:
                print(f"[DEBUG][run_stageD] pair shape mismatch: {z_trunk_tensor.shape[1]} != {num_residues}")

        refined_coords = run_stageD(
            cfg=cfg,
            coords=dummy_coords,
            s_trunk=s_trunk_tensor,
            z_trunk=z_trunk_tensor,
            s_inputs=s_inputs_tensor,
            input_feature_dict=dummy_embeddings,
            atom_metadata=atom_metadata_dict,
        )
        # We already have debug_logging from earlier, no need to extract it again
        if debug_logging:
            # Check if result is a tensor or tuple before accessing shape
            if isinstance(refined_coords, torch.Tensor):
                log.info(f"Successfully refined coordinates: {refined_coords.shape}")
            else:
                log.info("Successfully refined coordinates (training mode)")
    except Exception as e:
        log.error(f"Error during Stage D execution: {str(e)}")
        raise


if __name__ == "__main__":
    hydra_main()
