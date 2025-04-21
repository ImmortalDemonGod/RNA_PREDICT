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
    ref_element_size = stage_cfg.ref_element_size
    features["ref_element"] = torch.zeros(batch_size, num_atoms, ref_element_size, device=device)

    ref_atom_name_chars_size = stage_cfg.ref_atom_name_chars_size
    features["ref_atom_name_chars"] = torch.zeros(batch_size, num_atoms, ref_atom_name_chars_size, device=device)

    # Initialize atom_to_token_idx mapping
    residue_indices_tensor = torch.tensor(residue_indices, device=device, dtype=torch.long)
    features["atom_to_token_idx"] = residue_indices_tensor.unsqueeze(0).expand(batch_size, -1)  # [batch_size, num_atoms]
    log.info(f"Using atom metadata for atom_to_token_idx mapping with {num_residues} residues")

    # Initialize additional features required by the model
    features["restype"] = torch.zeros(batch_size, num_residues, device=device, dtype=torch.long)  # [batch_size, num_residues]

    # Use default profile dimension
    profile_size = 32  # Default value
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
        features["ref_element"] = torch.zeros(batch_size, num_atoms, 128, device=device)  # [batch_size, num_atoms, 128]
    if "ref_atom_name_chars" not in features:
        features["ref_atom_name_chars"] = torch.zeros(batch_size, num_atoms, 256, device=device)  # [batch_size, num_atoms, 256]

    # Add all the required features to match the expected dimension
    # Calculate the total dimension dynamically from the config values
    ref_element_dim = stage_cfg.ref_element_size
    ref_atom_name_chars_dim = stage_cfg.ref_atom_name_chars_size
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

    # Mode and device
    mode = getattr(stage_cfg, "mode", "inference")
    device = getattr(stage_cfg, "device", "cpu")
    debug_logging = getattr(stage_cfg, "debug_logging", False)
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

@hydra.main(config_path="../../conf", config_name="default", version_base=None)
def hydra_main(cfg: DictConfig) -> None:
    """Main entry point for running Stage D with Hydra configuration.

    Args:
        cfg: Hydra configuration object
    """
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

    # Validate test_data configuration
    if not hasattr(cfg, "test_data") or not hasattr(cfg.test_data, "sequence"):
        raise ValueError("Configuration must contain test_data.sequence section")

    if not hasattr(cfg.test_data, "atoms_per_residue"):
        raise ValueError("Configuration must contain test_data.atoms_per_residue")

    # Use standardized test sequence from config
    sequence_str = cfg.test_data.sequence
    sequence = list(sequence_str)  # Convert to list of characters
    atoms_per_residue = cfg.test_data.atoms_per_residue
    print(f"Using standardized test sequence: {sequence_str} with {atoms_per_residue} atoms per residue")

    num_residues = len(sequence)
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
    # Get dimensions from config or use defaults
    c_s = stage_cfg.c_s if hasattr(stage_cfg, 'c_s') else 384
    c_s_inputs = stage_cfg.c_s_inputs if hasattr(stage_cfg, 'c_s_inputs') else 32
    c_z = stage_cfg.c_z if hasattr(stage_cfg, 'c_z') else 128

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
        "sequence": sequence,  # Add sequence for proper bridging
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
