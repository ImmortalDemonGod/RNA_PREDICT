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

# Add the project root to the path to enable absolute imports
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../")
    ),
)

import logging
from typing import Union, Tuple, Dict, Any, Optional, List
import hydra
from omegaconf import DictConfig
import psutil
import torch
from rna_predict.conf.config_schema import register_configs
from rna_predict.pipeline.stageD.validation_utils import validate_run_stageD_inputs
from rna_predict.pipeline.stageD.config_utils import (
    _print_config_debug,
    _validate_and_extract_stageD_config,
    _get_debug_logging,
    _validate_and_extract_test_data_cfg,
    _extract_diffusion_dims,
)
from rna_predict.pipeline.stageD.bridging_utils import check_and_bridge_embeddings
from rna_predict.pipeline.stageD.output_utils import run_diffusion_and_handle_output
from rna_predict.pipeline.stageD.context import StageDContext
from rna_predict.pipeline.stageD.feature_utils import (
    _validate_feature_config, _validate_atom_metadata, _init_feature_tensors, initialize_features_from_config
)

print("[HYDRA DEBUG] CWD:", os.getcwd())
print("[HYDRA DEBUG] SCRIPT DIR:", os.path.dirname(__file__))
print("[HYDRA DEBUG] sys.path:", sys.path)

# Removed unused snoop import and decorator

log = logging.getLogger(__name__)

# Register Hydra configurations
register_configs()


def log_mem(stage):
    process = psutil.Process(os.getpid())
    print(
        f"[MEMORY-LOG][{stage}] Memory usage: {process.memory_info().rss / (1024 * 1024):.2f} MB"
    )


def _prepare_trunk_embeddings(
    s_trunk, s_inputs, z_trunk, input_feature_dict, atom_metadata
):
    """Prepares trunk_embeddings dict and validates residue/atom-level shapes."""
    trunk_embeddings = {"s_trunk": s_trunk, "s_inputs": s_inputs, "pair": z_trunk}
    features = initialize_features_from_config(input_feature_dict, atom_metadata)
    return trunk_embeddings, features


def _prepare_diffusion_inputs(context):
    """
    Helper to extract and validate all tensors and metadata for diffusion call.
    Reduces argument count and centralizes checks.
    """
    s_trunk_tensor = context.s_trunk
    z_trunk_tensor = context.z_trunk
    s_inputs_tensor = context.s_inputs
    if not isinstance(s_trunk_tensor, torch.Tensor):
        raise TypeError(f"s_trunk must be a torch.Tensor, got {type(s_trunk_tensor)}")
    if not isinstance(z_trunk_tensor, torch.Tensor):
        raise TypeError(f"z_trunk must be a torch.Tensor, got {type(z_trunk_tensor)}")
    if not isinstance(s_inputs_tensor, torch.Tensor):
        raise TypeError(f"s_inputs must be a torch.Tensor, got {type(s_inputs_tensor)}")
    # Ensure atom_metadata is a dictionary
    if context.atom_metadata is not None and not isinstance(context.atom_metadata, dict):
        context.atom_metadata = {}
    return s_trunk_tensor, z_trunk_tensor, s_inputs_tensor, context.atom_metadata


def _run_diffusion_step(context):
    """
    Main diffusion step logic, extracted from main for complexity reduction.
    """
    s_trunk_tensor, z_trunk_tensor, s_inputs_tensor, atom_metadata = _prepare_diffusion_inputs(context)
    # Debug logging for shape verification
    if context.debug_logging:
        print(f"[DEBUG][run_stageD] s_trunk shape: {s_trunk_tensor.shape}")
        print(f"[DEBUG][run_stageD] z_trunk shape: {z_trunk_tensor.shape}")
        print(f"[DEBUG][run_stageD] s_inputs shape: {s_inputs_tensor.shape}")
    # Call diffusion manager and handle output
    run_diffusion_and_handle_output(context)


def _run_stageD_impl(
    context: StageDContext,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Run diffusion refinement on the input coordinates using the unified Stage D runner."""
    log_mem("StageD ENTRY")
    cfg = context.cfg
    coords = context.coords
    s_trunk = context.s_trunk
    z_trunk = context.z_trunk
    s_inputs = context.s_inputs
    input_feature_dict = context.input_feature_dict
    atom_metadata = context.atom_metadata
    validate_run_stageD_inputs(
        cfg, coords, s_trunk, z_trunk, s_inputs, input_feature_dict, atom_metadata
    )
    # --- Refactored: validate config and atom metadata
    stage_cfg = _validate_feature_config(cfg)
    context.stage_cfg = stage_cfg
    residue_indices, num_residues = _validate_atom_metadata(atom_metadata)
    context.residue_indices = residue_indices
    context.num_residues = num_residues
    # ---
    # Determine expected token dimension for s_trunk from config
    n_atoms = coords.shape[1]
    n_residues = s_trunk.shape[1]
    # DEBUG: Print out all relevant config values before any bridging or model instantiation
    print("[HYDRA-CONF-DEBUG][StageD] Dumping config values before diffusion:")
    if hasattr(stage_cfg, 'diffusion'):
        print("  stage_cfg.diffusion:", dict(stage_cfg.diffusion))
    if hasattr(stage_cfg, 'model_architecture'):
        print("  stage_cfg.model_architecture:", dict(stage_cfg.model_architecture))
    if hasattr(stage_cfg, 'feature_dimensions'):
        print("  stage_cfg.feature_dimensions:", dict(stage_cfg.feature_dimensions))
    print("  n_atoms:", n_atoms, "n_residues:", n_residues)
    print("  s_trunk.shape:", getattr(s_trunk, 'shape', None))
    print("  s_inputs.shape:", getattr(s_inputs, 'shape', None))
    print("  z_trunk.shape:", getattr(z_trunk, 'shape', None))
    print("  atom_metadata keys:", list(atom_metadata.keys()) if atom_metadata else None)
    print("  context.device:", context.device)
    print("  context.mode:", context.mode)
    print("  context.debug_logging:", context.debug_logging)
    print("[HYDRA-CONF-DEBUG][StageD] END CONFIG DUMP")
    expected_token_dim = n_atoms if getattr(stage_cfg, 'token_level', 'atom') == 'atom' else n_residues
    # If config expects atom-level and s_trunk is residue-level, bridge now
    if expected_token_dim == n_atoms and s_trunk.shape[1] == n_residues:
        from rna_predict.utils.tensor_utils.embedding import residue_to_atoms
        # Build residue_atom_map from atom_metadata
        residue_atom_map = [[] for _ in range(n_residues)]
        for atom_idx, res_idx in enumerate(atom_metadata['residue_indices']):
            residue_atom_map[res_idx].append(atom_idx)
        s_trunk = residue_to_atoms(s_trunk.squeeze(0) if s_trunk.dim() == 3 else s_trunk, residue_atom_map)
        if s_trunk.dim() == 2:
            s_trunk = s_trunk.unsqueeze(0)
        print(f"[HYDRA-CONF-BRIDGE][StageD] Bridged s_trunk to atom-level: {s_trunk.shape}")
    trunk_embeddings, features = _prepare_trunk_embeddings(
        s_trunk, s_inputs, z_trunk, input_feature_dict, atom_metadata
    )
    context.trunk_embeddings = trunk_embeddings
    context.features = features
    c_s, c_s_inputs, c_z = _extract_diffusion_dims(stage_cfg)
    context.c_s = c_s
    context.c_s_inputs = c_s_inputs
    context.c_z = c_z
    context.mode = getattr(stage_cfg, "mode", None)
    context.device = getattr(stage_cfg, "device", None)
    context.debug_logging = _get_debug_logging(stage_cfg)
    context.diffusion_cfg = getattr(stage_cfg, "diffusion", None)
    log_mem("Before bridging residue-to-atom")
    if atom_metadata is not None:
        num_atoms = coords.shape[1]
        features = _init_feature_tensors(
            batch_size=s_trunk.shape[0],
            num_atoms=num_atoms,
            device=context.device,
            stage_cfg=stage_cfg
        )
    check_and_bridge_embeddings(trunk_embeddings, features, input_feature_dict, coords, atom_metadata)
    log_mem("After bridging residue-to-atom")
    log_mem("Before diffusion")
    print("[DEBUG][run_stageD] Calling unified Stage D runner with DiffusionConfig.")
    _run_diffusion_step(context)
    # Return the result from the diffusion step
    if hasattr(context, 'result') and context.result is not None:
        return context.result
    # Fallback to returning the diffusion config
    return context.diffusion_cfg


def run_stageD(
    cfg,
    coords,
    s_trunk,
    z_trunk,
    s_inputs,
    input_feature_dict,
    atom_metadata=None
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Compatibility wrapper for the old function signature.
    Creates a StageDContext and calls the implementation function.

    This maintains backward compatibility with existing tests while allowing
    the main implementation to use the new context-based approach.
    """
    # Create a context object with the provided arguments
    context = StageDContext(
        cfg=cfg,
        coords=coords,
        s_trunk=s_trunk,
        z_trunk=z_trunk,
        s_inputs=s_inputs,
        input_feature_dict=input_feature_dict,
        atom_metadata=atom_metadata
    )

    # Call the implementation function with the context
    return _run_stageD_impl(context)


# Note: register_configs() is already called at the beginning of the file


# PATCH: Specify config_path and config_name for Hydra best practices
@hydra.main(version_base=None, config_path="../../conf", config_name="default.yaml")
def hydra_main(cfg: DictConfig) -> None:
    _print_config_debug(cfg)
    print("[DEBUG][hydra_main] cfg.model:", cfg.model)
    if hasattr(cfg.model, "stageD"):
        print("[DEBUG][hydra_main] cfg.model.stageD:", cfg.model.stageD)
    else:
        print("[DEBUG][hydra_main] cfg.model.stageD is missing!")

    # Provide default values for stageD configuration if missing
    if not hasattr(cfg.model, "stageD"):
        log.warning("model.stageD configuration missing, creating default configuration")
        from omegaconf import OmegaConf
        default_stageD_config = {
            "enabled": True,
            "mode": "inference",
            "device": "cpu",
            "debug_logging": True,
            "ref_element_size": 128,
            "ref_atom_name_chars_size": 256,
            "profile_size": 32,
            "model_architecture": {
                "c_token": 384,
                "c_s": 384,
                "c_z": 128,
                "c_s_inputs": 449,
                "c_atom": 128,
                "c_atompair": 32,
                "c_noise_embedding": 32,
                "sigma_data": 16.0
            },
            "diffusion": {
                "enabled": True,
                "mode": "inference",
                "device": "cpu"
            }
        }
        cfg.model.stageD = OmegaConf.create(default_stageD_config)

    stage_cfg = _validate_and_extract_stageD_config(cfg)
    debug_logging = _get_debug_logging(stage_cfg)
    _log_stageD_start(debug_logging)
    batch_size = 1  # Use a single batch for testing
    sequence_str, atoms_per_residue, c_s, c_s_inputs, c_z = _extract_sequence_and_dims(cfg, stage_cfg, batch_size)
    _debug_entry_shapes(debug_logging, batch_size, sequence_str, c_z)

    # Generate dummy input data for context
    dummy_embeddings, features = _generate_dummy_inputs(
        batch_size,
        len(sequence_str),
        len(sequence_str) * atoms_per_residue,
        c_s,
        c_s_inputs,
        c_z,
        atoms_per_residue,
        sequence_str,
    )
    context = StageDContext(
        cfg=cfg,
        coords=torch.randn(batch_size, len(sequence_str) * atoms_per_residue, 3),
        s_trunk=dummy_embeddings["s_trunk"],
        z_trunk=dummy_embeddings["pair"],
        s_inputs=dummy_embeddings["s_inputs"],
        input_feature_dict=dummy_embeddings,  # FIX: pass the dict, not the tensor
        atom_metadata=dummy_embeddings.get("atom_metadata", {}),
        debug_logging=debug_logging
    )
    # Call the robust main logic context runner
    _run_stageD_main_logic_context(context)

    _main_stageD_orchestration(cfg)


def _log_stageD_start(debug_logging):
    if debug_logging:
        log.info("Running Stage D Standalone Demo")
        log.debug("[UNIQUE-DEBUG-STAGED-TEST] Stage D runner started.")


def _extract_sequence_and_dims(cfg, stage_cfg, batch_size):
    sequence_str, atoms_per_residue = _validate_and_extract_test_data_cfg(cfg)
    c_s, c_s_inputs, c_z = _extract_diffusion_dims(stage_cfg)
    print(
        f"Using standardized test sequence: {sequence_str} with {atoms_per_residue} atoms per residue"
    )
    return sequence_str, atoms_per_residue, c_s, c_s_inputs, c_z


def _debug_entry_shapes(debug_logging, batch_size, sequence_str, c_z):
    if debug_logging:
        print(
            f"[DEBUG][run_stageD] ENTRY: z_trunk.shape = {torch.randn(batch_size, len(sequence_str), len(sequence_str), c_z).shape}"
        )


def _main_stageD_orchestration(cfg):
    stage_cfg = _validate_and_extract_stageD_config(cfg)
    debug_logging = _get_debug_logging(stage_cfg)
    _log_stageD_start(debug_logging)
    batch_size = 1  # Use a single batch for testing
    sequence_str, atoms_per_residue, c_s, c_s_inputs, c_z = _extract_sequence_and_dims(cfg, stage_cfg, batch_size)
    _debug_entry_shapes(debug_logging, batch_size, sequence_str, c_z)

    # Generate dummy input data for context
    dummy_embeddings, features = _generate_dummy_inputs(
        batch_size,
        len(sequence_str),
        len(sequence_str) * atoms_per_residue,
        c_s,
        c_s_inputs,
        c_z,
        atoms_per_residue,
        sequence_str,
    )
    context = StageDContext(
        cfg=cfg,
        coords=torch.randn(batch_size, len(sequence_str) * atoms_per_residue, 3),
        s_trunk=dummy_embeddings["s_trunk"],
        z_trunk=dummy_embeddings["pair"],
        s_inputs=dummy_embeddings["s_inputs"],
        input_feature_dict=dummy_embeddings,  # FIX: pass the dict, not the tensor
        atom_metadata=dummy_embeddings.get("atom_metadata", {}),
        debug_logging=debug_logging
    )
    # Call the robust main logic context runner
    _run_stageD_main_logic_context(context)


def _run_stageD_main_logic_context(context: StageDContext):
    """Executes Stage D given a StageDContext object. Handles errors and logs as needed."""
    import torch
    try:
        # Ensure we have proper tensor types
        s_trunk_tensor = context.s_trunk
        z_trunk_tensor = context.z_trunk
        s_inputs_tensor = context.s_inputs
        # Verify tensor types
        if not isinstance(s_trunk_tensor, torch.Tensor):
            raise TypeError(f"s_trunk must be a torch.Tensor, got {type(s_trunk_tensor)}")
        if not isinstance(z_trunk_tensor, torch.Tensor):
            raise TypeError(f"z_trunk must be a torch.Tensor, got {type(z_trunk_tensor)}")
        if not isinstance(s_inputs_tensor, torch.Tensor):
            raise TypeError(f"s_inputs must be a torch.Tensor, got {type(s_inputs_tensor)}")
        # Ensure atom_metadata is a dictionary
        if context.atom_metadata is not None and not isinstance(context.atom_metadata, dict):
            context.atom_metadata = {}
        # Debug logging for shape verification
        if context.debug_logging:
            print(f"[DEBUG][run_stageD] s_trunk shape: {s_trunk_tensor.shape}")
            print(f"[DEBUG][run_stageD] z_trunk shape: {z_trunk_tensor.shape}")
            print(f"[DEBUG][run_stageD] s_inputs shape: {s_inputs_tensor.shape}")
            print(f"[DEBUG][run_stageD] num_atoms: {context.coords.shape[1]}")
        refined_coords = _run_stageD_impl(context)
        if context.debug_logging:
            if isinstance(refined_coords, torch.Tensor):
                log.info(f"Successfully refined coordinates: {refined_coords.shape}")
            else:
                log.info("Successfully refined coordinates (training mode)")
    except Exception as e:
        log.error(f"Error during Stage D execution: {str(e)}")
        raise


def _generate_dummy_inputs(
    batch_size,
    num_residues,
    num_atoms,
    c_s,
    c_s_inputs,
    c_z,
    atoms_per_residue,
    sequence_str,
):
    """Generates dummy input tensors and metadata for Stage D."""
    import torch

    atom_to_token_idx = torch.repeat_interleave(
        torch.arange(num_residues), atoms_per_residue
    ).unsqueeze(0)
    dummy_embeddings = {
        "s_trunk": torch.randn(batch_size, num_residues, c_s),
        "s_inputs": torch.randn(batch_size, num_residues, c_s_inputs),
        "pair": torch.randn(batch_size, num_residues, num_residues, c_z),
        "atom_metadata": {
            "residue_indices": atom_to_token_idx.squeeze(0),
            "atom_type": torch.arange(atoms_per_residue).repeat(num_residues),
            "is_backbone": torch.ones(num_atoms, dtype=torch.bool),
        },
        "sequence": sequence_str,
        "atom_to_token_idx": atom_to_token_idx,
        "ref_space_uid": torch.zeros(batch_size, num_atoms, 3),
    }
    return dummy_embeddings, atom_to_token_idx


if __name__ == "__main__":
    hydra_main()
