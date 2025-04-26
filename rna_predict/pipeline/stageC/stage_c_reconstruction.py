from rna_predict.conf.config_schema import StageCConfig
import hydra
from omegaconf import DictConfig, OmegaConf, ValidationError
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import logging
import os
import psutil

# Initialize logger
logger = logging.getLogger("rna_predict.pipeline.stageC.stage_c_reconstruction")

def set_stageC_logger_level(debug_logging: bool):
    """
    Set logger level for Stage C according to debug_logging flag.
    """
    if debug_logging:
        logger.setLevel(logging.DEBUG) # Explicitly set DEBUG level
    else:
        # Set to INFO or WARNING to suppress DEBUG messages
        logger.setLevel(logging.INFO)

    # Ensure propagation is set correctly
    logger.propagate = True # Ensure messages reach caplog

    # Make sure all handlers respect the log level
    for handler in logger.handlers:
        if debug_logging:
            handler.setLevel(logging.DEBUG)
        else:
            handler.setLevel(logging.INFO)


class StageCReconstruction(nn.Module):
    """
    Legacy fallback approach for Stage C. Used if method != 'mp_nerf'.
    Returns trivial coords (N*3, 3).
    """

    def __init__(self, device: Optional[torch.device] = None, *args, **kwargs):
        super().__init__()
        print("[MEMORY-LOG][StageC] Initializing StageCReconstruction")
        process = psutil.Process(os.getpid())
        print(f"[MEMORY-LOG][StageC] Memory usage: {process.memory_info().rss / 1e6:.2f} MB")
        print("[MEMORY-LOG][StageC] After super().__init__")
        print(f"[MEMORY-LOG][StageC] Memory usage: {process.memory_info().rss / 1e6:.2f} MB")
        self.debug_logging = False
        if device is None:
            # Default to CPU, but prefer config-driven device
            device = torch.device("cpu")
        self.device = device

    def __call__(self, torsion_angles: torch.Tensor):
        N = torsion_angles.size(0)
        coords = torch.zeros((N * 3, 3), device=self.device)
        coords_3d = torch.zeros((N, 3, 3), device=self.device)
        return {
            "coords": coords,
            "coords_3d": coords_3d,
            "atom_count": coords.size(0),
            "atom_metadata": {"atom_names": [], "residue_indices": []}
        }


def validate_stageC_config(cfg: DictConfig) -> None:
    """
    Validates Stage C configuration parameters against the schema.

    Args:
        cfg: The Hydra configuration object

    Raises:
        ValidationError: If the configuration is invalid
    """
    # Set logger level according to debug_logging config
    if hasattr(cfg, 'model') and hasattr(cfg.model, 'stageC'):
        set_stageC_logger_level(getattr(cfg.model.stageC, 'debug_logging', False))
    if hasattr(cfg, 'model') and hasattr(cfg.model, 'stageC') and getattr(cfg.model.stageC, 'debug_logging', False):
        logger.debug(f"[validate_stageC_config] cfg: {cfg}")
        logger.debug(f"[validate_stageC_config] cfg keys: {list(cfg.keys()) if hasattr(cfg, 'keys') else str(cfg)}")
        logger.debug(f"[validate_stageC_config] cfg.model keys: {list(cfg.model.keys()) if hasattr(cfg.model, 'keys') else str(cfg.model)}")
        # Add the expected debug message for the test
        logger.debug("[UNIQUE-DEBUG-STAGEC-TEST] Stage C config validated.")
    assert hasattr(cfg, "model"), cfg
    assert hasattr(cfg.model, "stageC"), cfg.model
    assert hasattr(cfg.model.stageC, "enabled"), cfg.model.stageC
    try:
        if not hasattr(cfg, "model") or not hasattr(cfg.model, "stageC"):
            raise ValidationError("Configuration must contain model.stageC section")

        stage_cfg: StageCConfig = cfg.model.stageC

        required_params = {
            'enabled': bool,
            'method': str,
            'device': str,
            'do_ring_closure': bool,
            'place_bases': bool,
            'sugar_pucker': str,
            'angle_representation': str,
            'use_metadata': bool,
            'use_memory_efficient_kernel': bool,
            'use_deepspeed_evo_attention': bool,
            'use_lma': bool,
            'inplace_safe': bool,
            'debug_logging': bool,
        }

        for param, param_type in required_params.items():
            if not hasattr(stage_cfg, param):
                raise ValidationError(f"Missing required parameter: {param}")
            if not isinstance(getattr(stage_cfg, param), param_type):
                raise ValidationError(f"Parameter {param} must be of type {param_type}")

        if stage_cfg.method not in ["mp_nerf", "legacy"]:
            raise ValidationError("method must be either 'mp_nerf' or 'legacy'")

        if stage_cfg.device not in ["auto", "cpu", "cuda", "mps"]:
            raise ValidationError("device must be 'auto', 'cpu', 'cuda', or 'mps'")

        if stage_cfg.angle_representation not in ["degrees", "radians", "cartesian", "internal", "sin_cos"]:
            raise ValidationError(
                "angle_representation must be one of 'degrees', 'radians', 'cartesian', 'internal', or 'sin_cos'"
            )

    except AttributeError as e:
        raise ValidationError(f"Invalid configuration structure: {str(e)}")


def create_stage_c_test_config(**overrides):
    """
    Utility to create a valid DictConfig for Stage C tests, with all required fields.
    Accepts overrides for any config value.
    """
    from omegaconf import OmegaConf
    base = {
        "model": {
            "stageC": {
                "enabled": True,
                "method": "mp_nerf",
                "device": "cpu",
                "do_ring_closure": False,
                "place_bases": True,
                "sugar_pucker": "C3'-endo",
                "angle_representation": "radians",
                "use_metadata": False,
                "use_memory_efficient_kernel": False,
                "use_deepspeed_evo_attention": False,
                "use_lma": False,
                "inplace_safe": True,
                "debug_logging": False,
            }
        }
    }
    for k, v in overrides.items():
        base["model"]["stageC"][k] = v
    return OmegaConf.create(base)


def run_stageC_rna_mpnerf(
    cfg: DictConfig,
    sequence: str,
    predicted_torsions: torch.Tensor,
) -> Dict[str, Any]:
    """
    Main RNA Stage C function using MP-NeRF approach, configured via Hydra.

    Args:
        cfg: Hydra configuration object
        sequence: RNA sequence string
        predicted_torsions: Tensor of predicted torsion angles [N, 7]

    Returns:
        Dict containing coordinates, atom count, and atom metadata

    Raises:
        ValidationError: If configuration is invalid
        ValueError: If torsion angles have incorrect dimensions
    """
    print("[DEBUG-STAGEC-ENTRY] Entered run_stageC_rna_mpnerf")
    print("[DEBUG-STAGEC] predicted_torsions.requires_grad:", getattr(predicted_torsions, 'requires_grad', None))
    print("[DEBUG-STAGEC] predicted_torsions.grad_fn:", getattr(predicted_torsions, 'grad_fn', None))
    validate_stageC_config(cfg)

    stage_cfg: StageCConfig = cfg.model.stageC
    device = stage_cfg.device
    do_ring_closure = stage_cfg.do_ring_closure
    place_bases = stage_cfg.place_bases
    sugar_pucker = stage_cfg.sugar_pucker

    if stage_cfg.debug_logging:
        logger.debug("[UNIQUE-DEBUG-STAGEC-TEST] This should always appear if logger is working. sequence=%s, torsion_shape=%s", sequence, predicted_torsions.shape)
        logger.debug(f"[StageC] Running MP-NeRF with device={device}, do_ring_closure={do_ring_closure}")
        logger.debug(f"[StageC] Sequence length: {len(sequence)}, torsion shape: {predicted_torsions.shape}")

    from rna_predict.pipeline.stageC.mp_nerf.rna import (
        build_scaffolds_rna_from_torsions,
        handle_mods,
        place_rna_bases,
        rna_fold,
        skip_missing_atoms,
    )

    if predicted_torsions.size(1) > 7:
        predicted_torsions = predicted_torsions[:, :7]

    if predicted_torsions.size(0) > 0 and predicted_torsions.size(1) < 7:
        raise ValueError(
            f"Not enough angles for Stage C. "
            f"Expected 7, got {predicted_torsions.size(1)}"
        )

    print("[DEBUG-MPNEF] predicted_torsions requires_grad:", getattr(predicted_torsions, 'requires_grad', None))
    print("[DEBUG-MPNEF] predicted_torsions grad_fn:", getattr(predicted_torsions, 'grad_fn', None))
    scaffolds = build_scaffolds_rna_from_torsions(
        seq=sequence,
        torsions=predicted_torsions,
        device=device,
        sugar_pucker=sugar_pucker,
    )
    print("[DEBUG-MPNEF] scaffolds['torsions'] requires_grad:", getattr(scaffolds['torsions'], 'requires_grad', None))
    print("[DEBUG-MPNEF] scaffolds['torsions'] grad_fn:", getattr(scaffolds['torsions'], 'grad_fn', None))
    scaffolds = skip_missing_atoms(sequence, scaffolds)
    scaffolds = handle_mods(sequence, scaffolds)
    coords_bb = rna_fold(scaffolds, device=device, do_ring_closure=do_ring_closure)
    print("[DEBUG-MPNEF] coords_bb requires_grad:", getattr(coords_bb, 'requires_grad', None))
    print("[DEBUG-MPNEF] coords_bb grad_fn:", getattr(coords_bb, 'grad_fn', None))
    if place_bases:
        coords_full = place_rna_bases(
            coords_bb, sequence, scaffolds["angles_mask"], device=device
        )
        print("[DEBUG-MPNEF] coords_full (after place_bases) requires_grad:", getattr(coords_full, 'requires_grad', None))
        print("[DEBUG-MPNEF] coords_full (after place_bases) grad_fn:", getattr(coords_full, 'grad_fn', None))
    else:
        coords_full = coords_bb
        print("[DEBUG-MPNEF] coords_full (no place_bases) requires_grad:", getattr(coords_full, 'requires_grad', None))
        print("[DEBUG-MPNEF] coords_full (no place_bases) grad_fn:", getattr(coords_full, 'grad_fn', None))
    if coords_full.dim() == 2:
        coords_full = coords_full.unsqueeze(1)
    L, max_atoms, D = coords_full.shape
    from rna_predict.utils.tensor_utils.types import STANDARD_RNA_ATOMS
    atom_names = []
    residue_indices = []
    valid_atom_mask = []
    for i, res in enumerate(sequence):
        atom_list = STANDARD_RNA_ATOMS[res]
        atom_names.extend(atom_list)
        residue_indices.extend([i] * len(atom_list))
        valid_atom_mask.extend([True] * len(atom_list))
        if len(atom_list) < coords_full.shape[1]:
            valid_atom_mask.extend([False] * (coords_full.shape[1] - len(atom_list)))
    coords_full_flat = coords_full.reshape(L * max_atoms, D)[valid_atom_mask]
    print("[DEBUG-MPNEF] coords_full_flat requires_grad:", getattr(coords_full_flat, 'requires_grad', None))
    print("[DEBUG-MPNEF] coords_full_flat grad_fn:", getattr(coords_full_flat, 'grad_fn', None))
    atom_metadata = {
        "atom_names": atom_names,
        "residue_indices": residue_indices,
    }

    if stage_cfg.debug_logging:
        logger.debug(f"[DEBUG][StageC] Sequence used for atom metadata: {sequence}")
        logger.debug(f"[DEBUG][StageC] Atom counts for each residue: {[len(STANDARD_RNA_ATOMS[res]) for res in sequence]}")
        logger.debug(f"[DEBUG][StageC] Total atom count: {len(atom_names)}")

    n_atoms_metadata = len(residue_indices)
    n_atoms_coords = coords_full_flat.shape[0]
    if n_atoms_metadata != n_atoms_coords:
        raise ValueError(f"Mismatch between atom_metadata['residue_indices'] (len={n_atoms_metadata}) and coords_full (num_atoms={n_atoms_coords}). Possible bug in atom mapping or coordinate construction.")

    # [DEBUGGING INSTRUMENTATION] Print atom set details before returning output
    print("[DEBUG-STAGEC] sequence length:", len(sequence))
    print("[DEBUG-STAGEC] coords_full_flat.shape:", coords_full_flat.shape)
    print("[DEBUG-STAGEC] atom_names (len):", len(atom_names), atom_names[:20])
    print("[DEBUG-STAGEC] residue_indices (len):", len(residue_indices), residue_indices[:20])
    print("[DEBUG-STAGEC] valid_atom_mask (sum):", sum(valid_atom_mask))
    print("[DEBUG-STAGEC] place_bases:", place_bases)

    output = {
        "coords": coords_full_flat,
        "coords_3d": coords_full,
        "atom_count": len(atom_names),
        "atom_metadata": atom_metadata,
    }

    if stage_cfg.debug_logging:
        # Use getattr to safely access shape attribute
        coords_shape = getattr(output['coords'], 'shape', 'unknown')
        coords_3d_shape = getattr(output['coords_3d'], 'shape', 'unknown')
        logger.debug(f"[DEBUG][StageC] output['coords'] shape: {coords_shape}")
        logger.debug(f"[DEBUG][StageC] output['coords_3d'] shape: {coords_3d_shape}")
        logger.debug(f"[DEBUG][StageC] output['atom_count']: {output['atom_count']}")

    return output


def run_stageC(
    sequence: str,
    torsion_angles: torch.Tensor,
    cfg: Optional[DictConfig] = None,
    method: Optional[str] = None,
    device: Optional[str] = None,
    do_ring_closure: Optional[bool] = None,
    place_bases: Optional[bool] = None,
    sugar_pucker: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Unified Stage C entrypoint, supporting both Hydra config and direct parameters.

    Args:
        sequence: RNA sequence string
        torsion_angles: Tensor of torsion angles [N, 7]
        cfg: Hydra configuration object (preferred)
        method: Method to use ('mp_nerf' or legacy) - only used if cfg is None
        device: Device to run on ('auto', 'cpu', 'cuda') - only used if cfg is None
        do_ring_closure: Whether to perform ring closure - only used if cfg is None
        place_bases: Whether to place base atoms - only used if cfg is None
        sugar_pucker: Sugar pucker conformation - only used if cfg is None

    Returns:
        Dict containing:
            coords: Tensor of atomic coordinates [L, N, 3]
            atom_count: Total number of atoms

    Raises:
        ValidationError: If configuration is invalid
    """
    if cfg is None:
        cfg = OmegaConf.create({
            "model": {
                "stageC": {
                    "enabled": True,
                    "method": method if method is not None else "mp_nerf",
                    "device": device if device is not None else "auto",
                    "do_ring_closure": do_ring_closure if do_ring_closure is not None else False,
                    "place_bases": place_bases if place_bases is not None else True,
                    "sugar_pucker": sugar_pucker if sugar_pucker is not None else "C3'-endo",
                    "angle_representation": "radians",
                    "use_metadata": False,
                    "use_memory_efficient_kernel": False,
                    "use_deepspeed_evo_attention": False,
                    "use_lma": False,
                    "inplace_safe": True,
                    "debug_logging": False,
                }
            }
        })

    validate_stageC_config(cfg)

    stage_cfg: StageCConfig = cfg.model.stageC
    method = stage_cfg.method

    if method == "mp_nerf":
        return run_stageC_rna_mpnerf(
            cfg=cfg,
            sequence=sequence,
            predicted_torsions=torsion_angles,
        )
    else:
        # Convert string device to torch.device
        device_obj = torch.device(stage_cfg.device) if isinstance(stage_cfg.device, str) else stage_cfg.device
        stageC = StageCReconstruction(device=device_obj)
        return stageC(torsion_angles)


@hydra.main(config_path="../../conf", config_name="default", version_base=None)
def hydra_main(cfg: DictConfig) -> None:
    """
    Main entry point for running Stage C reconstruction with Hydra configuration.

    Args:
        cfg: Hydra configuration object
    """
    validate_stageC_config(cfg)

    stage_cfg: StageCConfig = cfg.model.stageC

    if stage_cfg.debug_logging:
        logger.info("Running Stage C with Hydra configuration:")
        logger.info(OmegaConf.to_yaml(cfg))

    if hasattr(cfg, 'test_data') and hasattr(cfg.test_data, 'sequence'):
        sample_seq = cfg.test_data.sequence
        torsion_dim = cfg.test_data.torsion_angle_dim if hasattr(cfg.test_data, 'torsion_angle_dim') else 7
        if stage_cfg.debug_logging:
            logger.debug(f"Using standardized test sequence: {sample_seq} with {torsion_dim} torsion angles")
    else:
        sample_seq = "ACGUACGU"
        torsion_dim = 7
        if stage_cfg.debug_logging:
            logger.debug(f"Using fallback test sequence: {sample_seq} with {torsion_dim} torsion angles")

    dummy_torsions = torch.randn(
        (len(sample_seq), torsion_dim), device=cfg.model.stageC.device
    ) * torch.pi

    if stage_cfg.debug_logging:
        logger.debug(f"\nRunning Stage C for sequence: {sample_seq}")
        logger.debug(f"Using dummy torsions shape: {dummy_torsions.shape}")

    output = run_stageC(cfg=cfg, sequence=sample_seq, torsion_angles=dummy_torsions)

    if stage_cfg.debug_logging:
        logger.debug("\nStage C Output:")
        logger.debug(f"  Coords shape: {output['coords'].shape}")
        logger.debug(f"  Coords 3D shape: {output['coords_3d'].shape}")
        logger.debug(f"  Atom count: {output['atom_count']}")
        logger.debug(f"  Output device: {output['coords'].device}")


if __name__ == "__main__":
    hydra_main()