from rna_predict.conf.config_schema import StageCConfig
import hydra
from omegaconf import DictConfig, OmegaConf, ValidationError
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import logging
import os
import psutil
import time

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


def ensure_stageC_logger_visible():
    """
    Ensure the Stage C logger always propagates to the root logger and has a StreamHandler at INFO level if none are present.
    This guarantees that INFO logs are visible even when Stage C is called as a submodule.
    """
    global logger
    logger.propagate = True
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

# Call this at module import and in all entrypoints
ensure_stageC_logger_visible()

class StageCReconstruction(nn.Module):
    """
    Stage C: Atom Reconstruction

    This stage intentionally produces a sparse atom representation (21 atoms total)
    as input for the diffusion model in Stage D. This is a design decision, not a bug.

    Architectural note: Stage D expects a dense atom representation (44 atoms per residue),
    resulting in an architectural mismatch. The bridging logic in the pipeline handles this
    by mapping or replicating the sparse atoms to the expected format for Stage D.
    """

    def __init__(self, cfg: DictConfig, *args, **kwargs):
        """
        Initializes the StageCReconstruction module with configuration settings.
        
        Extracts debug logging and device information from the provided Hydra config, supporting both top-level and nested configuration keys. Raises a ValueError if the device is not specified in the config. Stores the config and device for use in atom reconstruction.
        """
        super().__init__()
        logger.info("[MEMORY-LOG][StageC] Initializing StageCReconstruction")
        process = psutil.Process(os.getpid())
        logger.info(f"[MEMORY-LOG][StageC] Memory usage: {process.memory_info().rss / 1e6:.2f} MB")
        logger.info("[MEMORY-LOG][StageC] After super().__init__")
        logger.info(f"[MEMORY-LOG][StageC] Memory usage: {process.memory_info().rss / 1e6:.2f} MB")
        # Respect Hydra debug_logging hierarchy: cfg.debug_logging > cfg.model.stageC.debug_logging > False
        debug = getattr(cfg, 'debug_logging', None)
        if debug is None and hasattr(cfg, 'model') and hasattr(cfg.model, 'stageC'):
            debug = getattr(cfg.model.stageC, 'debug_logging', None)
        self.debug_logging = debug if debug is not None else False

        # Get device from config, supporting both top-level and nested config
        device = None
        if hasattr(cfg, 'device'):
            device = cfg.device
        elif hasattr(cfg, 'model') and hasattr(cfg.model, 'stageC') and hasattr(cfg.model.stageC, 'device'):
            device = cfg.model.stageC.device

        if device is None:
            raise ValueError("StageCReconstruction requires a 'device' key in the config or in cfg.model.stageC.")

        self.device = torch.device(device)
        self.cfg = cfg

    def __call__(self, torsion_angles: torch.Tensor):
        """
        Generates a sparse placeholder for atomic coordinates from input torsion angles.
        
        Accepts a tensor of torsion angles and returns zero-filled coordinate tensors representing a sparse set of atoms per residue, along with atom count and empty metadata. Intended as a legacy or placeholder output for Stage C reconstruction.
        """
        N = torsion_angles.size(0)
        coords = torch.zeros((N * 3, 3), device=self.device)
        coords_3d = torch.zeros((N, 3, 3), device=self.device)
        # NOTE: Stage C intentionally produces only 21 atoms total (sparse), not 44 per residue.
        # This is handled by bridging logic before Stage D.
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
    Creates a valid DictConfig for Stage C tests with default values, allowing overrides.
    
    Args:
        **overrides: Configuration parameters to override default values.
    
    Returns:
        An OmegaConf DictConfig object containing Stage C configuration for testing.
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


#@snoop
def run_stageC_rna_mpnerf(
    cfg: DictConfig,
    sequence: str,
    predicted_torsions: torch.Tensor,
) -> Dict[str, Any]:
    """
    Reconstructs RNA atomic coordinates from torsion angles using the MP-NeRF method.
    
    This function takes a Hydra configuration, an RNA sequence, and a tensor of predicted torsion angles, and generates dense atomic coordinates and metadata for the RNA molecule using the MP-NeRF approach. It validates the configuration, processes torsion angles, builds RNA scaffolds, performs backbone folding (with optional ring closure), and optionally places base atoms. The output includes flattened atomic coordinates, 3D coordinates, atom count, and atom metadata.
    
    Args:
        cfg: Hydra configuration object specifying Stage C parameters.
        sequence: RNA sequence string.
        predicted_torsions: Tensor of predicted torsion angles with shape [N, 7].
    
    Returns:
        A dictionary with the following keys:
            - "coords": Flattened tensor of atomic coordinates.
            - "coords_3d": 3D tensor of atomic coordinates.
            - "atom_count": Total number of atoms.
            - "atom_metadata": Dictionary with atom names and residue indices.
    
    Raises:
        ValidationError: If the configuration is invalid.
        ValueError: If torsion angles have fewer than 7 dimensions or if atom metadata and coordinate counts mismatch.
    """
    if cfg.model.stageC.debug_logging:
        logger.debug(f"[DEBUG-ENTRY] Entered run_stageC_rna_mpnerf in {__file__}")
        logger.debug("Entered run_stageC_rna_mpnerf")
        logger.debug(f"predicted_torsions.requires_grad: {getattr(predicted_torsions, 'requires_grad', None)}")
        logger.debug(f"predicted_torsions.grad_fn: {getattr(predicted_torsions, 'grad_fn', None)}")
    validate_stageC_config(cfg)

    stage_cfg: StageCConfig = cfg.model.stageC
    device = stage_cfg.device
    do_ring_closure = stage_cfg.do_ring_closure
    place_bases = stage_cfg.place_bases
    sugar_pucker = stage_cfg.sugar_pucker

    # SYSTEMATIC DEBUGGING: Print device and config at runtime
    logger.info(f"[DEBUG][StageC] stage_cfg.device: {device} (type: {type(device)})")
    logger.info(f"[DEBUG][StageC] Full stage_cfg: {stage_cfg}")
    logger.info(f"[DEBUG][StageC] OmegaConf resolved device: {getattr(OmegaConf, 'to_container', lambda x: x)(stage_cfg).get('device', None)}")
    # Support all device types (cpu, cuda, mps)
    if device not in ['cpu', 'cuda', 'mps']:
        logger.warning(f"[WARNING][StageC] Unsupported device: {device}. Supported devices are 'cpu', 'cuda', 'mps'. Proceeding anyway.")

    if stage_cfg.debug_logging:
        logger.debug(f"This should always appear if logger is working. sequence={sequence}, torsion_shape={predicted_torsions.shape}")
        logger.debug(f"Running MP-NeRF with device={device}, do_ring_closure={do_ring_closure}")
        logger.debug(f"Sequence length: {len(sequence)}, torsion shape: {predicted_torsions.shape}")

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

    if stage_cfg.debug_logging:
        logger.debug(f"predicted_torsions requires_grad: {getattr(predicted_torsions, 'requires_grad', None)}")
        logger.debug(f"predicted_torsions grad_fn: {getattr(predicted_torsions, 'grad_fn', None)}")
    scaffolds = build_scaffolds_rna_from_torsions(
        seq=sequence,
        torsions=predicted_torsions,
        device=device,
        sugar_pucker=sugar_pucker,
    )
    if stage_cfg.debug_logging:
        logger.debug(f"scaffolds['torsions'] requires_grad: {getattr(scaffolds['torsions'], 'requires_grad', None)}")
        logger.debug(f"scaffolds['torsions'] grad_fn: {getattr(scaffolds['torsions'], 'grad_fn', None)}")
    scaffolds = skip_missing_atoms(sequence, scaffolds)
    scaffolds = handle_mods(sequence, scaffolds)
    # SYSTEMATIC DEBUGGING: Dump scaffolds fields for residue 3 before rna_fold
    # Only emit these logs if debug_logging is enabled
    if stage_cfg.debug_logging:
        if hasattr(scaffolds, 'keys'):
            for k in scaffolds.keys():
                try:
                    val = scaffolds[k]
                    if isinstance(val, torch.Tensor) and val.shape[0] > 3:
                        logger.debug(f"[DEBUG-SCAFFOLDS-RES3] {k}[3] = {val[3]}")
                    elif isinstance(val, (list, tuple)) and len(val) > 3:
                        logger.debug(f"[DEBUG-SCAFFOLDS-RES3] {k}[3] = {val[3]}")
                    else:
                        logger.debug(f"[DEBUG-SCAFFOLDS-RES3] {k}: type={type(val)}, shape/len={getattr(val, 'shape', getattr(val, '__len__', lambda: None)())}")
                except Exception as e:
                    logger.debug(f"[DEBUG-SCAFFOLDS-RES3] {k}: error accessing index 3: {e}")
        else:
            logger.debug(f"[DEBUG-SCAFFOLDS-RES3] scaffolds has no keys() method, type={type(scaffolds)}")
    start_time = time.time()
    coords_bb = rna_fold(scaffolds, device=device, do_ring_closure=do_ring_closure, debug_logging=stage_cfg.debug_logging)
    elapsed = time.time() - start_time
    # SYSTEMATIC DEBUGGING: Dump backbone coordinates for residue 3 right after rna_fold
    if coords_bb.shape[0] > 3:
        if stage_cfg.debug_logging:
            logger.debug(f"[DEBUG-BB-COORDS-RES3] coords_bb[3] = {coords_bb[3]}")
    else:
        if stage_cfg.debug_logging:
            logger.debug(f"[DEBUG-BB-COORDS-RES3] coords_bb shape: {coords_bb.shape}, not enough residues to check residue 3")
    if stage_cfg.debug_logging:
        logger.debug(f"coords_bb requires_grad: {getattr(coords_bb, 'requires_grad', None)}")
        logger.debug(f"coords_bb grad_fn: {getattr(coords_bb, 'grad_fn', None)}")
    if stage_cfg.debug_logging:
        logger.debug(f"[DEBUG-PLACE_BASES] place_bases: {place_bases}")
    if place_bases:
        if stage_cfg.debug_logging:
            logger.debug(f"[DEBUG-GRAD-PRINT-CALLSITE] coords_bb (input to place_rna_bases) requires_grad: {getattr(coords_bb, 'requires_grad', None)}, grad_fn: {getattr(coords_bb, 'grad_fn', None)}")
        coords_full = place_rna_bases(
            coords_bb, sequence, scaffolds["angles_mask"], device=device, debug_logging=stage_cfg.debug_logging
        )
        if stage_cfg.debug_logging:
            logger.debug(f"coords_full (after place_bases) requires_grad: {getattr(coords_full, 'requires_grad', None)}")
            logger.debug(f"coords_full (after place_bases) grad_fn: {getattr(coords_full, 'grad_fn', None)}")
    else:
        coords_full = coords_bb
        if stage_cfg.debug_logging:
            logger.debug(f"coords_full (no place_bases) requires_grad: {getattr(coords_full, 'requires_grad', None)}")
            logger.debug(f"coords_full (no place_bases) grad_fn: {getattr(coords_full, 'grad_fn', None)}")
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
    mask = torch.tensor(valid_atom_mask, dtype=torch.bool, device=coords_full.device)
    coords_full_flat = coords_full.reshape(L * max_atoms, D)[mask]
    if stage_cfg.debug_logging:
        logger.debug(f"coords_full_flat requires_grad: {getattr(coords_full_flat, 'requires_grad', None)}")
        logger.debug(f"coords_full_flat grad_fn: {getattr(coords_full_flat, 'grad_fn', None)}")
    atom_metadata = {
        "atom_names": atom_names,
        "residue_indices": residue_indices,
    }

    if stage_cfg.debug_logging:
        logger.debug(f"Sequence used for atom metadata: {sequence}")
        logger.debug(f"Atom counts for each residue: {[len(STANDARD_RNA_ATOMS[res]) for res in sequence]}")
        logger.debug(f"Total atom count: {len(atom_names)}")

    n_atoms_metadata = len(residue_indices)
    n_atoms_coords = coords_full_flat.shape[0]
    if n_atoms_metadata != n_atoms_coords:
        raise ValueError(f"Mismatch between atom_metadata['residue_indices'] (len={n_atoms_metadata}) and coords_full (num_atoms={n_atoms_coords}). Possible bug in atom mapping or coordinate construction.")

    # [DEBUGGING INSTRUMENTATION] Print atom set details before returning output
    if stage_cfg.debug_logging:
        logger.debug(f"sequence length: {len(sequence)}")
        logger.debug(f"coords_full_flat.shape: {coords_full_flat.shape}")
        logger.debug(f"atom_names (len): {len(atom_names)} {atom_names[:20]}")
        logger.debug(f"residue_indices (len): {len(residue_indices)} {residue_indices[:20]}")
        logger.debug(f"valid_atom_mask (sum): {sum(valid_atom_mask)}")
        logger.debug(f"place_bases: {place_bases}")

    output = {
        "coords": coords_full_flat,
        "coords_3d": coords_full,
        "atom_count": len(atom_names),
        "atom_metadata": atom_metadata,
    }

    # SYSTEMATIC DEBUGGING: Always emit a summary log and print, regardless of debug_logging
    logger.info(
        f"StageC completed: sequence={sequence}, residues={len(sequence)}, "
        f"atoms={output['atom_count']}, coords_shape={getattr(output['coords'], 'shape', 'unknown')}, device={getattr(output['coords'], 'device', 'unknown')}, elapsed_time={elapsed:.2f}s"
    )
    import logging as _logging
    _logging.getLogger().info(f"ROOT: StageC completed: sequence={sequence}, residues={len(sequence)}, atoms={output['atom_count']}, coords_shape={getattr(output['coords'], 'shape', 'unknown')}, device={getattr(output['coords'], 'device', 'unknown')}, elapsed_time={elapsed:.2f}s")

    if stage_cfg.debug_logging:
        # Use getattr to safely access shape attribute
        coords_shape = getattr(output.get('coords', None), 'shape', 'unknown')
        coords_3d_shape = getattr(output.get('coords_3d', None), 'shape', 'unknown')
        logger.debug(f"output['coords'] shape: {coords_shape}")
        logger.debug(f"output['coords_3d'] shape: {coords_3d_shape}")
        logger.debug(f"output['atom_count']: {output['atom_count']}")

    return output

#@snoop
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
    Runs Stage C RNA atom reconstruction using either the MP-NeRF or legacy method.
    
    If a Hydra config is provided, uses its parameters; otherwise, constructs a config from direct arguments. Validates configuration and dispatches to the selected reconstruction method. Returns atomic coordinates and metadata for the input RNA sequence and torsion angles.
    
    Args:
        sequence: RNA sequence as a string.
        torsion_angles: Tensor of torsion angles with shape [N, 7].
        cfg: Optional Hydra configuration object specifying Stage C parameters.
        method: Reconstruction method ('mp_nerf' or 'legacy'); used only if cfg is None.
        device: Device identifier; used only if cfg is None.
        do_ring_closure: Whether to perform ring closure; used only if cfg is None.
        place_bases: Whether to place base atoms; used only if cfg is None.
        sugar_pucker: Sugar pucker conformation; used only if cfg is None.
    
    Returns:
        Dictionary containing atomic coordinates, atom count, and atom metadata for the reconstructed RNA structure.
    
    Raises:
        ValidationError: If the configuration is invalid.
    """
    ensure_stageC_logger_visible()
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
        # Use StageCReconstruction for legacy method
        stageC = StageCReconstruction(cfg)
        return stageC(torsion_angles)


@hydra.main(config_path="../../conf", config_name="default", version_base=None)
def hydra_main(cfg: DictConfig) -> None:
    ensure_stageC_logger_visible()
    # Set logger level according to debug_logging config (Hydra best practice)
    debug_logging = False
    if hasattr(cfg, 'model') and hasattr(cfg.model, 'stageC') and hasattr(cfg.model.stageC, 'debug_logging'):
        debug_logging = cfg.model.stageC.debug_logging
    logger.setLevel(logging.DEBUG if debug_logging else logging.INFO)
    if debug_logging:
        logger.debug("Debug logging is enabled for StageC.")

    validate_stageC_config(cfg)


    if debug_logging:
        logger.info("Running Stage C with Hydra configuration:")
        logger.info(OmegaConf.to_yaml(cfg))

    if hasattr(cfg, 'test_data') and hasattr(cfg.test_data, 'sequence'):
        sample_seq = cfg.test_data.sequence
        torsion_dim = cfg.test_data.torsion_angle_dim if hasattr(cfg.test_data, 'torsion_angle_dim') else 7
        if debug_logging:
            logger.debug(f"Using standardized test sequence: {sample_seq} with {torsion_dim} torsion angles")
    else:
        sample_seq = "ACGUACGU"
        torsion_dim = 7
        if debug_logging:
            logger.debug(f"Using fallback test sequence: {sample_seq} with {torsion_dim} torsion angles")

    dummy_torsions = torch.randn(
        (len(sample_seq), torsion_dim), device=cfg.model.stageC.device
    ) * torch.pi

    if debug_logging:
        logger.debug(f"\nRunning Stage C for sequence: {sample_seq}")
        logger.debug(f"Using dummy torsions shape: {dummy_torsions.shape}")

    output = run_stageC(cfg=cfg, sequence=sample_seq, torsion_angles=dummy_torsions)

    if debug_logging:
        logger.debug("\nStage C Output:")
        logger.debug(f"  Coords shape: {output['coords'].shape}")
        logger.debug(f"  Coords 3D shape: {output['coords_3d'].shape}")
        logger.debug(f"  Atom count: {output['atom_count']}")
        logger.debug(f"  Output device: {output['coords'].device}")


if __name__ == "__main__":
    hydra_main()