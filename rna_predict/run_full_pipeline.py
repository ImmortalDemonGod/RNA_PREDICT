# Standard library imports
import logging
from dataclasses import dataclass
from typing import Optional, TypedDict, Any, Dict

import numpy as np

# Third-party imports
import torch
from numpy.typing import NDArray
from omegaconf import DictConfig
from torch import device as torch_device

# Local imports
from rna_predict.pipeline.stageA.adjacency.rfold_predictor import StageARFoldPredictor
from rna_predict.pipeline.stageB.main import run_stageB_combined
from rna_predict.pipeline.stageB.pairwise.pairformer_wrapper import PairformerWrapper
from rna_predict.pipeline.stageB.torsion.torsion_bert_predictor import (
    StageBTorsionBertPredictor,
)
from rna_predict.pipeline.stageC.stage_c_reconstruction import run_stageC
from rna_predict.utils.tensor_utils.embedding import residue_to_atoms

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Stage A

# Stage B


# Group all merger inputs into a dataclass for clarity and code quality
@dataclass
class LatentInputs:
    adjacency: torch.Tensor
    angles: torch.Tensor
    s_emb: torch.Tensor
    z_emb: torch.Tensor
    partial_coords: Optional[torch.Tensor] = None


class SimpleLatentMerger(torch.nn.Module):
    """
    Optional: merges adjacency, angles, single embeddings, pair embeddings,
    plus partial coords, into a single per-residue latent.
    """

    def __init__(self, dim_angles: int, dim_s: int, dim_z: int, dim_out: int):
        super().__init__()
        # For example: after pooling z, we have (N, dim_z)
        # angles: (N, dim_angles)
        # s_emb:  (N, dim_s)
        # => total in_dim = dim_angles + dim_s + dim_z
        self.expected_dim_angles = dim_angles
        self.expected_dim_s = dim_s
        self.expected_dim_z = dim_z
        self.dim_out = dim_out

        # Initialize with a placeholder MLP that will be replaced in forward()
        # This fixes the linter errors about assigning Sequential to None
        in_dim = dim_angles + dim_s + dim_z  # Initial expected dimensions
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, dim_out),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_out, dim_out),
        )

    def forward(self, inputs: LatentInputs):
        """
        Merge multiple representations into a unified latent

        Args:
            inputs: LatentInputs dataclass containing adjacency, angles, s_emb, z_emb, and partial_coords

        Returns:
            [N, dim_out] unified latent representation
        """
        # Use the inputs from the dataclass
        adjacency = inputs.adjacency
        angles = inputs.angles
        s_emb = inputs.s_emb
        z_emb = inputs.z_emb
        partial_coords = inputs.partial_coords

        # Example: pool z => shape [N, dim_z]
        z_pooled = z_emb.mean(dim=1)

        # Get actual dimensions
        actual_dim_angles = angles.shape[-1]
        actual_dim_s = s_emb.shape[-1]
        actual_dim_z = z_pooled.shape[-1]

        # Create MLP if dimensions have changed from the current MLP
        total_in_dim = actual_dim_angles + actual_dim_s + actual_dim_z
        if self.mlp[0].in_features != total_in_dim:
            print(
                f"[Debug] Creating MLP with dimensions: {total_in_dim} -> {self.dim_out}"
            )
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(total_in_dim, self.dim_out),
                torch.nn.ReLU(),
                torch.nn.Linear(self.dim_out, self.dim_out),
            ).to(angles.device)
        elif self.mlp[0].weight.device != angles.device:
            # Ensure MLP is on the correct device
            self.mlp = self.mlp.to(angles.device)

        # cat angles + s_emb + z_pooled
        x = torch.cat([angles, s_emb, z_pooled], dim=-1)
        out = self.mlp(x)
        return out


class AtomMetadata(TypedDict):
    residue_indices: torch.Tensor


class PipelineResults(TypedDict):
    partial_coords: Optional[torch.Tensor]
    atom_metadata: Optional[AtomMetadata]
    final_coords: Optional[torch.Tensor]


def get_nan_handling_config(cfg=None) -> tuple[bool, float]:
    """
    Extracts ignore_nan_values and nan_replacement_value from config, or returns defaults.
    """
    ignore_nan_values = False
    nan_replacement_value = 0.0
    if cfg is not None and hasattr(cfg, "pipeline"):
        if hasattr(cfg.pipeline, "ignore_nan_values"):
            ignore_nan_values = cfg.pipeline.ignore_nan_values
        if hasattr(cfg.pipeline, "nan_replacement_value"):
            nan_replacement_value = cfg.pipeline.nan_replacement_value
    return ignore_nan_values, nan_replacement_value


def is_torch_tensor(x):
    return isinstance(x, torch.Tensor)


def is_numpy_array(x):
    return isinstance(x, np.ndarray)


def has_nan_value(tensor):
    if is_torch_tensor(tensor):
        return torch.isnan(tensor).any()
    elif is_numpy_array(tensor):
        return np.isnan(tensor).any()
    return False


def replace_nans_torch(tensor, nan_replacement_value):
    tensor.data = torch.nan_to_num(tensor.data, nan=nan_replacement_value)
    return tensor


def replace_nans_numpy(tensor, nan_replacement_value):
    return np.nan_to_num(tensor, nan=nan_replacement_value)


def handle_nans(
    tensor, name: str, ignore_nan_values: bool, nan_replacement_value: float
):
    """
    Handles NaNs in torch tensors or numpy arrays according to config.
    """
    is_torch = is_torch_tensor(tensor)
    is_numpy = is_numpy_array(tensor)
    if not (is_torch or is_numpy):
        return tensor
    if not has_nan_value(tensor):
        return tensor
    msg_type = " (numpy array)" if is_numpy else ""
    if ignore_nan_values:
        logger.warning(
            f"NaNs found in {name}{msg_type}, replacing with {nan_replacement_value}"
        )
        if is_torch:
            return replace_nans_torch(tensor, nan_replacement_value)
        else:
            return replace_nans_numpy(tensor, nan_replacement_value)
    else:
        logger.error(f"NaNs found in {name}{msg_type}!")
        raise ValueError(f"NaNs found in {name}{msg_type}!")


def check_for_nans(tensor, name: str, cfg=None):
    """
    Checks for NaNs in a tensor or numpy array and handles them according to config.
    """
    ignore_nan_values, nan_replacement_value = get_nan_handling_config(cfg)
    return handle_nans(tensor, name, ignore_nan_values, nan_replacement_value)


def run_stage_a(cfg: DictConfig) -> tuple[torch.Tensor, str]:
    """
    Runs Stage A and returns adjacency and sequence tensors.
    """
    device = torch_device(cfg.device)
    sequence = cfg.sequence
    logger.info(
        f"Starting RNA prediction pipeline for sequence of length {len(sequence)}"
    )
    logger.info(f"Using device: {device}")

    # Stage A: Get adjacency matrix
    logger.info("Stage A: Predicting RNA adjacency matrix...")
    # Use objects for model handles if provided (test/mocking)
    stageA_predictor = None
    if hasattr(cfg, "_objects") and "stageA_predictor" in cfg._objects:
        stageA_predictor = cfg._objects["stageA_predictor"]
    else:
        if not hasattr(cfg, "model") or not hasattr(cfg.model, "stageA"):
            raise ValueError("Configuration must contain model.stageA section")
        stageA_predictor = StageARFoldPredictor(
            stage_cfg=cfg.model.stageA, device=device
        )
    adjacency_np: NDArray = stageA_predictor.predict_adjacency(sequence)
    adjacency_np = check_for_nans(adjacency_np, "adjacency_np (Stage A output)", cfg)
    adjacency = torch.from_numpy(adjacency_np).float().to(device)
    adjacency = check_for_nans(adjacency, "adjacency (Stage A output, torch)", cfg)
    logger.info(f"Stage A completed. Adjacency matrix shape: {adjacency.shape}")
    return adjacency, sequence


def run_stage_b(cfg: DictConfig) -> dict:
    """
    Runs Stage B and returns a dictionary with torsion_angles, s_embeddings, z_embeddings, s_inputs.
    """
    device = torch_device(cfg.device)
    sequence = cfg.sequence
    logger.info("Stage B: Running TorsionBERT and Pairformer models...")
    if hasattr(cfg, "_objects") and "torsion_bert_model" in cfg._objects:
        torsion_bert_model = cfg._objects["torsion_bert_model"]
    else:
        torsion_bert_model = StageBTorsionBertPredictor(cfg)
    if hasattr(cfg, "_objects") and "pairformer_model" in cfg._objects:
        pairformer_model = cfg._objects["pairformer_model"]
    else:
        pairformer_model = PairformerWrapper(cfg)
    stage_b_output = run_stageB_combined(
        sequence=sequence,
        adjacency_matrix=torch.eye(len(sequence), device=device),
        torsion_bert_model=torsion_bert_model,
        pairformer_model=pairformer_model,
        device=str(device),
        cfg=cfg,
    )
    torsion_angles = stage_b_output["torsion_angles"]
    torsion_angles = check_for_nans(
        torsion_angles, "torsion_angles (Stage B output)", cfg
    )
    s_embeddings = stage_b_output["s_embeddings"]
    s_embeddings = check_for_nans(s_embeddings, "s_embeddings (Stage B output)", cfg)
    z_embeddings = stage_b_output["z_embeddings"]
    z_embeddings = check_for_nans(z_embeddings, "z_embeddings (Stage B output)", cfg)
    s_inputs = stage_b_output.get("s_inputs", None)
    if s_inputs is not None:
        s_inputs = check_for_nans(s_inputs, "s_inputs (Stage B output)", cfg)

    logger.info(f"Stage B completed. Torsion angles shape: {torsion_angles.shape}")

    return {
        "torsion_angles": torsion_angles,
        "s_embeddings": s_embeddings,
        "z_embeddings": z_embeddings,
        "s_inputs": s_inputs,
    }


def run_stage_c(cfg: DictConfig, sequence: str, torsion_angles: torch.Tensor) -> dict:
    """
    Runs Stage C reconstruction using the unified Hydra config group.
    Args:
        cfg: The full pipeline config (must contain model.stageC)
        sequence: RNA sequence string
        torsion_angles: Tensor of torsion angles [N, 7]
    Returns:
        Dict with keys: coords, atom_count, atom_metadata, etc.
    """
    return run_stageC(sequence=sequence, torsion_angles=torsion_angles, cfg=cfg)


def run_stage_d(
    cfg: Any,
    coords: torch.Tensor,
    s_embeddings: torch.Tensor,
    z_embeddings: torch.Tensor,
    atom_metadata: Dict[str, Any],
) -> Any:
    """
    Modular Stage D entry point for the RNA pipeline, matching pipeline conventions.
    Args:
        cfg: Full Hydra config (must contain model.stageD)
        coords: Atom coordinates [B, N_atom, 3] or [N_atom, 3]
        s_embeddings: Residue-level embeddings [B, N_res, C] or [N_res, C]
        z_embeddings: Pairwise residue embeddings [B, N_res, N_res, C] or [N_res, N_res, C]
        atom_metadata: Must contain 'residue_indices'
    Returns:
        Output from Stage D diffusion
    """
    from rna_predict.pipeline.stageD.run_stageD import run_stageD
    from rna_predict.pipeline.stageD.context import StageDContext
    from rna_predict.utils.tensor_utils.embedding import residue_to_atoms
    # Bridge residue-level embeddings to atom-level for s_inputs (conditioning)
    residue_indices = atom_metadata.get('residue_indices', None)
    if residue_indices is None:
        raise ValueError("atom_metadata must contain 'residue_indices' for Stage D bridging.")
    n_residues = max(residue_indices) + 1
    residue_atom_map = [[] for _ in range(n_residues)]
    for atom_idx, res_idx in enumerate(residue_indices):
        residue_atom_map[res_idx].append(atom_idx)
    atom_s_inputs = residue_to_atoms(s_embeddings, residue_atom_map)

    def ensure_batch(x):
        return x.unsqueeze(0) if x.dim() == 2 else x
    coords = ensure_batch(coords)
    s_embeddings = ensure_batch(s_embeddings)  # residue-level
    atom_s_inputs = ensure_batch(atom_s_inputs)
    z_embeddings = ensure_batch(z_embeddings)

    stage_d_context = StageDContext(
        cfg=cfg,
        coords=coords,
        s_trunk=s_embeddings,           # RESIDUE-LEVEL
        z_trunk=z_embeddings,
        s_inputs=atom_s_inputs,         # ATOM-LEVEL (conditioning)
        input_feature_dict={},
        atom_metadata=atom_metadata,
    )
    return run_stageD(stage_d_context)


def run_full_pipeline(cfg: DictConfig) -> dict:
    """
    Orchestrates the RNA prediction pipeline (Stages A, B, C, and D).
    Returns a dictionary of pipeline outputs.
    """
    logger.info(f"Pipeline device: {cfg.device}")
    # Stage A
    adjacency, sequence = run_stage_a(cfg)
    # Stage B
    stage_b_output = run_stage_b(cfg)
    torsion_angles = stage_b_output["torsion_angles"]
    torsion_angles = check_for_nans(
        torsion_angles, "torsion_angles (Stage B output)", cfg
    )
    s_embeddings = stage_b_output["s_embeddings"]
    s_embeddings = check_for_nans(s_embeddings, "s_embeddings (Stage B output)", cfg)
    z_embeddings = stage_b_output["z_embeddings"]
    z_embeddings = check_for_nans(z_embeddings, "z_embeddings (Stage B output)", cfg)

    # Stage C: Reconstruct atom coordinates
    stage_c_output = run_stage_c(cfg, sequence, torsion_angles)
    coords = stage_c_output["coords"]
    atom_count = stage_c_output["atom_count"]
    atom_metadata = stage_c_output.get("atom_metadata", None)

    logger.info(f"Stage C completed. Coordinates shape: {coords.shape}")

    # Stage D: Diffusion refinement
    stage_d_output = run_stage_d(
        cfg=cfg,
        coords=coords,
        s_embeddings=s_embeddings,
        z_embeddings=z_embeddings,
        atom_metadata=atom_metadata,
    )

    return {
        "torsion_angles": torsion_angles,
        "s_embeddings": s_embeddings,
        "z_embeddings": z_embeddings,
        "coords": coords,
        "atom_count": atom_count,
        "atom_metadata": atom_metadata,
        "stage_d_output": stage_d_output,
    }
