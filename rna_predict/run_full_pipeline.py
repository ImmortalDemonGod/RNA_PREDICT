from typing import Any, Dict, Optional

import numpy as np
import torch

# Stage A
from rna_predict.pipeline.stageA.run_stageA import run_stageA

# Stage B
from rna_predict.pipeline.stageB.main import run_stageB_combined
from rna_predict.pipeline.stageB.pairwise.pairformer_wrapper import PairformerWrapper
from rna_predict.pipeline.stageB.torsion.torsion_bert_predictor import (
    StageBTorsionBertPredictor,
)

# Stage C
from rna_predict.pipeline.stageC.stage_c_reconstruction import StageCReconstruction

# Stage D
try:
    from rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager import (
        ProtenixDiffusionManager,
    )
    from rna_predict.pipeline.stageD.run_stageD_unified import run_stageD_diffusion

    STAGE_D_AVAILABLE = True
except ImportError:
    STAGE_D_AVAILABLE = False
    print(
        "[Warning] Stage D modules could not be imported. Stage D functionality will be disabled."
    )


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

    def forward(
        self,
        adjacency: torch.Tensor,
        angles: torch.Tensor,
        s_emb: torch.Tensor,
        z_emb: torch.Tensor,
        partial_coords: Optional[torch.Tensor] = None,
    ):
        """
        Merge multiple representations into a unified latent

        Args:
            adjacency: [N, N] adjacency matrix
            angles: [N, dim_angles] torsion angles
            s_emb: [N, dim_s] single-residue embeddings
            z_emb: [N, N, dim_z] pair embeddings
            partial_coords: optional [N, 3] or [N*#atoms, 3] partial coordinates

        Returns:
            [N, dim_out] unified latent representation
        """
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


def run_full_pipeline(
    sequence: str, config: Dict[str, Any], device: str = "cuda"
) -> Dict[str, Any]:
    """
    A top-level function orchestrating:
       Stage A -> adjacency
       Stage B -> (TorsionBERT + Pairformer)
       (Optionally) Stage C -> partial coords
       (Optionally) unify or pass to Stage D

    Args:
        sequence: RNA sequence string
        config: Dictionary with configuration parameters and model instances:
            - stageA_predictor: adjacency predictor instance
            - torsion_bert_model: TorsionBertPredictor instance
            - pairformer_model: PairformerWrapper instance
            - enable_stageC: bool, whether to run Stage C
            - init_z_from_adjacency: bool, whether to initialize pair embeddings from adjacency
            - merge_latent: bool, whether to create a unified latent representation
            - merger: Optional, instance of SimpleLatentMerger if merge_latent=True
            - run_stageD: bool, whether to run Stage D diffusion
            - diffusion_manager: Optional, instance of ProtenixDiffusionManager if run_stageD=True
            - stageD_config: Optional, config dict for Stage D if run_stageD=True
        device: device to run computations on ("cpu" or "cuda")

    Returns:
        Dictionary with results:
        {
            "adjacency": torch.Tensor,
            "torsion_angles": torch.Tensor,
            "s_embeddings": torch.Tensor,
            "z_embeddings": torch.Tensor,
            "partial_coords": torch.Tensor or None,
            "unified_latent": torch.Tensor or None,
            "final_coords": torch.Tensor or None
        }
    """

    # 1) Stage A: adjacency
    stageA_predictor = config.get("stageA_predictor")
    if not stageA_predictor:
        raise ValueError("Config must provide 'stageA_predictor' for adjacency.")
    adjacency_np = run_stageA(sequence, stageA_predictor)
    adjacency_t = torch.from_numpy(adjacency_np).float().to(device)

    # 2) Stage B: TorsionBERT + Pairformer
    torsion_model = config.get("torsion_bert_model")
    pairformer_model = config.get("pairformer_model")
    if not torsion_model or not pairformer_model:
        raise ValueError(
            "Need both 'torsion_bert_model' and 'pairformer_model' in config."
        )

    stageB_out = run_stageB_combined(
        sequence=sequence,
        adjacency_matrix=adjacency_t,
        torsion_bert_model=torsion_model,
        pairformer_model=pairformer_model,
        device=device,
        init_z_from_adjacency=config.get("init_z_from_adjacency", False),
    )

    torsion_angles = stageB_out["torsion_angles"]
    s_emb = stageB_out["s_embeddings"]
    z_emb = stageB_out["z_embeddings"]

    # 3) Optional Stage C
    partial_coords = None
    if config.get("enable_stageC", False):
        stage_c = StageCReconstruction()
        partial_res = stage_c(torsion_angles.to(device))
        partial_coords = partial_res[
            "coords"
        ]  # e.g., shape [N * #atoms, 3] or [N, #atoms, 3]

    # 4) Optional unify/merge
    unified_latent = None
    if config.get("merge_latent", False):
        if "merger" not in config:
            raise ValueError(
                "Config sets 'merge_latent' but no 'merger' object provided."
            )
        merger_module = config["merger"]

        # Print tensor shapes for debugging
        print("[Debug] Tensor shapes for merger:")
        print(f"  adjacency: {adjacency_t.shape}")
        print(f"  torsion_angles: {torsion_angles.shape}")
        print(f"  s_emb: {s_emb.shape}")
        print(f"  z_emb: {z_emb.shape}")

        unified_latent = merger_module(
            adjacency=adjacency_t,
            angles=torsion_angles,
            s_emb=s_emb,
            z_emb=z_emb,
            partial_coords=partial_coords,
        )

    # 5) Optional Stage D
    final_coords = None
    if config.get("run_stageD", False) and STAGE_D_AVAILABLE:
        diffusion_manager = config.get("diffusion_manager")
        stageD_config = config.get("stageD_config", {})
        if not diffusion_manager:
            raise ValueError(
                "Config indicates 'run_stageD' but no 'diffusion_manager' provided."
            )

        # If you want to incorporate unified_latent, you'd adapt run_stageD_diffusion.
        # For now, we demonstrate the simpler approach using 's_embeddings' & 'z_embeddings'.
        trunk_embeds = {
            "s_trunk": s_emb.unsqueeze(0),
            "pair": z_emb.unsqueeze(0),
        }

        # Check if we have partial coordinates
        if partial_coords is None:
            # Create random initial coords as placeholder
            N = len(sequence)
            N_atom = N * 5  # rough estimate for RNA
            partial_coords = torch.randn((1, N_atom, 3), device=device)
        # Make sure partial_coords has batch dimension
        elif len(partial_coords.shape) == 2:
            partial_coords = partial_coords.unsqueeze(0)

        final_coords = run_stageD_diffusion(
            partial_coords=partial_coords,
            trunk_embeddings=trunk_embeds,
            diffusion_config=stageD_config,
            mode="inference",
            device=device,
        )

    return {
        "adjacency": adjacency_t,
        "torsion_angles": torsion_angles,
        "s_embeddings": s_emb,
        "z_embeddings": z_emb,
        "partial_coords": partial_coords,
        "unified_latent": unified_latent,
        "final_coords": final_coords,
    }


if __name__ == "__main__":
    print("Example usage of run_full_pipeline with dummy config.")
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    class DummyStageAPredictor:
        """Dummy predictor that returns a simple adjacency matrix"""

        def predict_adjacency(self, seq):
            N = len(seq)
            adj = np.eye(N, dtype=np.float32)
            if N > 1:
                adj[0, 1] = adj[1, 0] = 1.0
            return adj

    # Create dummy torsion model, pairformer model, etc.
    try:
        tmodel = StageBTorsionBertPredictor(
            model_name_or_path="sayby/rna_torsionbert", device=dev
        )
    except Exception as e:
        print(
            f"[Warning] Could not load 'sayby/rna_torsionbert'. Using dummy path. Error: {str(e)}"
        )
        tmodel = StageBTorsionBertPredictor(model_name_or_path="dummy_path", device=dev)

    pfmodel = PairformerWrapper(n_blocks=2, c_z=32, c_s=64, use_checkpoint=False).to(
        dev
    )

    # For merging:
    merger = SimpleLatentMerger(dim_angles=7, dim_s=64, dim_z=32, dim_out=128).to(dev)

    # Build configuration for pipeline
    pipeline_config = {
        "stageA_predictor": DummyStageAPredictor(),
        "torsion_bert_model": tmodel,
        "pairformer_model": pfmodel,
        "merger": merger,
        "enable_stageC": True,
        "merge_latent": True,
        "init_z_from_adjacency": True,
    }

    # Add Stage D components if available
    if STAGE_D_AVAILABLE:
        # For Stage D:
        dummy_diffusion_config = {
            "sigma_data": 16.0,
            "c_atom": 128,
            "c_atompair": 16,
            "c_token": 768,
            "c_s": 64,
            "c_z": 32,
            "c_s_inputs": 384,
            "atom_encoder": {"n_blocks": 1, "n_heads": 2},
            "transformer": {"n_blocks": 1, "n_heads": 2},
            "atom_decoder": {"n_blocks": 1, "n_heads": 2},
            "initialization": {},
        }
        diffusion_manager = ProtenixDiffusionManager(dummy_diffusion_config, device=dev)
        pipeline_config.update(
            {
                "diffusion_manager": diffusion_manager,
                "stageD_config": dummy_diffusion_config,
                # Enable Stage D now that we've fixed the index out of bounds issue
                "run_stageD": True,
            }
        )

    seq_input = "AUGCAUGG"
    final_res = run_full_pipeline(seq_input, pipeline_config, device=dev)

    print("\n--- Pipeline Output ---")
    for k, v in final_res.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={tuple(v.shape)}")
        else:
            print(f"  {k}: {v}")
    print("Done.")
