import os
import sys

import numpy as np
import torch

# Add the parent directory to path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import pipeline components
from rna_predict.pipeline.stageB.pairwise.pairformer_wrapper import PairformerWrapper
from rna_predict.pipeline.stageB.torsion.torsion_bert_predictor import (
    StageBTorsionBertPredictor,
)
from rna_predict.run_full_pipeline import SimpleLatentMerger, run_full_pipeline

try:
    from rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager import (
        ProtenixDiffusionManager,
    )

    STAGE_D_AVAILABLE = True
except ImportError:
    STAGE_D_AVAILABLE = False


# Helper function to print tensors with truncation
def print_tensor_example(name, tensor, max_items=5):
    if tensor is None:
        print(f"  {name}: None")
        return

    # Print shape
    shape_str = f"shape={tuple(tensor.shape)}"

    # Get tensor data as numpy for easier handling
    if isinstance(tensor, torch.Tensor):
        data = tensor.detach().cpu().numpy()
    else:
        data = tensor

    # Format the data depending on dimensions
    if len(data.shape) == 1:
        # 1D tensor
        truncated = str(data[:max_items])
        if data.shape[0] > max_items:
            truncated = truncated[:-1] + ", ...]"
    elif len(data.shape) == 2:
        # 2D tensor
        rows = min(data.shape[0], max_items)
        cols = min(data.shape[1], max_items)
        truncated = "["
        for i in range(rows):
            row = data[i, :cols]
            row_str = np.array2string(row, precision=4, suppress_small=True)
            if cols < data.shape[1]:
                row_str = row_str[:-1] + ", ...]"
            truncated += row_str + ",\n "
        if rows < data.shape[0]:
            truncated += " ...]"
        else:
            truncated += "]"
    else:
        # Higher-dimensional tensor - just show the first slice
        truncated = f"First slice: {data[0, : min(data.shape[1], max_items), : min(data.shape[2], max_items)]}"

    print(f"  {name}: {shape_str}")
    print(f"  Example values: {truncated}")
    print()


# Set up pipeline components
def setup_pipeline():
    # Use CPU for testing
    device = "cpu"

    # Create dummy Stage A predictor
    class DummyStageAPredictor:
        def predict_adjacency(self, seq):
            N = len(seq)
            adj = np.eye(N, dtype=np.float32)
            # Add some off-diagonal interactions
            if N > 1:
                for i in range(N - 1):
                    adj[i, i + 1] = adj[i + 1, i] = 0.8
                # Add some non-local interactions
                if N > 4:
                    adj[0, N - 1] = adj[N - 1, 0] = 0.5
            return adj

    # Create models
    try:
        tmodel = StageBTorsionBertPredictor(
            model_name_or_path="sayby/rna_torsionbert", device=device
        )
    except Exception:
        print("[Warning] Could not load 'sayby/rna_torsionbert'. Using dummy model.")
        # Create a dummy torsion model
        tmodel = StageBTorsionBertPredictor(model_name_or_path=None, device=device)
        tmodel.output_dim = 14

        # Mock the predict method
        def mock_predict(seq):
            N = len(seq)
            return torch.randn((N, 14))

        tmodel.predict = mock_predict

    # Create a simple Pairformer model
    c_s = 64  # Single embedding dimension
    c_z = 32  # Pair embedding dimension
    pfmodel = PairformerWrapper(n_blocks=2, c_z=c_z, c_s=c_s, use_checkpoint=False).to(
        device
    )

    # Create merger
    merger = SimpleLatentMerger(dim_angles=14, dim_s=c_s, dim_z=c_z, dim_out=128).to(
        device
    )

    # Build configuration
    pipeline_config = {
        "stageA_predictor": DummyStageAPredictor(),
        "torsion_bert_model": tmodel,
        "pairformer_model": pfmodel,
        "merger": merger,
        "enable_stageC": True,
        "merge_latent": True,
        "init_z_from_adjacency": True,
    }

    # Add Stage D if available
    if STAGE_D_AVAILABLE:
        dummy_diffusion_config = {
            "sigma_data": 16.0,
            "c_atom": 128,
            "c_atompair": 16,
            "c_token": 768,
            "c_s": c_s,
            "c_z": c_z,
            "c_s_inputs": 384,
            "atom_encoder": {"n_blocks": 1, "n_heads": 2},
            "transformer": {"n_blocks": 1, "n_heads": 2},
            "atom_decoder": {"n_blocks": 1, "n_heads": 2},
            "initialization": {},
        }
        diffusion_manager = ProtenixDiffusionManager(
            dummy_diffusion_config, device=device
        )
        pipeline_config.update(
            {
                "diffusion_manager": diffusion_manager,
                "stageD_config": dummy_diffusion_config,
                "run_stageD": True,
            }
        )

    return pipeline_config, device


def main():
    # Set up the pipeline
    config, device = setup_pipeline()

    # Example RNA sequence
    sequence = "AUGCAUGG"

    print(f"Running RNA prediction pipeline on sequence: {sequence}")
    print(f"Stage D available: {STAGE_D_AVAILABLE}")

    # Run the pipeline
    try:
        results = run_full_pipeline(sequence, config, device=device)

        # Print output with examples
        print("\n--- Pipeline Output with Examples ---")
        for k, v in results.items():
            print_tensor_example(k, v)

    except Exception as e:
        print(f"Error running pipeline: {str(e)}")
        import traceback

        traceback.print_exc()

    print("Done.")


if __name__ == "__main__":
    main()
