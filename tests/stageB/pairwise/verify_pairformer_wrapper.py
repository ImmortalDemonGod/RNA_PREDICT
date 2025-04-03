"""
Simple verification script for PairformerWrapper
"""

import sys
from pathlib import Path

import torch

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    # Import the necessary modules
    from rna_predict.pipeline.stageB.pairwise.pairformer_wrapper import (
        PairformerWrapper,
    )

    # Print success message
    print("✅ Successfully imported PairformerWrapper")

    # Try to instantiate the wrapper
    wrapper = PairformerWrapper()
    print("✅ Successfully instantiated PairformerWrapper with default parameters")
    print(f"   - n_blocks: {wrapper.n_blocks}")
    print(f"   - c_z: {wrapper.c_z}")
    print(f"   - c_s: {wrapper.c_s}")

    # Create test tensors
    batch_size = 1
    seq_length = 20
    node_features = 384  # c_s
    edge_features = 128  # c_z

    s = torch.randn(batch_size, seq_length, node_features)
    z = torch.randn(batch_size, seq_length, seq_length, edge_features)
    # Use float tensor for pair_mask instead of boolean to avoid subtraction issues
    pair_mask = torch.ones(batch_size, seq_length, seq_length, dtype=torch.float32)

    # Try to run a forward pass
    try:
        s_updated, z_updated = wrapper(s, z, pair_mask)
        print("✅ Successfully ran forward pass")
        print(f"   - s shape: {s.shape} -> {s_updated.shape}")
        print(f"   - z shape: {z.shape} -> {z_updated.shape}")

        # Check for NaN or Inf values
        if (
            not torch.isnan(s_updated).any()
            and not torch.isinf(s_updated).any()
            and not torch.isnan(z_updated).any()
            and not torch.isinf(z_updated).any()
        ):
            print("✅ No NaN or Inf values in the output")
        else:
            print("❌ Found NaN or Inf values in the output")

        # Try with a different sequence length
        seq_length = 15
        s = torch.randn(batch_size, seq_length, node_features)
        z = torch.randn(batch_size, seq_length, seq_length, edge_features)
        # Use float tensor for pair_mask instead of boolean to avoid subtraction issues
        pair_mask = torch.ones(batch_size, seq_length, seq_length, dtype=torch.float32)

        s_updated, z_updated = wrapper(s, z, pair_mask)
        print("✅ Successfully ran forward pass with different sequence length (15)")
        print(f"   - s shape: {s.shape} -> {s_updated.shape}")
        print(f"   - z shape: {z.shape} -> {z_updated.shape}")

        # Check gradient flow
        s = torch.randn(batch_size, seq_length, node_features, requires_grad=True)
        z = torch.randn(
            batch_size, seq_length, seq_length, edge_features, requires_grad=True
        )

        s_updated, z_updated = wrapper(s, z, pair_mask)
        loss = s_updated.mean() + z_updated.mean()
        loss.backward()

        if s.grad is not None and z.grad is not None:
            print("✅ Gradients flow through the module")
        else:
            print("❌ Gradients do not flow through the module")

        # Count parameters
        param_count = sum(p.numel() for p in wrapper.parameters())
        print(f"✅ Parameter count: {param_count}")

        print("\n🎉 All verification steps completed successfully!")

    except Exception as e:
        print(f"❌ Error during forward pass: {e}")

except ImportError as e:
    print(f"❌ Error importing modules: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
