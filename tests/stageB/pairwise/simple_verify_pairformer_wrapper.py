"""
Simple verification script for PairformerWrapper with a smaller model
"""

import torch

from rna_predict.pipeline.stageB.pairwise.pairformer_wrapper import PairformerWrapper


def main():
    """
    Run a simple verification of the PairformerWrapper with a smaller model
    """
    print("🔍 Testing PairformerWrapper with a smaller model")

    # Create a smaller model for faster testing
    n_blocks = 2  # Much smaller than the default 48
    c_z = 64  # Smaller than default 128
    c_s = 128  # Smaller than default 384

    print(f"Creating PairformerWrapper with n_blocks={n_blocks}, c_z={c_z}, c_s={c_s}")
    wrapper = PairformerWrapper(n_blocks=n_blocks, c_z=c_z, c_s=c_s)

    # Create test tensors
    batch_size = 1
    seq_length = 10  # Smaller sequence length

    s = torch.randn(batch_size, seq_length, c_s)
    z = torch.randn(batch_size, seq_length, seq_length, c_z)
    # Use float tensor for pair_mask instead of boolean to avoid subtraction issues
    pair_mask = torch.ones(batch_size, seq_length, seq_length, dtype=torch.float32)

    print("Running forward pass...")
    s_updated, z_updated = wrapper(s, z, pair_mask)

    print("✅ Forward pass successful")
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

    # Check gradient flow
    s = torch.randn(batch_size, seq_length, c_s, requires_grad=True)
    z = torch.randn(batch_size, seq_length, seq_length, c_z, requires_grad=True)

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


if __name__ == "__main__":
    main()
