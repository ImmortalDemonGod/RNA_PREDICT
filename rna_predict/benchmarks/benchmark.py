################################################################################

import os
import sys
import time

import torch
import torch.nn as nn

from rna_predict.models.encoder.input_feature_embedding import (
    InputFeatureEmbedder,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def benchmark_decoding_latency_and_memory(
    N_atom_list=[128, 256, 512],
    N_token_list=[32, 64, 128],
    block_size=16,
    device="cuda",
    num_warmup=5,
    num_iters=10,
):
    """
    Measures the forward-pass ("decoding") latency and peak GPU memory usage.

    Args:
      N_atom_list: list of N_atom sizes to benchmark.
      N_token_list: list of N_token sizes to benchmark.
      block_size: local attention block size.
      device: "cuda" or "cpu".
      num_warmup: warmup iterations (not timed).
      num_iters: timed iterations.
    """

    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available. Switching to CPU.")
        device = "cpu"
    embedder = InputFeatureEmbedder(
        c_token=384,
        restype_dim=32,
        profile_dim=32,
        c_atom=128,
        c_pair=32,
        num_heads=4,
        num_layers=3,
    ).to(device)

    # Switch to eval mode and no_grad for pure inference
    embedder.eval()

    for N_atom in N_atom_list:
        for N_token in N_token_list:
            print(f"\n=== Decoding Benchmark N_atom={N_atom}, N_token={N_token} ===")

            # Prepare synthetic features on the specified device
            f = {}
            f["ref_pos"] = torch.randn(N_atom, 3, device=device)
            f["ref_charge"] = torch.randint(-2, 3, (N_atom,), device=device).float()
            f["ref_element"] = torch.randn(N_atom, 128, device=device)
            f["ref_atom_name_chars"] = torch.randn(N_atom, 16, device=device)
            f["atom_to_token"] = torch.randint(0, N_token, (N_atom,), device=device)

            f["restype"] = torch.randn(N_token, 32, device=device)
            f["profile"] = torch.randn(N_token, 32, device=device)
            f["deletion_mean"] = torch.randn(N_token, device=device)

            block_index = torch.randint(0, N_atom, (N_atom, block_size), device=device)

            # Warmup (not timed)
            with torch.no_grad():
                for _ in range(num_warmup):
                    _ = embedder(
                        f,
                        trunk_sing=None,
                        trunk_pair=None,
                        block_index=block_index,
                    )
                    torch.cuda.synchronize(device) if device == "cuda" else None

            # Timed decoding + memory usage
            fwd_time = 0.0

            with torch.no_grad():
                for _ in range(num_iters):
                    # Reset peak GPU memory stats
                    if device == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)

                    start = time.time()
                    out = embedder(
                        f,
                        trunk_sing=None,
                        trunk_pair=None,
                        block_index=block_index,
                    )
                    torch.cuda.synchronize(device) if device == "cuda" else None
                    end = time.time()

                    # Record forward (decoding) time
                    fwd_time += end - start

                    if device == "cuda":
                        peak_mem_bytes = torch.cuda.max_memory_allocated(device)
                        peak_mem_mb = peak_mem_bytes / (1024**2)
                        print(f"   Iter peak GPU memory usage: {peak_mem_mb:.2f} MB")

            avg_fwd = fwd_time / num_iters
            print(f"Avg Decoding (Forward) Time: {avg_fwd:.4f} s")


###############################################################################
# Key Points
# 1) Local Block Attention (Naive vs. Optimized):
#    - The naive path uses a Python loop (LocalBlockSparseAttentionNaive).
#    - The optimized path uses block_sparse_attn_func from block_sparse_attn.
#
# 2) Atom-to-Token Aggregation:
#    The per–atom embeddings are aggregated via scatter–mean to form a per–token embedding.
#
# 3) Trunk Recycling (Optional):
#    If available, previous “trunk” embeddings can be added to the per–atom and pair embeddings.
#
# 4) Integration of Extra Token Features:
#    Additional token–level features (restype, profile, deletion stats) are linearly embedded
#    and added to the aggregated atom output before a final layer norm.
#
# This code provides a strong *baseline* for an AlphaFold 3–style input embedding stage,
# now with shape-consistent pair-bias. You can choose naive or block-sparse local attention.
###############################################################################


def benchmark_input_embedding(
    N_atom_list=[128, 256, 512],
    N_token_list=[32, 64, 128],
    block_size=16,
    device="cuda",
    num_warmup=5,
    num_iters=10,
    use_optimized=False,
):
    """
    Benchmarks the InputFeatureEmbedder on random synthetic data,
    measuring forward + backward pass times.
    Toggle use_optimized = True/False to compare naive vs. block-sparse.
    """

    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available. Switching to CPU.")
        device = "cpu"
    embedder = InputFeatureEmbedder(
        c_token=384,
        restype_dim=32,
        profile_dim=32,
        c_atom=128,
        c_pair=32,
        num_heads=4,
        num_layers=3,
        use_optimized=use_optimized,
    ).to(device)

    criterion = nn.MSELoss().to(device)

    for N_atom in N_atom_list:
        for N_token in N_token_list:
            print(
                f"\n=== Benchmarking N_atom={N_atom}, N_token={N_token}, optimized={use_optimized} ==="
            )

            f = {}
            f["ref_pos"] = torch.randn(N_atom, 3, device=device)
            f["ref_charge"] = torch.randint(-2, 3, (N_atom,), device=device).float()
            f["ref_element"] = torch.randn(N_atom, 128, device=device)
            f["ref_atom_name_chars"] = torch.randn(N_atom, 16, device=device)
            f["atom_to_token"] = torch.randint(0, N_token, (N_atom,), device=device)

            f["restype"] = torch.randn(N_token, 32, device=device)
            f["profile"] = torch.randn(N_token, 32, device=device)
            f["deletion_mean"] = torch.randn(N_token, device=device)

            block_index = torch.randint(0, N_atom, (N_atom, block_size), device=device)

            # Warmup
            for _ in range(num_warmup):
                out = embedder(
                    f,
                    trunk_sing=None,
                    trunk_pair=None,
                    block_index=block_index,
                )
                loss = criterion(out, torch.randn(N_token, 384, device=device))
                loss.backward()

            torch.cuda.synchronize(device) if device == "cuda" else None

            fwd_time = 0.0
            bwd_time = 0.0

            for _ in range(num_iters):
                start = time.time()
                out = embedder(
                    f,
                    trunk_sing=None,
                    trunk_pair=None,
                    block_index=block_index,
                )
                torch.cuda.synchronize(device) if device == "cuda" else None
                end = time.time()
                fwd_time += end - start

                loss = criterion(out, torch.randn(N_token, 384, device=device))
                start = time.time()
                loss.backward()
                torch.cuda.synchronize(device) if device == "cuda" else None
                end = time.time()
                bwd_time += end - start

                embedder.zero_grad(set_to_none=True)

            avg_fwd = fwd_time / num_iters
            avg_bwd = bwd_time / num_iters
            print(f"Avg Forward: {avg_fwd:.4f}s,  Avg Backward: {avg_bwd:.4f}s")


if __name__ == "__main__":
    # Example usage: compare naive vs. optimized side by side
    print("NAIVE Implementation:\n")
    benchmark_input_embedding(use_optimized=False)  # old naive path
    print("\nOPTIMIZED Implementation:\n")
    benchmark_input_embedding(use_optimized=True)
    benchmark_decoding_latency_and_memory()
