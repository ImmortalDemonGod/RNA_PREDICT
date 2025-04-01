################################################################################

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from rna_predict.pipeline.stageA.input_embedding.current.embedders import (
    InputFeatureEmbedder,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def resolve_device(device: str) -> str:
    """
    Checks if CUDA is available. If 'cuda' is requested but not available,
    fallback to 'cpu'.
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available. Switching to CPU.")
        return "cpu"
    return device


@dataclass
class BenchmarkConfig:
    """
    Parameter object to group common benchmarking parameters,
    reducing argument counts and clarifying shared context.
    """

    N_atom_list: List[int] = field(default_factory=lambda: [128, 256, 512])
    N_token_list: List[int] = field(default_factory=lambda: [32, 64, 128])
    block_size: int = 16
    device: str = "cuda"
    num_warmup: int = 5
    num_iters: int = 10
    use_optimized: bool = False


def create_embedder(
    device: str,
    use_optimized: bool = False,
    c_token: int = 384,
    restype_dim: int = 32,
    profile_dim: int = 32,
    c_atom: int = 128,
    c_pair: int = 32,
    num_heads: int = 4,
    num_layers: int = 3,
) -> Tuple[nn.Module, str]:
    """
    Build an InputFeatureEmbedder on the specified device.
    Returns:
      (embedder, actual_device) where device may be switched to 'cpu'
      if CUDA is not available.
    """
    device = resolve_device(device)

    embedder = InputFeatureEmbedder(
        c_token=c_token,
        restype_dim=restype_dim,
        profile_dim=profile_dim,
        c_atom=c_atom,
        c_pair=c_pair,
        num_heads=num_heads,
        num_layers=num_layers,
        use_optimized=use_optimized,
    )
    # Place on CPU/CUDA
    embedder.to(device)
    embedder.eval()
    return embedder, device


def generate_synthetic_features(
    N_atom: int, N_token: int, device: str
) -> Dict[str, torch.Tensor]:
    """
    Create a dictionary of random synthetic input features,
    used as input to the InputFeatureEmbedder in benchmarks.
    """
    f = {}
    f["ref_pos"] = torch.randn(N_atom, 3, device=device)
    f["ref_charge"] = torch.randint(-2, 3, (N_atom,), device=device).float()
    f["ref_element"] = torch.randn(N_atom, 128, device=device)
    f["ref_atom_name_chars"] = torch.randn(N_atom, 256, device=device)
    f["atom_to_token"] = torch.randint(0, N_token, (N_atom,), device=device)
    # Add atom_to_token_idx for compatibility with the encoder
    f["atom_to_token_idx"] = f["atom_to_token"]
    # Add ref_space_uid for compatibility with the encoder's trunk logic
    f["ref_space_uid"] = torch.zeros(N_atom, dtype=torch.int64, device=device)
    # Add ref_mask to indicate all atoms are valid
    f["ref_mask"] = torch.ones(N_atom, device=device)

    f["restype"] = torch.randn(N_token, 32, device=device)
    f["profile"] = 2 * torch.rand(N_token, 32, device=device) - 1
    f["deletion_mean"] = torch.randn(N_token, device=device)
    return f


def warmup_inference(
    embedder: nn.Module,
    f: Dict[str, torch.Tensor],
    block_index: torch.Tensor,
    device: str,
    num_warmup: int,
) -> None:
    """
    Run a series of warmup inferences (not timed),
    allowing GPU kernels to initialize and caches to fill.
    """
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = embedder(
                f,
                trunk_sing=None,
                trunk_pair=None,
                block_index=block_index,
            )
            if device == "cuda":
                torch.cuda.synchronize(device)


# Alias for compatibility
warmup_decoding = warmup_inference
# warmup_embedding will be defined later after warmup_input_embedding is defined


def measure_inference_time_and_memory(
    embedder: nn.Module,
    f: Dict[str, torch.Tensor],
    block_index: torch.Tensor,
    device: str,
    num_iters: int,
) -> float:
    """
    Measure the forward pass latency and peak GPU memory usage
    over multiple timed iterations.
    """
    fwd_time = 0.0
    with torch.no_grad():
        for _ in range(num_iters):
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            start = time.time()
            embedder(f, trunk_sing=None, trunk_pair=None, block_index=block_index)
            if device == "cuda":
                torch.cuda.synchronize(device)
            end = time.time()
            fwd_time += end - start

            if device == "cuda":
                peak_mem_bytes = torch.cuda.max_memory_allocated(device)
                peak_mem_mb = peak_mem_bytes / (1024**2)
                print(f"   Iter peak GPU memory usage: {peak_mem_mb:.2f} MB")

    return fwd_time / num_iters


def benchmark_decoding_latency_and_memory(
    N_atom_list=[128, 256, 512],
    N_token_list=[32, 64, 128],
    block_size=16,
    device="cuda",
    num_warmup=5,
    num_iters=10,
):
    """
    Measures the forward-pass ("decoding") latency and peak GPU memory usage
    for multiple (N_atom, N_token) combos.
    """
    config = BenchmarkConfig(
        N_atom_list=N_atom_list,
        N_token_list=N_token_list,
        block_size=block_size,
        device=device,
        num_warmup=num_warmup,
        num_iters=num_iters,
        use_optimized=False,
    )

    embedder, actual_device = create_embedder(config.device, use_optimized=False)

    for N_atom in config.N_atom_list:
        for N_token in config.N_token_list:
            print(f"\n=== Decoding Benchmark N_atom={N_atom}, N_token={N_token} ===")

            # Prepare synthetic features
            f = generate_synthetic_features(N_atom, N_token, actual_device)
            block_index = torch.randint(
                0, N_atom, (N_atom, config.block_size), device=actual_device
            )

            # Warmup (not timed)
            warmup_inference(embedder, f, block_index, actual_device, config.num_warmup)

            # Timed decoding + memory usage
            avg_fwd = measure_inference_time_and_memory(
                embedder, f, block_index, actual_device, config.num_iters
            )
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
#    If available, previous "trunk" embeddings can be added to the per–atom and pair embeddings.
#
# 4) Integration of Extra Token Features:
#    Additional token–level features (restype, profile, deletion stats) are linearly embedded
#    and added to the aggregated atom output before a final layer norm.
#
# This code provides a strong *baseline* for an AlphaFold 3–style input embedding stage,
# now with shape-consistent pair-bias. You can choose naive or block-sparse local attention.
###############################################################################


def warmup_input_embedding(
    embedder: nn.Module,
    f: Dict[str, torch.Tensor],
    block_index: torch.Tensor,
    device: str,
    num_warmup: int,
    criterion: nn.Module,
) -> None:
    """
    Run a series of warmup forward/backward passes (not timed),
    letting caches warm up for accurate timing.
    """
    for _ in range(num_warmup):
        out = embedder(
            f,
            trunk_sing=None,
            trunk_pair=None,
            block_index=block_index,
        )
        # The shape of out can be [N_token, c_token] or [batch, N_token, c_token]
        # Create a target with matching dimensions
        if out.dim() == 2:
            target = torch.randn(out.size(0), out.size(1), device=device)
        else:  # Handle 3D case or higher
            target = torch.randn_like(out)
        loss = criterion(out, target)
        loss.backward()

        if device == "cuda":
            torch.cuda.synchronize(device)

        embedder.zero_grad(set_to_none=True)

# Alias for compatibility
warmup_embedding = warmup_input_embedding


def time_input_embedding(
    embedder: nn.Module,
    f: Dict[str, torch.Tensor],
    block_index: torch.Tensor,
    device: str,
    num_iters: int,
    criterion: nn.Module,
) -> Tuple[float, float]:
    """
    Measure forward/backward times for the input embedding stage,
    returning the average forward time and backward time in seconds.
    """
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
        if device == "cuda":
            torch.cuda.synchronize(device)
        end = time.time()
        fwd_time += end - start

        # Create a target with matching dimensions
        if out.dim() == 2:
            target = torch.randn(out.size(0), out.size(1), device=device)
        else:  # Handle 3D case or higher
            target = torch.randn_like(out)
        start = time.time()
        loss = criterion(out, target)
        loss.backward()
        if device == "cuda":
            torch.cuda.synchronize(device)
        end = time.time()
        bwd_time += end - start

        embedder.zero_grad(set_to_none=True)

    return fwd_time / num_iters, bwd_time / num_iters


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
    """
    config = BenchmarkConfig(
        N_atom_list=N_atom_list,
        N_token_list=N_token_list,
        block_size=block_size,
        device=device,
        num_warmup=num_warmup,
        num_iters=num_iters,
        use_optimized=use_optimized,
    )

    embedder, actual_device = create_embedder(
        config.device,
        use_optimized=config.use_optimized,
        c_token=384,
        restype_dim=32,
        profile_dim=32,
        c_atom=128,
        c_pair=32,
        num_heads=4,
        num_layers=3,
    )

    criterion = nn.MSELoss().to(actual_device)

    for N_atom in config.N_atom_list:
        for N_token in config.N_token_list:
            print(
                f"\n=== Benchmarking N_atom={N_atom}, N_token={N_token}, optimized={config.use_optimized} ==="
            )

            # Generate synthetic features
            f = generate_synthetic_features(N_atom, N_token, actual_device)
            block_index = torch.randint(
                0, N_atom, (N_atom, config.block_size), device=actual_device
            )

            # Warmup forward/backward
            warmup_input_embedding(
                embedder, f, block_index, actual_device, config.num_warmup, criterion
            )

            # Timed run
            avg_fwd, avg_bwd = time_input_embedding(
                embedder, f, block_index, actual_device, config.num_iters, criterion
            )
            print(f"Avg Forward: {avg_fwd:.4f}s,  Avg Backward: {avg_bwd:.4f}s")


# Additional aliases for test_benchmark usage:
timed_embedding = time_input_embedding


def timed_decoding(
    embedder: nn.Module,
    f: Dict[str, torch.Tensor],
    block_index: torch.Tensor,
    device: str,
    iters: int,
) -> float:
    """
    Measure decoding time by calling measure_inference_time_and_memory.
    """
    return measure_inference_time_and_memory(embedder, f, block_index, device, iters)


if __name__ == "__main__":
    # Example usage: compare naive vs. optimized side by side
    print("NAIVE Implementation:\n")
    benchmark_input_embedding(use_optimized=False)  # old naive path
    print("\nOPTIMIZED Implementation:\n")
    benchmark_input_embedding(use_optimized=True)
    benchmark_decoding_latency_and_memory()