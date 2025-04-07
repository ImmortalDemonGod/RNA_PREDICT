"""
Memory optimization functions for stageD.
"""

import torch
import gc
from typing import Dict, Tuple, Any

def clear_memory():
    """Clear memory by running garbage collection and emptying CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def apply_memory_fixes(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply memory efficiency fixes to the configuration.
    
    Args:
        config: Original configuration dictionary
        
    Returns:
        Modified configuration with memory efficiency settings
    """
    # Create a deep copy to avoid modifying the original
    fixed_config = config.copy()
    
    # Reduce number of diffusion steps
    fixed_config["inference"]["num_steps"] = 5
    
    # Reduce transformer complexity
    fixed_config["transformer"]["n_heads"] = 2
    fixed_config["transformer"]["n_blocks"] = 1
    
    # Reduce conditioning network size
    fixed_config["conditioning"]["hidden_dim"] = 16
    fixed_config["conditioning"]["num_layers"] = 2
    
    # Reduce manager network size
    fixed_config["manager"]["hidden_dim"] = 16
    fixed_config["manager"]["num_layers"] = 2
    
    # Add memory efficiency options
    fixed_config["memory_efficient"] = True
    fixed_config["use_checkpointing"] = True
    fixed_config["chunk_size"] = 5
    
    return fixed_config

def preprocess_inputs(
    partial_coords: torch.Tensor,
    trunk_embeddings: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Preprocess inputs to reduce memory usage.
    
    Args:
        partial_coords: Input coordinates tensor
        trunk_embeddings: Dictionary of trunk embeddings
        
    Returns:
        Tuple of (processed_coords, processed_embeddings)
    """
    # Limit sequence length to 25
    max_seq_len = 25
    
    # Process coordinates
    processed_coords = partial_coords[:, :max_seq_len, :]
    
    # Process embeddings
    processed_embeddings = {}
    for key, tensor in trunk_embeddings.items():
        if key == "pair":
            # For pair embeddings, reduce both dimensions
            processed_embeddings[key] = tensor[:, :max_seq_len, :max_seq_len, :]
        else:
            # For other embeddings, reduce sequence length
            processed_embeddings[key] = tensor[:, :max_seq_len, :]
    
    return processed_coords, processed_embeddings

def run_stageD_with_memory_fixes(
    partial_coords: torch.Tensor,
    trunk_embeddings: Dict[str, torch.Tensor],
    diffusion_config: Dict[str, Any],
    mode: str = "inference",
    device: str = "cuda"
) -> torch.Tensor:
    """Run stageD with memory efficiency fixes.
    
    Args:
        partial_coords: Input coordinates tensor
        trunk_embeddings: Dictionary of trunk embeddings
        diffusion_config: Configuration dictionary
        mode: Running mode ("inference" or "training")
        device: Device to run on ("cuda" or "cpu")
        
    Returns:
        Refined coordinates tensor
    """
    # Clear memory before starting
    clear_memory()
    
    # Apply memory fixes to config
    fixed_config = apply_memory_fixes(diffusion_config)
    
    # Preprocess inputs
    coords, embeddings = preprocess_inputs(partial_coords, trunk_embeddings)
    
    # Move tensors to device
    coords = coords.to(device)
    embeddings = {k: v.to(device) for k, v in embeddings.items()}
    
    # Initialize model with memory-efficient settings
    from rna_predict.pipeline.stageD.diffusion.model import DiffusionModel
    model = DiffusionModel(fixed_config)
    model = model.to(device)
    
    if fixed_config["use_checkpointing"]:
        model.enable_checkpointing()
    
    # Run inference in chunks
    chunk_size = fixed_config["chunk_size"]
    num_steps = fixed_config["inference"]["num_steps"]
    
    refined_coords = coords
    for i in range(0, num_steps, chunk_size):
        # Process chunk
        chunk_steps = min(chunk_size, num_steps - i)
        refined_coords = model(
            refined_coords,
            embeddings,
            num_steps=chunk_steps,
            mode=mode
        )
        
        # Clear memory after each chunk
        clear_memory()
    
    return refined_coords 