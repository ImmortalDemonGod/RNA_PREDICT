"""
Debug script for AdaptiveLayerNorm.
"""

import torch
from rna_predict.pipeline.stageA.input_embedding.current.primitives.adaptive_layer_norm import AdaptiveLayerNorm

# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

def debug_adaptive_layernorm():
    """Debug the AdaptiveLayerNorm class with different sequence lengths."""
    print("Creating AdaptiveLayerNorm instance...")
    adaln = AdaptiveLayerNorm(c_a=768, c_s=384)
    
    print("\nTest case 1: Matching sequence lengths")
    a = torch.randn(2, 10, 768)  # [batch, seq, features]
    s = torch.randn(2, 10, 384)  # Matching sequence length
    try:
        result = adaln(a, s)
        print(f"Success! Result shape: {result.shape}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\nTest case 2: s has fewer tokens than a")
    a = torch.randn(2, 10, 768)  # [batch, seq, features]
    s = torch.randn(2, 5, 384)   # Fewer tokens
    try:
        result = adaln(a, s)
        print(f"Success! Result shape: {result.shape}")
    except Exception as e:
        print(f"Error: {e}")
        
    print("\nTest case 3: s has more tokens than a")
    a = torch.randn(2, 5, 768)   # [batch, seq, features]
    s = torch.randn(2, 10, 384)  # More tokens
    try:
        result = adaln(a, s)
        print(f"Success! Result shape: {result.shape}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_adaptive_layernorm()
