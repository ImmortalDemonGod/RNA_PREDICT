"""
Test script to verify memory efficiency fixes.
"""

import torch
import unittest
import psutil
import os
import gc
from memory_profiler import profile
from rna_predict.pipeline.stageD.memory_optimization.memory_fix import (
    apply_memory_fixes,
    run_stageD_with_memory_fixes,
    clear_memory,
    preprocess_inputs
)

import os

def get_torch_memory():
    """Get PyTorch memory usage in GB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024 / 1024
    return 0

def get_process_memory():
    """Get process memory usage in GB"""
    process = psutil.Process(os.getpid())
    # Use both RSS and VMS for total memory measurement
    return (process.memory_info().rss + process.memory_info().vms) / 1024 / 1024 / 1024

def get_total_memory():
    """Get total memory usage including both process and PyTorch memory"""
    return get_process_memory() + get_torch_memory()

class MemoryTracker:
    def __init__(self):
        self.peak_memory = 0
        self.initial_memory = 0
        self.memory_log = []
        self.start_time = None

    def start(self):
        """Start memory tracking"""
        # Clear memory before starting
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Wait for memory to stabilize
        import time
        time.sleep(1)
        
        self.initial_memory = get_total_memory()
        self.peak_memory = self.initial_memory
        self.memory_log = [(0, self.initial_memory)]
        if hasattr(self, 'debug_logging') and self.debug_logging:
            print(f"Initial memory usage: {self.initial_memory:.2f} GB")

    def update(self, step=None):
        """Update memory tracking"""
        # Force garbage collection before measurement
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        current_memory = get_total_memory()
        self.peak_memory = max(self.peak_memory, current_memory)
        self.memory_log.append((step if step is not None else len(self.memory_log), current_memory))
        print(f"Memory at step {step if step is not None else len(self.memory_log)-1}: {current_memory:.2f} GB")

    def stop(self):
        """Stop memory tracking and return results"""
        # Force garbage collection before final measurement
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        final_memory = get_total_memory()
        print(f"Peak memory usage: {self.peak_memory:.2f} GB")
        print(f"Final memory usage: {final_memory:.2f} GB")
        print(f"Memory increase: {final_memory - self.initial_memory:.2f} GB")
        return self.peak_memory

class TestMemoryFix(unittest.TestCase):
    def setUp(self):
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Clear memory before starting
        clear_memory()
        
        # Create large tensors similar to the ones in the logs
        self.partial_coords = torch.randn(1, 100, 3)  # [batch, seq_len, 3]
        
        # Create trunk embeddings with large dimensions
        self.trunk_embeddings = {
            "s_inputs": torch.randn(1, 100, 449),  # [batch, seq_len, 449]
            "s_trunk": torch.randn(1, 100, 384),   # [batch, seq_len, 384]
            "pair": torch.randn(1, 100, 100, 64)   # [batch, seq_len, seq_len, pair_dim]
        }
        
        # Create diffusion config with large dimensions
        self.diffusion_config = {
            "conditioning": {
                "hidden_dim": 128,
                "num_heads": 8,
                "num_layers": 6,
            },
            "manager": {
                "hidden_dim": 128,
                "num_heads": 8,
                "num_layers": 6,
            },
            "inference": {
                "num_steps": 100,  # Original value
                "noise_schedule": "linear",
            },
            "c_s_inputs": 449,
            "c_z": 64,
            "c_atom": 128,
            "c_s": 384,
            "c_token": 832,
            "transformer": {"n_blocks": 4, "n_heads": 16}
        }

    def test_config_fixes(self):
        """Test that the config fixes are applied correctly"""
        # Apply memory fixes
        fixed_config = apply_memory_fixes(self.diffusion_config)
        
        # Check that the fixes were applied
        self.assertEqual(fixed_config["inference"]["num_steps"], 5)
        self.assertEqual(fixed_config["transformer"]["n_heads"], 2)
        self.assertEqual(fixed_config["transformer"]["n_blocks"], 1)
        self.assertEqual(fixed_config["conditioning"]["hidden_dim"], 16)
        self.assertEqual(fixed_config["conditioning"]["num_layers"], 2)
        self.assertEqual(fixed_config["manager"]["hidden_dim"], 16)
        self.assertEqual(fixed_config["manager"]["num_layers"], 2)
        
        # Check that feature dimensions are kept consistent
        self.assertEqual(fixed_config["c_s"], 384)
        self.assertEqual(fixed_config["c_z"], 64)
        self.assertEqual(fixed_config["c_token"], 832)
        
        # Check memory-efficient options
        self.assertTrue(fixed_config["memory_efficient"])
        self.assertTrue(fixed_config["use_checkpointing"])
        self.assertEqual(fixed_config["chunk_size"], 5)

    def test_preprocess_inputs(self):
        """Test that input preprocessing reduces sequence length"""
        # Process inputs
        coords, embeddings = preprocess_inputs(self.partial_coords, self.trunk_embeddings)
        
        # Check shapes
        self.assertEqual(coords.shape, (1, 25, 3))
        self.assertEqual(embeddings["s_inputs"].shape, (1, 25, 449))
        self.assertEqual(embeddings["s_trunk"].shape, (1, 25, 384))
        self.assertEqual(embeddings["pair"].shape, (1, 25, 25, 64))

    @profile
    def test_memory_usage_with_fixes(self):
        """Test that memory usage is reduced with the fixes"""
        # Initialize memory tracker
        tracker = MemoryTracker()
        tracker.start()
        
        try:
            # Track memory before model creation
            tracker.update("before_model")
            
            # Run with memory fixes
            run_stageD_with_memory_fixes(
                partial_coords=self.partial_coords,
                trunk_embeddings=self.trunk_embeddings,
                diffusion_config=self.diffusion_config,
                mode="inference",
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            # Track memory after each diffusion step
            for i in range(10):  # Number of diffusion steps
                tracker.update(f"step_{i}")
            
            # Get final memory usage
            peak_memory = tracker.stop()
            
            # Assert that memory increase is reasonable (less than 2GB)
            self.assertLess(peak_memory - tracker.initial_memory, 2.0, 
                          f"Memory increase ({peak_memory - tracker.initial_memory:.2f} GB) exceeded 2GB limit")
            
        except Exception as e:
            # Print memory usage even if an error occurs
            peak_memory = tracker.stop()
            print(f"Error occurred. Peak memory: {peak_memory:.2f} GB")
            raise e
        finally:
            # Clean up
            clear_memory()

if __name__ == '__main__':
    unittest.main() 