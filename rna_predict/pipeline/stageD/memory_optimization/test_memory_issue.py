import torch
import unittest
import psutil
import os
from rna_predict.pipeline.stageD.diffusion.run_stageD_unified import run_stageD_diffusion

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024  # Convert to GB

def clear_memory():
    """Clear memory by running garbage collection"""
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

class TestMemoryIssue(unittest.TestCase):
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
                "num_steps": 20,  # Reduced from 100 to make testing faster
                "noise_schedule": "linear",
            },
            "c_s_inputs": 449,
            "c_z": 64,
            "c_atom": 128,
            "c_s": 384,
            "c_token": 832,
            "transformer": {"n_blocks": 4, "n_heads": 16}
        }

    def test_memory_usage(self):
        """Test that replicates the memory issue"""
        # Print initial memory usage
        initial_memory = get_memory_usage()
        print(f"Initial memory usage: {initial_memory:.2f} GB")
        
        try:
            # Run the diffusion process with memory tracking
            def memory_tracking_hook(module, input, output):
                current_memory = get_memory_usage()
                print(f"Memory after {module.__class__.__name__}: {current_memory:.2f} GB")
                return output
            
            # Register hooks for memory tracking
            from rna_predict.pipeline.stageA.input_embedding.current.transformer import AtomTransformer, AtomAttentionEncoder
            from rna_predict.pipeline.stageD.diffusion.components.diffusion_module import DiffusionModule
            
            # Track memory in key components
            for module_class in [AtomTransformer, AtomAttentionEncoder, DiffusionModule]:
                for name, module in module_class.__dict__.items():
                    if isinstance(module, torch.nn.Module):
                        module.register_forward_hook(memory_tracking_hook)
            
            # Run the diffusion process
            run_stageD_diffusion(
                partial_coords=self.partial_coords,
                trunk_embeddings=self.trunk_embeddings,
                diffusion_config=self.diffusion_config,
                mode="inference",
                device="cpu"
            )
            
            # Print final memory usage
            final_memory = get_memory_usage()
            print(f"Final memory usage: {final_memory:.2f} GB")
            print(f"Memory increase: {final_memory - initial_memory:.2f} GB")
            
            # Assert that memory usage is reasonable (less than 16GB)
            self.assertLess(final_memory, 16.0, 
                          f"Memory usage ({final_memory:.2f} GB) exceeded 16GB limit")
            
        except Exception as e:
            # Print memory usage even if an error occurs
            final_memory = get_memory_usage()
            print(f"Error occurred. Final memory usage: {final_memory:.2f} GB")
            print(f"Memory increase: {final_memory - initial_memory:.2f} GB")
            raise e
        finally:
            # Clean up
            clear_memory()

if __name__ == '__main__':
    unittest.main() 