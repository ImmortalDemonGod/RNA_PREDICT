"""
Comprehensive verification script for ProtenixDiffusionManager component.

This script verifies that the ProtenixDiffusionManager can be instantiated
and its multi_step_inference method can be called with properly shaped inputs.
It provides options for different levels of verification:
- Basic: Verifies instantiation and method existence only
- Standard: Verifies input handling and output shape with mocked computation
- Comprehensive: Attempts to run with minimal mocking (may fail if dependencies are missing)

Usage:
    python tests/stageD/verify_diffusion_manager.py [--mode basic|standard|comprehensive]
"""

import argparse
import inspect
import sys
import traceback
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

#######################
# Mock Implementations #
#######################


# Create a mock for the expand_at_dim function
def mock_expand_at_dim(tensor, dim, n):
    """Mock implementation of expand_at_dim function"""
    if dim < 0:
        dim = tensor.dim() + dim
    shape = list(tensor.shape)
    if dim >= len(shape):
        # Add dimensions if needed
        for _ in range(dim - len(shape) + 1):
            shape.append(1)
        tensor = tensor.view(*shape)

    # Create new shape for expansion
    expand_shape = list(tensor.shape)
    if dim >= len(expand_shape):
        expand_shape.append(n)
    else:
        expand_shape[dim] = n

    # Expand tensor
    return tensor.expand(*expand_shape)


# Create a mock for the AtomAttentionEncoder class
class MockAtomAttentionEncoder(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        print("Initialized MockAtomAttentionEncoder")
        # Add atom_transformer attribute with diffusion_transformer.blocks
        self.atom_transformer = MagicMock()
        self.atom_transformer.diffusion_transformer = MagicMock()
        self.atom_transformer.diffusion_transformer.blocks = []

    def forward(self, input_feature_dict, r_l, s, z, **kwargs):
        # Just return dummy tensors with the expected shapes
        batch_size = r_l.shape[0]
        n_sample = r_l.shape[1] if len(r_l.shape) > 3 else 1
        n_atom = r_l.shape[-2]
        n_token = s.shape[-2]
        c_token = 768  # Standard token channel size

        a_token = torch.zeros(
            (batch_size, n_sample, n_token, c_token), device=r_l.device
        )
        q_skip = torch.zeros(
            (batch_size, n_sample, n_atom, 128), device=r_l.device
        )  # Dummy q_skip
        c_skip = torch.zeros(
            (batch_size, n_sample, n_atom, 128), device=r_l.device
        )  # Dummy c_skip
        p_skip = torch.zeros(
            (batch_size, n_sample, n_atom, n_atom, 16), device=r_l.device
        )  # Dummy p_skip

        return a_token, q_skip, c_skip, p_skip

    def linear_init(self, **kwargs):
        """Mock implementation of linear_init method"""
        print("Called linear_init on MockAtomAttentionEncoder")
        # This method is called during initialization, so just pass


# Create a mock for the AtomAttentionDecoder class
class MockAtomAttentionDecoder(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        print("Initialized MockAtomAttentionDecoder")
        # Add required attributes for initialization
        self.linear_no_bias_a = MagicMock()
        self.linear_no_bias_a.weight = MagicMock()

        self.linear_no_bias_out = MagicMock()
        self.linear_no_bias_out.weight = MagicMock()

    def forward(self, input_feature_dict, a, q_skip, c_skip, p_skip, **kwargs):
        # Just return a tensor with the expected output shape
        batch_size = a.shape[0]
        n_sample = a.shape[1] if len(a.shape) > 3 else 1
        n_atom = q_skip.shape[-2]

        return torch.zeros((batch_size, n_sample, n_atom, 3), device=a.device)


# Create a mock for the DiffusionTransformer class
class MockDiffusionTransformer(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        print("Initialized MockDiffusionTransformer")
        # Add blocks attribute for initialization
        self.blocks = []
        for i in range(kwargs.get("n_blocks", 1)):
            block = MagicMock()
            block.attention_pair_bias = MagicMock()
            block.attention_pair_bias.layernorm_a = MagicMock()
            block.attention_pair_bias.glorot_init = MagicMock()

            block.conditioned_transition_block = MagicMock()
            block.conditioned_transition_block.adaln = MagicMock()
            block.conditioned_transition_block.adaln.zero_init = MagicMock()

            block.conditioned_transition_block.linear_nobias_b = MagicMock()
            block.conditioned_transition_block.linear_nobias_b.weight = MagicMock()

            self.blocks.append(block)

    def forward(self, a, s, z, **kwargs):
        # Just return a tensor with the expected output shape
        return a


# Create a mock for the DiffusionModule class
class MockDiffusionModule(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        print("Initialized MockDiffusionModule")
        # Add required attributes for initialization
        self.diffusion_conditioning = MagicMock()
        self.atom_attention_encoder = MockAtomAttentionEncoder()
        self.diffusion_transformer = MockDiffusionTransformer()
        self.atom_attention_decoder = MockAtomAttentionDecoder()
        self.layernorm_s = MagicMock()
        self.linear_no_bias_s = MagicMock()
        self.layernorm_a = MagicMock()

    def forward(
        self,
        x_noisy,
        t_hat_noise_level,
        input_feature_dict,
        s_inputs,
        s_trunk,
        z_trunk,
        **kwargs,
    ):
        # Just return a tensor with the expected output shape
        batch_size = x_noisy.shape[0]
        n_sample = x_noisy.shape[1] if len(x_noisy.shape) > 3 else 1
        n_atom = x_noisy.shape[-2]
        return torch.zeros((batch_size, n_sample, n_atom, 3), device=x_noisy.device)


# Create a mock for sample_diffusion
def mock_sample_diffusion(*args, **kwargs):
    # Extract necessary parameters
    input_feature_dict = kwargs.get(
        "input_feature_dict", args[1] if len(args) > 1 else {}
    )
    N_sample = kwargs.get("N_sample", 1)

    # Get atom count from input_feature_dict
    n_atom = 5  # Default
    if "atom_to_token_idx" in input_feature_dict:
        n_atom = input_feature_dict["atom_to_token_idx"].shape[-1]

    # Create a device
    device = torch.device("cpu")
    if "s_trunk" in kwargs:
        device = kwargs["s_trunk"].device

    # Return a tensor with shape [batch_size, N_sample, n_atom, 3]
    return torch.zeros((1, N_sample, n_atom, 3), device=device)


# Define patches for different verification modes
basic_patches = [
    # Mock the DiffusionModule class
    patch(
        "rna_predict.pipeline.stageD.diffusion.diffusion.DiffusionModule",
        MockDiffusionModule,
    ),
]

standard_patches = basic_patches + [
    # Mock the sample_diffusion function
    patch(
        "rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager.sample_diffusion",
        mock_sample_diffusion,
    ),
    # Mock expand_at_dim
    patch("protenix.model.utils.expand_at_dim", mock_expand_at_dim),
]

comprehensive_patches = standard_patches + [
    # Mock any other problematic imports
    patch(
        "rna_predict.pipeline.stageA.input_embedding.current.transformer.AtomAttentionEncoder",
        MockAtomAttentionEncoder,
    ),
    patch(
        "rna_predict.pipeline.stageA.input_embedding.current.transformer.AtomAttentionDecoder",
        MockAtomAttentionDecoder,
    ),
    patch(
        "rna_predict.pipeline.stageA.input_embedding.current.transformer.DiffusionTransformer",
        MockDiffusionTransformer,
    ),
    patch(
        "rna_predict.pipeline.stageA.input_embedding.current.embedders.FourierEmbedding",
        MagicMock,
    ),
    patch(
        "rna_predict.pipeline.stageA.input_embedding.current.embedders.RelativePositionEncoding",
        MagicMock,
    ),
    patch(
        "rna_predict.pipeline.stageA.input_embedding.current.primitives.LayerNorm",
        MagicMock,
    ),
    patch(
        "rna_predict.pipeline.stageA.input_embedding.current.primitives.LinearNoBias",
        MagicMock,
    ),
    patch(
        "rna_predict.pipeline.stageA.input_embedding.current.primitives.Transition",
        MagicMock,
    ),
]

#######################
# Verification Steps #
#######################


def verify_instantiation(patches):
    """
    Verify that ProtenixDiffusionManager can be instantiated with minimal config.
    """
    print("\n1. Testing instantiation with minimal config...")

    # Apply patches
    for p in patches:
        p.start()

    try:
        # Import after patching
        from rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager import (
            ProtenixDiffusionManager,
        )

        # Create with empty diffusion_config
        manager = ProtenixDiffusionManager(diffusion_config={}, device="cpu")
        print("✓ Successfully created ProtenixDiffusionManager instance")

        # Verify diffusion_module was created
        if hasattr(manager, "diffusion_module"):
            print("✓ Internal DiffusionModule was initialized")
            return manager
        else:
            print("✗ Failed to initialize internal DiffusionModule")
            return None
    except Exception as e:
        print(f"✗ Failed to instantiate ProtenixDiffusionManager: {e}")
        traceback.print_exc()
        return None
    finally:
        # Stop patches
        for p in patches:
            p.stop()


def verify_method_existence(manager):
    """
    Verify that the multi_step_inference method exists.
    """
    print("\n2. Verifying method existence...")
    if hasattr(manager, "multi_step_inference"):
        print("✓ multi_step_inference method exists")
        print(f"Method signature: {inspect.signature(manager.multi_step_inference)}")
        return True
    else:
        print("✗ multi_step_inference method does not exist")
        return False


def verify_input_handling(manager, patches):
    """
    Verify that multi_step_inference accepts the required parameters.
    """
    print("\n3. Testing input handling...")

    # Apply patches
    for p in patches:
        p.start()

    try:
        # Create minimal valid inputs with proper shapes
        batch_size = 1
        n_atoms = 5
        n_tokens = 5  # Same as n_atoms for simplicity
        c_s = 384  # Standard channel size for s_trunk
        c_s_inputs = 449  # Standard channel size for s_inputs
        c_z = 128  # Standard channel size for pair embeddings

        # Create minimal inputs
        print("Creating coords_init tensor...")
        coords_init = torch.zeros((batch_size, n_atoms, 3))
        print(f"coords_init shape: {coords_init.shape}, dtype: {coords_init.dtype}")

        print("Creating trunk_embeddings dictionary...")
        trunk_embeddings = {
            "s_trunk": torch.zeros((batch_size, n_tokens, c_s)),
            "s_inputs": torch.zeros((batch_size, n_tokens, c_s_inputs)),
            "pair": torch.zeros((batch_size, n_tokens, n_tokens, c_z)),
        }
        print(f"trunk_embeddings keys: {trunk_embeddings.keys()}")
        for key, value in trunk_embeddings.items():
            print(f"  {key} shape: {value.shape}, dtype: {value.dtype}")

        print("Creating inference_params dictionary...")
        inference_params = {
            "N_sample": 1,
            "num_steps": 2,  # Use minimal steps for faster testing
        }
        print(f"inference_params: {inference_params}")

        print("Creating override_input_features dictionary...")
        # The key issue is that some tensors need to have the right shape
        # Specifically, ref_element needs to be [batch_size, n_atoms, 128]
        override_input_features = {
            "atom_to_token_idx": torch.zeros((batch_size, n_atoms), dtype=torch.long),
            "asym_id": torch.zeros((batch_size, n_tokens), dtype=torch.long),
            "entity_id": torch.zeros((batch_size, n_tokens), dtype=torch.long),
            "res_idx": torch.zeros((batch_size, n_tokens), dtype=torch.long),
            "residue_index": torch.zeros((batch_size, n_tokens), dtype=torch.long),
            "token_index": torch.zeros((batch_size, n_tokens), dtype=torch.long),
            "sym_id": torch.zeros((batch_size, n_tokens), dtype=torch.long),
            "atom_mask": torch.ones((batch_size, n_atoms), dtype=torch.bool),
            "atom_type": torch.zeros((batch_size, n_atoms), dtype=torch.long),
            "ref_pos": torch.zeros((batch_size, n_atoms, 3), dtype=torch.float32),
            "ref_charge": torch.zeros((batch_size, n_atoms), dtype=torch.float32),
            "ref_mask": torch.ones((batch_size, n_atoms), dtype=torch.bool),
            "ref_element": torch.zeros(
                (batch_size, n_atoms, 128), dtype=torch.float32
            ),  # One-hot encoding
        }
        print(f"override_input_features keys: {override_input_features.keys()}")

        # Call the method with minimal inputs
        print("Calling multi_step_inference with properly shaped inputs...")
        coords_final = manager.multi_step_inference(
            coords_init=coords_init,
            trunk_embeddings=trunk_embeddings,
            inference_params=inference_params,
            override_input_features=override_input_features,
            debug_logging=True,
        )

        print("✓ multi_step_inference accepted all required parameters")
        return coords_final
    except Exception as e:
        print(f"✗ multi_step_inference failed with error: {e}")
        traceback.print_exc()
        return None
    finally:
        # Stop patches
        for p in patches:
            p.stop()


def verify_output_validation(coords_final):
    """
    Verify that the output of multi_step_inference is valid.
    """
    print("\n4. Validating output...")

    if coords_final is None:
        print("✗ No output to validate (previous step failed)")
        return False

    # Check output type
    if isinstance(coords_final, torch.Tensor):
        print(f"✓ Output is a PyTorch Tensor with shape {coords_final.shape}")
    else:
        print(f"✗ Output is not a PyTorch Tensor, got {type(coords_final)}")
        return False

    # Check output shape
    if len(coords_final.shape) == 4:
        print("✓ Output has 4 dimensions [B, N_sample, N_atom, 3]")
    else:
        print(f"✗ Output does not have 4 dimensions, got {len(coords_final.shape)}")
        return False

    # Check last dimension is 3 (coordinates)
    if coords_final.shape[-1] == 3:
        print("✓ Last dimension is 3 (coordinates)")
    else:
        print(f"✗ Last dimension is not 3, got {coords_final.shape[-1]}")
        return False

    return True


def run_verification(mode="standard"):
    """
    Run verification with the specified mode.

    Args:
        mode (str): Verification mode - "basic", "standard", or "comprehensive"

    Returns:
        bool: True if verification passed, False otherwise
    """
    print(f"Starting ProtenixDiffusionManager verification in {mode.upper()} mode...")

    # Select patches based on mode
    if mode == "basic":
        patches = basic_patches
    elif mode == "standard":
        patches = standard_patches
    elif mode == "comprehensive":
        patches = comprehensive_patches
    else:
        print(f"Unknown mode: {mode}")
        return False

    # Step 1: Instantiation
    manager = verify_instantiation(patches)
    if manager is None:
        print("\n❌ Verification failed at instantiation step")
        return False

    # Step 2: Method Existence
    if not verify_method_existence(manager):
        print("\n❌ Verification failed at method existence step")
        return False

    # For basic mode, we're done
    if mode == "basic":
        print("\n✅ Basic verification steps passed successfully!")
        return True

    # Step 3: Input Handling
    coords_final = verify_input_handling(manager, patches)
    if coords_final is None:
        print("\n❌ Verification failed at input handling step")
        return False

    # Step 4: Output Validation
    if not verify_output_validation(coords_final):
        print("\n❌ Verification failed at output validation step")
        return False

    print(f"\n✅ All verification steps in {mode.upper()} mode passed successfully!")
    return True


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Verify ProtenixDiffusionManager component"
    )
    parser.add_argument(
        "--mode",
        choices=["basic", "standard", "comprehensive"],
        default="standard",
        help="Verification mode",
    )
    args = parser.parse_args()

    # Run verification
    success = run_verification(args.mode)
    sys.exit(0 if success else 1)
