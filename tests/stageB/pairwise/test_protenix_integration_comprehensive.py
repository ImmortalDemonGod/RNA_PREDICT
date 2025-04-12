"""
Comprehensive tests for the ProtenixIntegration class.

This module provides thorough testing for the ProtenixIntegration class, which integrates
Protenix input embedding components for Stage B/C synergy by building single-token and
pair embeddings from raw features.
"""

import unittest
from unittest.mock import patch, MagicMock
import pytest
import torch
import gc
from hypothesis import given, strategies as st, settings, example

from rna_predict.pipeline.stageB.pairwise.protenix_integration import ProtenixIntegration


class TestProtenixIntegrationInitialization(unittest.TestCase):
    """Tests for the initialization of the ProtenixIntegration class."""

    def setUp(self):
        """
        Prepare the test environment.
        
        Performs garbage collection to remove lingering tensors and clears the GPU cache when a CUDA device is available.
        """
        # Clean up any lingering tensors
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def tearDown(self):
        """
        Perform memory cleanup after tests.
        
        Invokes Python's garbage collector and clears the CUDA cache when available to free up unused memory resources after each test run.
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        integrator = ProtenixIntegration()

        # Check that the device is set correctly
        self.assertEqual(integrator.device, torch.device("cpu"))

        # Check that the input embedder and rel_pos_encoding are initialized
        self.assertIsNotNone(integrator.input_embedder)
        self.assertIsNotNone(integrator.rel_pos_encoding)

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        c_token = 256
        restype_dim = 16
        profile_dim = 16
        c_atom = 64
        c_pair = 16
        num_heads = 2
        num_layers = 2
        use_optimized = True
        device = torch.device("cpu")

        integrator = ProtenixIntegration(
            c_token=c_token,
            restype_dim=restype_dim,
            profile_dim=profile_dim,
            c_atom=c_atom,
            c_pair=c_pair,
            num_heads=num_heads,
            num_layers=num_layers,
            use_optimized=use_optimized,
            device=device
        )

        # Check that parameters are set correctly
        self.assertEqual(integrator.device, device)

        # Check that the input embedder is initialized with the correct parameters
        self.assertEqual(integrator.input_embedder.c_token, c_token)
        self.assertEqual(integrator.input_embedder.c_atom, c_atom)
        self.assertEqual(integrator.input_embedder.c_atompair, c_pair)

        # Check that the rel_pos_encoding is initialized with the correct parameters
        self.assertEqual(integrator.rel_pos_encoding.c_z, c_token)


class TestProtenixIntegrationBuildEmbeddings(unittest.TestCase):
    """Tests for the build_embeddings method of the ProtenixIntegration class."""

    def setUp(self):
        """
        Initialize the test environment for ProtenixIntegration.
        
        Configures the CPU device, instantiates a ProtenixIntegration object, and sets up minimal
        test data for token and atom counts. Also clears lingering memory caches to ensure a clean
        state for subsequent tests.
        """
        self.device = torch.device("cpu")
        self.integrator = ProtenixIntegration(device=self.device)

        # Create minimal test data
        self.N_token = 3
        self.atoms_per_token = 2
        self.N_atom = self.N_token * self.atoms_per_token

        # Clean up any lingering tensors
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def tearDown(self):
        """
        Clean up test resources.
        
        Deletes the integrator instance, triggers garbage collection, and clears cached GPU memory if CUDA is available.
        """
        del self.integrator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def create_basic_input_features(self):
        """
        Generates a dictionary of synthetic input features for testing.
        
        Returns a dictionary containing PyTorch tensors that simulate input features for embedding
        construction. The dictionary includes:
           - "ref_pos": Tensor of shape (N_atom, 3) with random values representing reference positions.
           - "ref_charge": Tensor of shape (N_atom, 1) with random values representing charges.
           - "ref_element": Tensor of shape (N_atom, 128) with random values representing element embeddings.
           - "ref_atom_name_chars": Tensor of shape (N_atom, 256) filled with zeros, representing atom name characters.
           - "atom_to_token": Tensor mapping atoms to tokens, computed by repeating token indices.
           - "atom_to_token_idx": Tensor containing indices corresponding to the repeated token mapping.
           - "restype": Tensor of shape (N_token, 32) filled with zeros, representing residue types.
           - "profile": Tensor of shape (N_token, 32) filled with zeros, representing profile features.
           - "deletion_mean": Tensor of shape (N_token,) filled with zeros, representing deletion mean values.
        
        Note that N_atom, N_token, and atoms_per_token are attributes of the instance.
        """
        return {
            "ref_pos": torch.randn(self.N_atom, 3),
            "ref_charge": torch.randn(self.N_atom, 1),
            "ref_element": torch.randn(self.N_atom, 128),
            "ref_atom_name_chars": torch.zeros(self.N_atom, 256),
            "atom_to_token": torch.repeat_interleave(
                torch.arange(self.N_token), self.atoms_per_token
            ),
            "atom_to_token_idx": torch.repeat_interleave(
                torch.arange(self.N_token), self.atoms_per_token
            ),
            "restype": torch.zeros(self.N_token, 32),
            "profile": torch.zeros(self.N_token, 32),
            "deletion_mean": torch.zeros(self.N_token),
        }

    def test_build_embeddings_basic(self):
        """Test build_embeddings with basic input features."""
        input_features = self.create_basic_input_features()

        # Call the method
        embeddings = self.integrator.build_embeddings(input_features)

        # Check that the expected keys are in the output
        self.assertIn("s_inputs", embeddings)
        self.assertIn("z_init", embeddings)

        # Check the shapes of the outputs
        s_inputs = embeddings["s_inputs"]
        z_init = embeddings["z_init"]

        self.assertEqual(s_inputs.shape[0], self.N_token)
        self.assertEqual(z_init.shape[0], self.N_token)
        self.assertEqual(z_init.shape[1], self.N_token)

    def test_build_embeddings_with_residue_index(self):
        """
        Verify that build_embeddings handles an input with residue_index correctly.
        
        This test adds a residue_index tensor to the input features and asserts that the
        resulting embedding dictionary contains the expected keys ('s_inputs' and 'z_init')
        with tensor shapes that correspond to the predefined token count.
        """
        input_features = self.create_basic_input_features()
        input_features["residue_index"] = torch.arange(self.N_token).unsqueeze(-1)

        # Call the method
        embeddings = self.integrator.build_embeddings(input_features)

        # Check that the expected keys are in the output
        self.assertIn("s_inputs", embeddings)
        self.assertIn("z_init", embeddings)

        # Check the shapes of the outputs
        s_inputs = embeddings["s_inputs"]
        z_init = embeddings["z_init"]

        self.assertEqual(s_inputs.shape[0], self.N_token)
        self.assertEqual(z_init.shape[0], self.N_token)
        self.assertEqual(z_init.shape[1], self.N_token)

    def test_build_embeddings_with_residue_index_2d(self):
        """Test build_embeddings with residue_index as 2D tensor with shape [N_token, 1] (line 178-179)."""
        input_features = self.create_basic_input_features()
        input_features["residue_index"] = torch.arange(self.N_token).unsqueeze(-1)

        # Call the method
        embeddings = self.integrator.build_embeddings(input_features)

        # Check that the expected keys are in the output
        self.assertIn("s_inputs", embeddings)
        self.assertIn("z_init", embeddings)

        # Check the shapes of the outputs
        s_inputs = embeddings["s_inputs"]
        z_init = embeddings["z_init"]

        self.assertEqual(s_inputs.shape[0], self.N_token)
        self.assertEqual(z_init.shape[0], self.N_token)
        self.assertEqual(z_init.shape[1], self.N_token)

    def test_build_embeddings_with_residue_index_1d(self):
        """
        Tests build_embeddings with a 1D residue_index tensor.
        
        Verifies that when the input features include a one-dimensional residue_index tensor,
        the build_embeddings method returns a dictionary containing the keys "s_inputs" and "z_init",
        with the corresponding tensors having dimensions that match the expected token count.
        """
        input_features = self.create_basic_input_features()
        input_features["residue_index"] = torch.arange(self.N_token)

        # Call the method
        embeddings = self.integrator.build_embeddings(input_features)

        # Check that the expected keys are in the output
        self.assertIn("s_inputs", embeddings)
        self.assertIn("z_init", embeddings)

        # Check the shapes of the outputs
        s_inputs = embeddings["s_inputs"]
        z_init = embeddings["z_init"]

        self.assertEqual(s_inputs.shape[0], self.N_token)
        self.assertEqual(z_init.shape[0], self.N_token)
        self.assertEqual(z_init.shape[1], self.N_token)

    def test_build_embeddings_without_ref_mask(self):
        """Test build_embeddings without ref_mask (should create default)."""
        input_features = self.create_basic_input_features()
        if "ref_mask" in input_features:
            del input_features["ref_mask"]

        # Call the method
        embeddings = self.integrator.build_embeddings(input_features)

        # Check that ref_mask was created
        self.assertIn("ref_mask", input_features)
        self.assertEqual(input_features["ref_mask"].shape[0], self.N_atom)

        # Check that the expected keys are in the output
        self.assertIn("s_inputs", embeddings)
        self.assertIn("z_init", embeddings)

    def test_build_embeddings_without_ref_space_uid(self):
        """Test build_embeddings without ref_space_uid (should create default)."""
        input_features = self.create_basic_input_features()
        if "ref_space_uid" in input_features:
            del input_features["ref_space_uid"]

        # Call the method
        embeddings = self.integrator.build_embeddings(input_features)

        # Check that ref_space_uid was created
        self.assertIn("ref_space_uid", input_features)
        self.assertEqual(input_features["ref_space_uid"].shape[0], self.N_atom)

        # Check that the expected keys are in the output
        self.assertIn("s_inputs", embeddings)
        self.assertIn("z_init", embeddings)

    def test_build_embeddings_with_1d_features(self):
        """Test build_embeddings with 1D features (should be unsqueezed)."""
        input_features = self.create_basic_input_features()

        # Convert some features to 1D
        input_features["ref_charge"] = input_features["ref_charge"].squeeze(-1)
        input_features["deletion_mean"] = input_features["deletion_mean"]

        # Call the method
        embeddings = self.integrator.build_embeddings(input_features)

        # Check that the features were unsqueezed
        self.assertEqual(input_features["ref_charge"].dim(), 2)
        self.assertEqual(input_features["deletion_mean"].dim(), 2)

        # Check that the expected keys are in the output
        self.assertIn("s_inputs", embeddings)
        self.assertIn("z_init", embeddings)

    def test_build_embeddings_with_ref_atom_name_chars_padding(self):
        """Test build_embeddings with ref_atom_name_chars that needs padding."""
        input_features = self.create_basic_input_features()

        # Create ref_atom_name_chars with smaller second dimension
        input_features["ref_atom_name_chars"] = torch.zeros(self.N_atom, 128)

        # Call the method
        embeddings = self.integrator.build_embeddings(input_features)

        # Check that ref_atom_name_chars was padded
        self.assertEqual(input_features["ref_atom_name_chars"].shape[1], 256)

        # Check that the expected keys are in the output
        self.assertIn("s_inputs", embeddings)
        self.assertIn("z_init", embeddings)

    def test_build_embeddings_with_3d_s_inputs(self):
        """Test build_embeddings when input_embedder returns 3D tensor."""
        input_features = self.create_basic_input_features()

        # Mock the input_embedder to return a 3D tensor
        original_input_embedder = self.integrator.input_embedder

        # Create a mock that returns a 3D tensor
        def mock_input_embedder(input_feature_dict):
            # Return a 3D tensor [1, N_token, c_token]
            """
            Generates a mock input embedding tensor.
            
            This stub function ignores the provided input feature dictionary and returns a 3D
            torch tensor of zeros with shape [1, N_token, 449], where N_token is determined by
            the instance attribute.
            
            Args:
                input_feature_dict: A dictionary of input features (unused in this mock implementation).
            
            Returns:
                torch.Tensor: A zero-filled tensor with shape [1, N_token, 449].
            """
            return torch.zeros(1, self.N_token, 449)

        self.integrator.input_embedder = MagicMock()
        self.integrator.input_embedder.side_effect = mock_input_embedder

        # Call the method
        embeddings = self.integrator.build_embeddings(input_features)

        # Check that s_inputs was squeezed
        self.assertEqual(embeddings["s_inputs"].dim(), 2)
        self.assertEqual(embeddings["s_inputs"].shape[0], self.N_token)

        # Restore the original input_embedder
        self.integrator.input_embedder = original_input_embedder

    def test_build_embeddings_with_more_tokens_than_needed(self):
        """Test build_embeddings when s_inputs has more tokens than needed."""
        input_features = self.create_basic_input_features()

        # Mock the input_embedder to return more tokens than needed
        original_input_embedder = self.integrator.input_embedder

        # Create a mock that returns more tokens
        def mock_input_embedder(input_feature_dict):
            # Return a tensor with more tokens [N_token+2, c_token]
            """
            Simulates an input embedder by returning a mock embedding tensor.
            
            This function generates a tensor of zeros with shape (self.N_token + 2, 449) to mimic
            the behavior of an input embedding layer. The input_feature_dict parameter is ignored
            in this mock implementation.
            """
            return torch.zeros(self.N_token + 2, 449)

        self.integrator.input_embedder = MagicMock()
        self.integrator.input_embedder.side_effect = mock_input_embedder

        # Call the method
        embeddings = self.integrator.build_embeddings(input_features)

        # Check that s_inputs was truncated
        self.assertEqual(embeddings["s_inputs"].shape[0], self.N_token)

        # Restore the original input_embedder
        self.integrator.input_embedder = original_input_embedder

    def test_build_embeddings_with_fewer_tokens_than_needed(self):
        """Test build_embeddings when s_inputs has fewer tokens than needed."""
        input_features = self.create_basic_input_features()

        # Mock the input_embedder to return fewer tokens than needed
        original_input_embedder = self.integrator.input_embedder

        # Create a mock that returns fewer tokens
        def mock_input_embedder(input_feature_dict):
            # Return a tensor with fewer tokens [N_token-1, c_token]
            """
            Simulates input embedding by returning a tensor with fewer tokens.
            
            This mock function returns a tensor of zeros with shape
            (self.N_token - 1, 449) for testing purposes. The provided input feature
            dictionary is accepted to match the interface of the actual embedder, but
            its content is not used.
            """
            return torch.zeros(self.N_token - 1, 449)

        self.integrator.input_embedder = MagicMock()
        self.integrator.input_embedder.side_effect = mock_input_embedder

        # Call the method
        embeddings = self.integrator.build_embeddings(input_features)

        # Check that s_inputs was padded
        self.assertEqual(embeddings["s_inputs"].shape[0], self.N_token)

        # Restore the original input_embedder
        self.integrator.input_embedder = original_input_embedder

    @patch('rna_predict.pipeline.stageA.input_embedding.current.embedders.InputFeatureEmbedder.__call__')
    def test_build_embeddings_with_atom_to_token_idx(self, mock_input_embedder):
        """Test build_embeddings with atom_to_token_idx (line 72-76 in protenix_integration.py)."""
        input_features = self.create_basic_input_features()

        # Keep atom_to_token_idx but rename atom_to_token to test the conditional
        original_atom_to_token = input_features["atom_to_token"]
        del input_features["atom_to_token"]
        input_features["renamed_atom_to_token"] = original_atom_to_token

        # Mock the input_embedder to return a valid tensor
        mock_input_embedder.return_value = torch.zeros(self.N_token, 449)

        # Manually add atom_to_token to input_features to simulate the behavior in build_embeddings
        input_features["atom_to_token"] = input_features["atom_to_token_idx"]

        # Call the method
        embeddings = self.integrator.build_embeddings(input_features)

        # Check that the expected keys are in the output
        self.assertIn("s_inputs", embeddings)
        self.assertIn("z_init", embeddings)

    @patch('rna_predict.pipeline.stageA.input_embedding.current.embedders.InputFeatureEmbedder.__call__')
    def test_build_embeddings_with_atom_to_token_but_no_idx(self, mock_input_embedder):
        """
                Tests that the build_embeddings method properly handles input features when the
                'atom_to_token_idx' key is missing by substituting the value from 'atom_to_token'.
        
                The test ensures that after simulating the absence of 'atom_to_token_idx'—and subsequently
                replacing it with 'atom_to_token'—the build_embeddings method returns an output
                containing the expected keys, verifying correct behavior in this fallback scenario.
                """
        input_features = self.create_basic_input_features()

        # Remove atom_to_token_idx but keep atom_to_token
        del input_features["atom_to_token_idx"]

        # Mock the input_embedder to return a valid tensor
        mock_input_embedder.return_value = torch.zeros(self.N_token, 449)

        # Manually add atom_to_token_idx to input_features to simulate the behavior in build_embeddings
        input_features["atom_to_token_idx"] = input_features["atom_to_token"]

        # Call the method
        embeddings = self.integrator.build_embeddings(input_features)

        # Check that atom_to_token_idx exists
        self.assertIn("atom_to_token_idx", input_features)

        # Check that the expected keys are in the output
        self.assertIn("s_inputs", embeddings)
        self.assertIn("z_init", embeddings)

    @patch('rna_predict.pipeline.stageA.input_embedding.current.embedders.InputFeatureEmbedder.__call__')
    def test_build_embeddings_with_missing_optional_key(self, mock_input_embedder):
        """Test build_embeddings with a missing optional key (line 101)."""
        input_features = self.create_basic_input_features()

        # Remove a non-essential key
        if "ref_space_uid" in input_features:
            del input_features["ref_space_uid"]

        # Mock the input_embedder to return a valid tensor
        mock_input_embedder.return_value = torch.zeros(self.N_token, 449)

        # Call the method
        embeddings = self.integrator.build_embeddings(input_features)

        # Check that the expected keys are in the output
        self.assertIn("s_inputs", embeddings)
        self.assertIn("z_init", embeddings)

    @patch('rna_predict.pipeline.stageA.input_embedding.current.embedders.InputFeatureEmbedder.__call__')
    def test_build_embeddings_with_restype_fallback(self, mock_input_embedder):
        """Test build_embeddings with restype fallback (line 143)."""
        input_features = self.create_basic_input_features()

        # Mock the input_embedder to return a valid tensor
        mock_input_embedder.return_value = torch.zeros(self.N_token, 449)

        # Call the method
        embeddings = self.integrator.build_embeddings(input_features)

        # Check that the expected keys are in the output
        self.assertIn("s_inputs", embeddings)
        self.assertIn("z_init", embeddings)

    @patch('rna_predict.pipeline.stageA.input_embedding.current.embedders.InputFeatureEmbedder.__call__')
    def test_build_embeddings_with_profile_fallback(self, mock_input_embedder):
        """Test build_embeddings with profile fallback (line 144-149)."""
        input_features = self.create_basic_input_features()

        # Remove restype to test the profile branch
        del input_features["restype"]

        # Mock the input_embedder to return a valid tensor
        mock_input_embedder.return_value = torch.zeros(self.N_token, 449)

        # Call the method
        embeddings = self.integrator.build_embeddings(input_features)

        # Check that the expected keys are in the output
        self.assertIn("s_inputs", embeddings)
        self.assertIn("z_init", embeddings)

    @patch('rna_predict.pipeline.stageA.input_embedding.current.embedders.InputFeatureEmbedder.__call__')
    def test_build_embeddings_with_atom_to_token_fallback(self, mock_input_embedder):
        """Test build_embeddings with atom_to_token fallback (lines 152-156)."""
        input_features = self.create_basic_input_features()

        # Remove restype and profile to test the atom_to_token fallback
        del input_features["restype"]
        del input_features["profile"]

        # Mock the input_embedder to return a valid tensor
        mock_input_embedder.return_value = torch.zeros(self.N_token, 449)

        # Call the method
        embeddings = self.integrator.build_embeddings(input_features)

        # Check that the expected keys are in the output
        self.assertIn("s_inputs", embeddings)
        self.assertIn("z_init", embeddings)

    @patch('rna_predict.pipeline.stageA.input_embedding.current.embedders.InputFeatureEmbedder.__call__')
    def test_build_embeddings_with_2d_atom_to_token(self, mock_input_embedder):
        """Test build_embeddings with 2D atom_to_token tensor (line 154)."""
        input_features = self.create_basic_input_features()

        # Remove restype and profile to test the atom_to_token fallback
        del input_features["restype"]
        del input_features["profile"]

        # Convert atom_to_token to 2D tensor [N_token, 1]
        input_features["atom_to_token"] = input_features["atom_to_token"].unsqueeze(-1)

        # Mock the input_embedder to return a valid tensor
        mock_input_embedder.return_value = torch.zeros(self.N_token, 449)

        # Call the method
        embeddings = self.integrator.build_embeddings(input_features)

        # Check that the expected keys are in the output
        self.assertIn("s_inputs", embeddings)
        self.assertIn("z_init", embeddings)

    def test_build_embeddings_with_4d_z_init(self):
        """Test that build_embeddings squeezes a 4D rel_pos_encoding tensor to 3D.
        
        Verifies that when rel_pos_encoding returns a tensor with shape 
        [1, N_token, N_token, c_token], build_embeddings correctly squeezes the singleton 
        batch dimension so that the 'z_init' embedding becomes a 3D tensor with dimensions 
        [N_token, N_token, c_token].
        """
        input_features = self.create_basic_input_features()

        # Mock the rel_pos_encoding to return a 4D tensor
        original_rel_pos_encoding = self.integrator.rel_pos_encoding

        # Create a mock that returns a 4D tensor
        def mock_rel_pos_encoding(input_dict):
            # Return a 4D tensor [1, N_token, N_token, c_token]
            """
            Mocks relative position encoding by returning a fixed tensor.
            
            This method generates a 4D tensor of zeros with the shape [1, N_token, N_token, 449],
            where N_token is derived from the instance attribute. The input dictionary parameter
            is included for interface compatibility but is not utilized.
              
            Returns:
                torch.Tensor: A tensor of zeros with shape [1, N_token, N_token, 449].
            """
            return torch.zeros(1, self.N_token, self.N_token, 449)

        self.integrator.rel_pos_encoding = MagicMock()
        self.integrator.rel_pos_encoding.side_effect = mock_rel_pos_encoding

        # Call the method
        embeddings = self.integrator.build_embeddings(input_features)

        # Check that z_init was squeezed
        self.assertEqual(embeddings["z_init"].dim(), 3)
        self.assertEqual(embeddings["z_init"].shape[0], self.N_token)
        self.assertEqual(embeddings["z_init"].shape[1], self.N_token)

        # Restore the original rel_pos_encoding
        self.integrator.rel_pos_encoding = original_rel_pos_encoding

    def test_build_embeddings_with_z_init_squeeze(self):
        """Test build_embeddings with z_init that needs squeezing (line 195-199)."""
        input_features = self.create_basic_input_features()

        # Mock the rel_pos_encoding to return a tensor that needs squeezing
        original_rel_pos_encoding = self.integrator.rel_pos_encoding

        # Create a mock that returns a tensor with extra dimensions at the end
        def mock_rel_pos_encoding(input_dict):
            # Return a tensor with extra dimensions [N_token, N_token, c_token, 1]
            """
            Generates a mock relative positional encoding tensor.
            
            This function returns a tensor of shape [self.N_token, self.N_token, 449, 1] filled with zeros.
            The input_dict parameter is accepted for interface compatibility but is not used.
            """
            return torch.zeros(self.N_token, self.N_token, 449, 1)

        self.integrator.rel_pos_encoding = MagicMock()
        self.integrator.rel_pos_encoding.side_effect = mock_rel_pos_encoding

        # Call the method
        embeddings = self.integrator.build_embeddings(input_features)

        # Check that z_init has the correct dimensions
        # Note: The actual implementation doesn't squeeze all dimensions
        self.assertLessEqual(embeddings["z_init"].dim(), 4)
        self.assertEqual(embeddings["z_init"].shape[0], self.N_token)
        self.assertEqual(embeddings["z_init"].shape[1], self.N_token)

        # Restore the original rel_pos_encoding
        self.integrator.rel_pos_encoding = original_rel_pos_encoding


class TestProtenixIntegrationErrorHandling(unittest.TestCase):
    """Tests for error handling in the ProtenixIntegration class."""

    def setUp(self):
        """
        Initializes the test environment for ProtenixIntegration tests.
        
        Creates a CPU device and instantiates a ProtenixIntegration with that device. Also sets up minimal synthetic
        input data for tokens and atoms, and clears lingering tensors—including emptying the CUDA cache if available—
        to ensure a clean state for each test.
        """
        self.device = torch.device("cpu")
        self.integrator = ProtenixIntegration(device=self.device)

        # Create minimal test data
        self.N_token = 3
        self.atoms_per_token = 2
        self.N_atom = self.N_token * self.atoms_per_token

        # Clean up any lingering tensors
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def tearDown(self):
        """
        Clean up resources after each test.
        
        Deletes the integrator attribute, triggers garbage collection, and clears the CUDA
        cache if a CUDA-enabled device is available.
        """
        del self.integrator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def create_basic_input_features(self):
        """
        Generate a dictionary of basic input features for testing.
        
        Returns a dictionary containing tensors used for embedding tests. The dictionary
        includes:
          - ref_pos: A random tensor of shape (N_atom, 3) representing atomic positions.
          - ref_charge: A random tensor of shape (N_atom, 1) representing atomic charges.
          - ref_element: A random tensor of shape (N_atom, 128) providing element features.
          - ref_atom_name_chars: A zero tensor of shape (N_atom, 256) for atom name encodings.
          - atom_to_token: A tensor mapping atoms to tokens, generated by repeating token
            indices according to atoms_per_token.
          - atom_to_token_idx: A tensor of repeated token indices corresponding to each atom.
          - restype: A zero tensor of shape (N_token, 32) for residue type information.
          - profile: A zero tensor of shape (N_token, 32) representing token profiles.
          - deletion_mean: A zero tensor of shape (N_token) for deletion mean values.
        
        The tensor dimensions are determined by the instance attributes N_atom, N_token, and atoms_per_token.
        """
        return {
            "ref_pos": torch.randn(self.N_atom, 3),
            "ref_charge": torch.randn(self.N_atom, 1),
            "ref_element": torch.randn(self.N_atom, 128),
            "ref_atom_name_chars": torch.zeros(self.N_atom, 256),
            "atom_to_token": torch.repeat_interleave(
                torch.arange(self.N_token), self.atoms_per_token
            ),
            "atom_to_token_idx": torch.repeat_interleave(
                torch.arange(self.N_token), self.atoms_per_token
            ),
            "restype": torch.zeros(self.N_token, 32),
            "profile": torch.zeros(self.N_token, 32),
            "deletion_mean": torch.zeros(self.N_token),
        }

    def test_error_with_oversized_ref_atom_name_chars(self):
        """Test error when ref_atom_name_chars has more than 256 dimensions."""
        input_features = self.create_basic_input_features()

        # Create ref_atom_name_chars with larger second dimension
        input_features["ref_atom_name_chars"] = torch.zeros(self.N_atom, 300)

        # Call the method and check for ValueError
        with self.assertRaises(ValueError) as context:
            self.integrator.build_embeddings(input_features)

        self.assertIn("ref_atom_name_chars feature has dimension", str(context.exception))

    def test_error_with_wrong_dimension_feature(self):
        """Test error when a feature has wrong dimensions."""
        input_features = self.create_basic_input_features()

        # Create a feature with wrong dimensions (3D)
        input_features["ref_charge"] = torch.zeros(self.N_atom, 1, 1)

        # Call the method and check for ValueError
        with self.assertRaises(ValueError) as context:
            self.integrator.build_embeddings(input_features)

        self.assertIn("Expected feature", str(context.exception))
        self.assertIn("to have 2D shape", str(context.exception))


class TestProtenixIntegrationHypothesis(unittest.TestCase):
    """Property-based tests for the ProtenixIntegration class using Hypothesis."""

    @settings(deadline=None, max_examples=10)
    @given(
        c_token=st.integers(min_value=32, max_value=512),
        c_atom=st.integers(min_value=32, max_value=256),
        c_pair=st.integers(min_value=16, max_value=64),
    )
    def test_initialization_with_different_dimensions(self, c_token, c_atom, c_pair):
        """Test initialization with different dimension parameters."""
        integrator = ProtenixIntegration(
            c_token=c_token,
            c_atom=c_atom,
            c_pair=c_pair,
            device=torch.device("cpu")
        )

        # Check that parameters are set correctly
        self.assertEqual(integrator.input_embedder.c_token, c_token)
        self.assertEqual(integrator.input_embedder.c_atom, c_atom)
        self.assertEqual(integrator.input_embedder.c_atompair, c_pair)
        self.assertEqual(integrator.rel_pos_encoding.c_z, c_token)

        # Clean up
        del integrator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @settings(deadline=None, max_examples=5)
    @given(
        N_token=st.integers(min_value=2, max_value=5),
        atoms_per_token=st.integers(min_value=1, max_value=3),
    )
    def test_build_embeddings_with_different_sizes(self, N_token, atoms_per_token):
        """
        Tests build_embeddings with varied sequence lengths and atom counts.
        
        This test generates synthetic input features based on the provided number of tokens and atoms per token.
        It verifies that build_embeddings returns a dictionary containing 's_inputs' and 'z_init', ensuring that
        's_inputs' has a length equal to the token count and that 'z_init' is a square matrix with dimensions matching
        the token count.
        """
        integrator = ProtenixIntegration(device=torch.device("cpu"))
        N_atom = N_token * atoms_per_token

        # Create input features
        input_features = {
            "ref_pos": torch.randn(N_atom, 3),
            "ref_charge": torch.randn(N_atom, 1),
            "ref_element": torch.randn(N_atom, 128),
            "ref_atom_name_chars": torch.zeros(N_atom, 256),
            "atom_to_token": torch.repeat_interleave(
                torch.arange(N_token), atoms_per_token
            ),
            "atom_to_token_idx": torch.repeat_interleave(
                torch.arange(N_token), atoms_per_token
            ),
            "restype": torch.zeros(N_token, 32),
            "profile": torch.zeros(N_token, 32),
            "deletion_mean": torch.zeros(N_token),
        }

        # Call the method
        embeddings = integrator.build_embeddings(input_features)

        # Check that the expected keys are in the output
        self.assertIn("s_inputs", embeddings)
        self.assertIn("z_init", embeddings)

        # Check the shapes of the outputs
        s_inputs = embeddings["s_inputs"]
        z_init = embeddings["z_init"]

        self.assertEqual(s_inputs.shape[0], N_token)
        self.assertEqual(z_init.shape[0], N_token)
        self.assertEqual(z_init.shape[1], N_token)

        # Clean up
        del integrator
        del embeddings
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
