import unittest
from typing import List
from unittest.mock import patch  # Added imports

import numpy as np
import pytest  # Added import
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
import random

from rna_predict.pipeline.stageC.mp_nerf.ml_utils import (
    _run_main_logic,  # Added import for the refactored function
    atom_selector,
    chain2atoms,
    fape_torch,
    get_symmetric_atom_pairs,
    rename_symmetric_atoms,
    scn_atom_embedd,
    torsion_angle_loss,
    noise_internals_legacy,
    combine_noise_legacy,
    process_coordinates,
)

from rna_predict.pipeline.stageC.mp_nerf.ml_utils.coordinate_transforms import (
    combine_noise_legacy as combine_noise,
    noise_internals_legacy as noise_internals,
    NoiseConfig,
)
from rna_predict.pipeline.stageC.mp_nerf.protein_utils import SUPREME_INFO

# Configure Hypothesis settings globally
settings.register_profile("slow", max_examples=10, deadline=None)
settings.load_profile("slow")


class TestScnAtomEmbedd(unittest.TestCase):
    """Test cases for the scn_atom_embedd function."""

    valid_aa = "A"  # Default valid amino acid

    @given(seq_list=st.lists(st.text(min_size=1, max_size=5), min_size=1, max_size=3))
    def test_basic_embedding(self, seq_list: List[str]):
        """Test basic embedding functionality with various sequences."""
        # Convert any invalid amino acids to valid ones
        valid_seq_list = [
            "".join(c if c in SUPREME_INFO else self.valid_aa for c in seq)
            for seq in seq_list
        ]
        result = scn_atom_embedd(valid_seq_list)

        # Check output shape
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(len(result.shape), 3)  # (batch, seq_len, 14)
        self.assertEqual(result.shape[0], len(seq_list))
        self.assertEqual(result.shape[1], max(len(seq) for seq in seq_list))
        self.assertEqual(result.shape[2], 14)  # 14 atoms per residue

        # Check data type
        self.assertEqual(result.dtype, torch.long)

    def test_invalid_amino_acid(self):
        """Test handling of invalid amino acids."""
        # Use characters guaranteed not to be in SUPREME_INFO
        seq_list = ["X", "B", "Z"]
        result = scn_atom_embedd(seq_list)

        # Check that invalid amino acids are converted to padding
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (3, 1, 14))
        # Ensure all elements are the padding token ID (0)
        self.assertTrue(torch.all(result == 0))

    def test_padding_handling(self):
        """Test handling of padding tokens."""
        seq_list = ["A", "AA", "AAA"]
        result = scn_atom_embedd(seq_list)

        # Check that shorter sequences are padded
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (3, 3, 14))
        self.assertTrue(torch.all(result[0, 1:] == 0))
        self.assertTrue(torch.all(result[1, 2:] == 0))


class TestChain2Atoms(unittest.TestCase):
    """Test cases for the chain2atoms function."""

    @given(
        x=arrays(
            dtype=np.float32,
            shape=st.tuples(
                st.integers(min_value=1, max_value=5),
                st.integers(min_value=1, max_value=5),
            ),
        ),
        mask_input=st.one_of(
            st.none(), st.booleans()
        ),  # Generate None or a single boolean
        c=st.integers(min_value=1, max_value=3),
    )
    def test_basic_expansion(self, x, mask_input, c):
        """Test basic expansion functionality."""
        x_tensor = torch.tensor(x)
        mask_tensor = None
        x_tensor.shape[0]

        if mask_input is not None:
            # Create a boolean tensor mask of the correct shape (L,)
            # For simplicity, make it all True or all False based on the input boolean
            mask_tensor = torch.full((x_tensor.shape[0],), mask_input, dtype=torch.bool)
            if mask_input:
                x_tensor.shape[0]
            else:
                pass  # If mask is all False, expect 0 rows

        result = chain2atoms(x_tensor, mask_tensor, c)

        # Check output shape
        self.assertIsInstance(result, torch.Tensor)
        # The function should always return 3 dims: (masked_size, c, feature_dim)
        # Note: If mask selects 0 elements, shape might be (0, c, feature_dim)
        self.assertEqual(
            len(result.shape), 3, f"Expected 3 dimensions, got {len(result.shape)}"
        )

        # Check the first dimension size (masked_size)
        if mask_tensor is not None:
            # Use mask_tensor.sum() which works even if all False
            self.assertEqual(
                result.shape[0], mask_tensor.sum().item(), "Masked size mismatch"
            )
        else:
            # If no mask, first dim should be original length
            self.assertEqual(
                result.shape[0], x_tensor.shape[0], "Unmasked size mismatch"
            )

        self.assertEqual(result.shape[1], c, "Dimension 'c' mismatch")
        self.assertEqual(
            result.shape[2], x_tensor.shape[1], "Feature dimension mismatch"
        )


class TestRenameSymmetricAtoms(unittest.TestCase):
    """Test cases for the rename_symmetric_atoms function."""

    valid_aa = "A"

    def test_basic_renaming(self):
        """Test basic atom renaming functionality."""
        seq = "AA"
        coords = torch.randn(28, 3)  # (num_atoms=2*14, 3)
        feats = torch.randn(28, 4)  # (num_atoms=2*14, 4)
        result_coords, result_feats = rename_symmetric_atoms(coords, feats, seq)

        # Check output shapes
        self.assertIsInstance(result_coords, torch.Tensor)
        self.assertIsInstance(result_feats, torch.Tensor)
        self.assertEqual(result_coords.shape, coords.shape)
        self.assertEqual(result_feats.shape, feats.shape)

    def test_no_sequence(self):
        """Test behavior when no sequence is provided."""
        coords = torch.randn(28, 3)
        feats = torch.randn(28, 4)
        result_coords, result_feats = rename_symmetric_atoms(coords, feats)

        # Should return inputs unchanged
        self.assertIsInstance(result_coords, torch.Tensor)
        self.assertIsInstance(result_feats, torch.Tensor)
        self.assertEqual(result_coords.shape, coords.shape)
        self.assertEqual(result_feats.shape, feats.shape)
        self.assertTrue(torch.equal(result_coords, coords))
        self.assertTrue(torch.equal(result_feats, feats))


class TestTorsionAngleLoss(unittest.TestCase):
    """Test cases for the torsion_angle_loss function."""

    @given(
        pred_torsions=arrays(
            dtype=np.float32,
            shape=st.tuples(
                st.integers(min_value=1, max_value=5),
                st.integers(min_value=1, max_value=5),
            ),
        ),
        true_torsions=arrays(
            dtype=np.float32,
            shape=st.tuples(
                st.integers(min_value=1, max_value=5),
                st.integers(min_value=1, max_value=5),
            ),
        ),
        coeff=st.floats(min_value=0.0, max_value=1.0),
        has_mask=st.booleans(),
    )
    def test_basic_loss(self, pred_torsions, true_torsions, coeff, has_mask):
        """Test basic loss calculation."""
        # Ensure shapes match
        if pred_torsions.shape != true_torsions.shape:
            true_torsions = np.resize(true_torsions, pred_torsions.shape)

        pred_tensor = torch.tensor(pred_torsions)
        true_tensor = torch.tensor(true_torsions)
        # Ensure mask is boolean type for indexing
        angle_mask = (
            torch.ones_like(pred_tensor, dtype=torch.bool) if has_mask else None
        )

        result = torsion_angle_loss(pred_tensor, true_tensor, coeff, angle_mask)

        # Check output shape
        self.assertIsInstance(result, torch.Tensor)

        # Check that loss is non-negative
        self.assertTrue(torch.all(result >= 0))

        # Check that masked values are 0
        if has_mask:
            self.assertTrue(torch.all(result[angle_mask] == 0))


class TestFapeTorch(unittest.TestCase):
    """Test cases for the fape_torch function."""

    # Define a strategy for the shape first
    coord_shape_strategy = st.tuples(
        st.integers(min_value=1, max_value=3),  # Batch size (B)
        st.integers(min_value=1, max_value=3),  # Length (L)
        st.integers(min_value=1, max_value=3),  # Atoms per residue (C)
        st.just(3),  # Dimensions (D)
    )

    @given(
        shape=coord_shape_strategy,
        coords_data=st.data(),  # Use st.data() to draw dependent arrays
        max_val=st.floats(
            min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False
        ),  # Avoid small max_val
        c_alpha=st.booleans(),
    )
    def test_basic_fape(self, shape, coords_data, max_val, c_alpha):
        """Test basic FAPE calculation."""
        # Generate arrays with the same shape using a more precise float strategy
        float_strategy = st.floats(
            min_value=-10.0,  # Reduce range to avoid precision issues
            max_value=10.0,
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=False,
            width=32,  # Explicitly use 32-bit floats
        )

        pred_coords = coords_data.draw(
            arrays(dtype=np.float32, shape=shape, elements=float_strategy)
        )
        true_coords = coords_data.draw(
            arrays(dtype=np.float32, shape=shape, elements=float_strategy)
        )

        pred_tensor = torch.tensor(pred_coords)
        true_tensor = torch.tensor(true_coords)
        # Create seq_list with length matching the batch size (shape[0])
        batch_size = shape[0]
        seq_list = ["A"] * batch_size  # Dummy sequence for each item in batch

        result = fape_torch(
            pred_tensor,
            true_tensor,
            max_val=max_val,
            c_alpha=c_alpha,
            seq_list=seq_list,
        )

        # Check output shape (should be B,)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(len(result.shape), 1)
        self.assertEqual(result.shape[0], batch_size)

        # Check that FAPE is between 0 and 1
        self.assertTrue(torch.all(result >= 0))
        self.assertTrue(torch.all(result <= 1))


class TestAtomSelector(unittest.TestCase):
    """Test cases for the atom_selector function."""

    valid_aa = "A"
    VALID_OPTIONS = [
        "backbone",
        "backbone-with-oxygen",
        "backbone-with-cbeta",
        "backbone-with-cbeta-and-oxygen",
        "all",
    ]

    def _create_input_tensor(self, scn_seq):
        """Create input tensor for atom selection test.

        Args:
            scn_seq: Sequence string
            **kwargs: Additional arguments (ignored)

        Returns:
            tuple: (input tensor, sequence list)
        """
        seq_len = len(scn_seq)
        # Create x as (1, seq_len * 14, 3) for batch size 1
        x = torch.randn(1, seq_len * 14, 3)  # Shape: (batch, num_atoms_flat, coords)
        # atom_selector expects scn_seq as a list of strings (batch)
        scn_seq_list = [scn_seq]
        return x, scn_seq_list

    def _validate_output_tensors(self, result, mask, seq_len):
        """Validate the basic properties of output tensors.

        Args:
            result: Result tensor from atom_selector
            mask: Mask tensor from atom_selector
            seq_len: Length of the sequence
        """
        # Check that result and mask are tensors
        self.assertIsInstance(result, torch.Tensor)
        self.assertIsInstance(mask, torch.Tensor)

        # Check that mask is a boolean tensor
        self.assertEqual(mask.dtype, torch.bool)
        # Mask shape should be (batch, seq_len * 14)
        self.assertEqual(mask.shape, (1, seq_len * 14))

    def _create_option_mask_from_string(self, option_str, char):
        """Create an option mask from a string option.

        Args:
            option_str: String selection option
            char: Residue character

        Returns:
            torch.Tensor: Option mask
        """
        option_mask_np = np.zeros(14, dtype=bool)
        # Map string options to boolean masks
        if "backbone" in option_str:
            option_mask_np[[0, 1, 2]] = True  # N, CA, C
        if "oxygen" in option_str:
            option_mask_np[3] = True  # O
        if "cbeta" in option_str:
            option_mask_np[4] = True  # CB
        if option_str == "all":
            option_mask_np[:] = True
        # Adjust for Glycine if CB is selected by the option
        if char == "G" and option_mask_np[4]:
            option_mask_np[4] = False  # Glycine has no CB
        return torch.tensor(option_mask_np, dtype=torch.bool)

    def _create_option_mask(self, option, char):
        """Create an option mask based on the option type and residue.

        Args:
            option: Selection option (string or tensor)
            char: Residue character

        Returns:
            torch.Tensor: Option mask
        """
        if isinstance(option, str):
            return self._create_option_mask_from_string(option, char)
        elif isinstance(option, torch.Tensor):
            return option.bool()  # Use the provided tensor mask
        else:
            self.fail(f"Invalid option type: {type(option)}")

    def _calculate_expected_atoms(self, scn_seq, option, discard_absent):
        """Calculate the expected number of atoms based on the selection criteria.

        Args:
            scn_seq: Sequence string
            option: Selection option
            discard_absent: Whether to discard absent atoms

        Returns:
            int: Expected total number of atoms
        """
        expected_total_atoms = 0
        for char in scn_seq:
            if char != "_":  # Skip padding
                # Get the option mask for this residue
                option_mask = self._create_option_mask(option, char)

                # Calculate expected count based on discard_absent
                if discard_absent:
                    # When discarding absent, count is the sum of the option mask
                    expected_total_atoms += option_mask.sum().item()
                else:
                    # When not discarding absent, count is the intersection with cloud_mask
                    residue_cloud_mask = torch.tensor(
                        SUPREME_INFO[char]["cloud_mask"], dtype=torch.bool
                    )
                    expected_total_atoms += (residue_cloud_mask & option_mask).sum().item()

        return expected_total_atoms

    def _validate_result_shape(self, result, expected_total_atoms):
        """Validate the shape of the result tensor.

        Args:
            result: Result tensor from atom_selector
            expected_total_atoms: Expected number of atoms
        """
        # Assert the shape of the result tensor
        self.assertEqual(result.shape[0], expected_total_atoms)
        if expected_total_atoms > 0:
            self.assertEqual(result.shape[1], 3, "Result shape should have 3 dimensions (coords)")
        else:
            # If 0 atoms selected, shape might be (0,) or (0, 3) depending on implementation
            self.assertTrue(result.shape[0] == 0, "Result shape should be (0, ...) when no atoms selected")

    @given(
        scn_seq=st.text(min_size=1, max_size=3).map(
            lambda s: "".join(c if c in SUPREME_INFO else "A" for c in s)
        ),
        option=st.one_of(
            st.sampled_from(VALID_OPTIONS), st.just(torch.ones(14, dtype=torch.bool))
        ),  # Only valid options
        discard_absent=st.booleans(),
    )
    def test_basic_selection(self, scn_seq, option, discard_absent):
        """Test basic atom selection functionality."""
        # Create input tensor and sequence list
        x, scn_seq_list = self._create_input_tensor(scn_seq)

        # Call the atom_selector function
        result, mask = atom_selector(scn_seq_list, x, option, discard_absent)

        # Validate basic properties of output tensors
        self._validate_output_tensors(result, mask, len(scn_seq))

        # Calculate expected number of atoms
        expected_total_atoms = self._calculate_expected_atoms(scn_seq, option, discard_absent)

        # Assert the total sum of the mask matches the calculated expected total
        self.assertEqual(
            mask.sum().item(),
            expected_total_atoms,
            f"Mask sum mismatch for seq='{scn_seq}', option='{option}', discard_absent={discard_absent}",
        )

        # Validate the shape of the result tensor
        self._validate_result_shape(result, expected_total_atoms)


class NoiseTestConfig:
    """Configuration for noise test parameters.

    This class encapsulates the parameters needed for noise testing,
    reducing the number of function arguments and improving code organization.
    """
    def __init__(self, seq, has_angles, has_coords, noise_scale, theta_scale):
        self.seq = seq
        self.has_angles = has_angles
        self.has_coords = has_coords
        self.noise_scale = noise_scale
        self.theta_scale = theta_scale

        # Initialize tensors based on configuration
        self.angles = torch.randn(len(seq), 14) if has_angles else None
        self.coords = torch.randn(len(seq), 14, 3) if has_coords else None


class TestNoiseInternals(unittest.TestCase):
    """Test cases for the noise_internals function."""

    valid_aa = "A"

    def _create_test_config(self, seq, has_angles, has_coords, noise_scale, theta_scale):
        """Create a test configuration object.

        Args:
            seq: Sequence string
            has_angles: Whether to include angles
            has_coords: Whether to include coordinates
            noise_scale: Scale of noise to apply
            theta_scale: Scale of theta noise to apply

        Returns:
            NoiseTestConfig: Test configuration object
        """
        return NoiseTestConfig(seq, has_angles, has_coords, noise_scale, theta_scale)

    @given(
        seq=st.text(min_size=1, max_size=3).map(
            lambda s: "".join(c if c in SUPREME_INFO else "A" for c in s)
        ),
        has_angles=st.booleans(),
        has_coords=st.booleans(),
        noise_scale=st.floats(min_value=0.0, max_value=0.1),  # Reduce max noise scale
        theta_scale=st.floats(min_value=0.0, max_value=0.1),  # Reduce max theta scale
    )
    def test_basic_noise(self, seq, has_angles, has_coords, noise_scale, theta_scale):
        """Test basic noise functionality."""
        # Ensure at least one of angles or coords is provided
        if not has_angles and not has_coords:
            has_angles = True  # Default to using angles if neither is selected
            
        angles = torch.randn(len(seq), 3) if has_angles else None
        coords = torch.randn(len(seq), 3, 3) if has_coords else None
        
        # Create NoiseConfig object
        noise_config = NoiseConfig(noise_scale=noise_scale, theta_scale=theta_scale)
        
        result = noise_internals_legacy(
            seq=seq,
            angles=angles,
            coords=coords,
            config=noise_config
        )
        
        # Verify the result is a tuple
        self.assertIsInstance(result, tuple)
        # Verify it contains coords and mask
        self.assertEqual(len(result), 2)


class CombineNoiseTestConfig:
    """Configuration for combine noise test parameters.

    This class encapsulates the parameters needed for combine noise testing,
    reducing the number of function arguments and improving code organization.
    """
    def __init__(self, true_coords, has_seq, has_int_seq, has_angles,
                 noise_internals, internals_scn_scale, sidechain_reconstruct):
        self.true_coords = torch.tensor(true_coords)
        self.has_seq = has_seq
        self.has_int_seq = has_int_seq
        self.has_angles = has_angles
        self.noise_internals = noise_internals
        self.internals_scn_scale = internals_scn_scale
        self.sidechain_reconstruct = sidechain_reconstruct

        # Generate sequence data based on configuration
        seq_len = self.true_coords.shape[0]
        self.seq = "A" * seq_len if has_seq else None
        self.int_seq = torch.ones(seq_len, dtype=torch.long) if has_int_seq else None
        self.angles = torch.randn(seq_len, 12) if has_angles else None


class TestCombineNoise(unittest.TestCase):
    """Test cases for the combine_noise function."""

    valid_aa = "A"

    def _create_combine_test_config(self, true_coords, has_seq, has_int_seq, has_angles,
                                    noise_internals, internals_scn_scale, sidechain_reconstruct):
        """Create a test configuration object for combine noise testing.

        Args:
            true_coords: True coordinates tensor
            has_seq: Whether to include sequence
            has_int_seq: Whether to include integer sequence
            has_angles: Whether to include angles
            noise_internals: Scale of internal noise
            internals_scn_scale: Scale for internal SCN
            sidechain_reconstruct: Whether to reconstruct sidechains

        Returns:
            CombineNoiseTestConfig: Test configuration object
        """
        return CombineNoiseTestConfig(
            true_coords, has_seq, has_int_seq, has_angles,
            noise_internals, internals_scn_scale, sidechain_reconstruct
        )

    def test_basic_combination(self):
        """Test basic functionality of combine_noise."""
        # Create a small test coordinate tensor
        test_coords = torch.randn(1, 6, 3)
        config = self._create_combine_test_config(
            true_coords=test_coords,
            has_seq=True,
            has_int_seq=True,
            has_angles=True,
            noise_internals=0.1,
            internals_scn_scale=0.1,
            sidechain_reconstruct=False
        )
        noised_coords, mask = combine_noise_legacy(
            true_coords=config.true_coords,
            seq=config.seq,
            int_seq=config.int_seq,
            angles=config.angles,
            noise_internals=config.noise_internals,
            internals_scn_scale=config.internals_scn_scale,
            sidechain_reconstruct=config.sidechain_reconstruct
        )
        
        # Verify output shapes match input
        self.assertEqual(noised_coords.shape, test_coords.shape, "Noised coordinates shape mismatch")
        self.assertEqual(mask.shape, (test_coords.shape[0], test_coords.shape[1]), "Mask shape mismatch")
        
        # Verify mask is boolean
        self.assertEqual(mask.dtype, torch.bool, "Mask should be boolean tensor")
        
        # Verify coordinates are finite
        self.assertTrue(torch.isfinite(noised_coords).all(), "Noised coordinates contain non-finite values")


class TestGetSymmetricAtomPairs(unittest.TestCase):
    """Test cases for the get_symmetric_atom_pairs function."""

    valid_aa = "A"

    @given(
        seq=st.text(min_size=1, max_size=3).map(
            lambda s: "".join(c if c in SUPREME_INFO else "A" for c in s)
        )
    )
    def test_basic_pairs(self, seq):
        """Test basic symmetric atom pair extraction."""
        result = get_symmetric_atom_pairs(seq)

        # Check that result is a dictionary
        self.assertIsInstance(result, dict)

        # Check that all keys are string representations of valid indices
        seq_len = len(seq)
        self.assertTrue(
            all(key.isdigit() and 0 <= int(key) < seq_len for key in result.keys()),
            f"Dictionary keys should be string indices from 0 to {seq_len-1}. Got: {list(result.keys())}",
        )

        # Check that all values are lists of tuples (representing atom index pairs)
        for pairs in result.values():
            self.assertIsInstance(pairs, list)
            for pair in pairs:
                self.assertIsInstance(pair, tuple)
                self.assertEqual(len(pair), 2)
                # Check that the elements within the pair are integers (indices)
                self.assertTrue(
                    all(isinstance(x, int) for x in pair),
                    f"Expected tuple of ints, got {pair}",
                )


# test ML utils
def test_scn_atom_embedd():
    seq_list = ["AGCDEFGIKLMNPQRSTVWY", "WERTQLITANMWTCSDAAA_"]
    embedds = scn_atom_embedd(seq_list)
    assert embedds.shape == torch.Size([2, 20, 14]), "Shapes don't match"


def test_chain_to_atoms():
    chain = torch.randn(100, 3)
    atoms = chain2atoms(chain, c=14)
    assert atoms.shape == torch.Size([100, 14, 3]), "Shapes don't match"


def test_rename_symmetric_atoms():
    seq_list = ["AGCDEFGIKLMNPQRSTV"]
    # Adjust shapes to match expected input format (batch, num_atoms, 3/features)
    # Assuming num_atoms = seq_len * 14 for SCN format
    seq_len = len(seq_list[0])
    num_atoms = seq_len * 14
    pred_coors = torch.randn(1, num_atoms, 3)  # Example: Batch size 1
    pred_feats = torch.randn(1, num_atoms, 16)  # Example: Batch size 1, 16 features
    # true_coors and cloud_mask are no longer needed for the refactored function call

    # Call with the updated signature
    renamed_coors, renamed_feats = rename_symmetric_atoms(
        pred_coors=pred_coors[0],  # Pass the first batch element
        pred_feats=pred_feats[0],  # Pass the first batch element
        seq=seq_list[0],
    )

    # Check output shapes (adjusting for single batch element processing)
    assert (
        renamed_coors.shape == pred_coors[0].shape
    ), f"Coordinate shapes don't match: Expected {pred_coors[0].shape}, Got {renamed_coors.shape}"
    assert (
        renamed_feats.shape == pred_feats[0].shape
    ), f"Feature shapes don't match: Expected {pred_feats[0].shape}, Got {renamed_feats.shape}"


def test_torsion_angle_loss():
    pred_torsions = torch.randn(1, 100, 7)
    true_torsions = torch.randn(1, 100, 7)

    loss = torsion_angle_loss(pred_torsions, true_torsions, coeff=2.0, angle_mask=None)
    assert loss.shape == pred_torsions.shape, "Shapes don't match"


def test_fape_loss_torch():
    seq_list = ["AGCDEFGIKLMNPQRSTV"]
    pred_coords = torch.randn(1, 18, 14, 3)
    true_coords = torch.randn(1, 18, 14, 3)

    fape_torch(pred_coords, true_coords, c_alpha=True, seq_list=seq_list)
    fape_torch(pred_coords, true_coords, c_alpha=False, seq_list=seq_list)

    assert True


if __name__ == "__main__":
    unittest.main()


# --- Test for refactored __main__ logic ---
@patch("builtins.print")  # Mock print to suppress output during test
@patch("joblib.load")  # Corrected patch target
def test__run_main_logic(mock_joblib_load, mock_print):
    """
    Tests the _run_main_logic function (code previously in if __name__ == '__main__').
    Mocks joblib.load to provide dummy data and ensures the function executes,
    thereby achieving coverage.
    """
    # --- Setup Dummy Data ---
    # Define shapes based on the code's usage
    seq_len = 5  # Example sequence length
    num_atoms = seq_len * 14
    num_angles = 12  # Based on noise_internals usage

    # Create dummy data matching the expected structure unpacked in the function
    dummy_seq = "A" * seq_len
    dummy_int_seq = torch.randint(0, 20, (seq_len,))
    # The code adds batch dim later, so load data without it initially
    dummy_true_coords = torch.randn(num_atoms, 3)
    dummy_angles = torch.randn(seq_len, num_angles)
    dummy_padding_seq = None  # Not used in the logic
    dummy_mask = None  # Not used in the logic
    dummy_pid = "dummy_protein"  # Not used in the logic

    dummy_protein_data = (
        dummy_seq,
        dummy_int_seq,
        dummy_true_coords,
        dummy_angles,
        dummy_padding_seq,
        dummy_mask,
        dummy_pid,
    )

    # Configure the mock to return a list containing the dummy data tuple
    mock_joblib_load.return_value = [dummy_protein_data]

    # --- Execute ---
    # Call the function that contains the logic from the original __main__ block
    try:
        _run_main_logic()
    except FileNotFoundError:
        # The mocked joblib.load should prevent this, but handle defensively
        pytest.fail(
            "FileNotFoundError occurred unexpectedly during _run_main_logic call."
        )
    except Exception as e:
        # Catch other potential errors during execution
        pytest.fail(f"_run_main_logic execution failed with error: {e}")

    # --- Assert ---
    # Verify joblib.load was called once with the expected hardcoded path
    mock_joblib_load.assert_called_once_with(
        "some_route_to_local_serialized_file_with_prots"
    )

    # Verify print was called (optional, depends on whether you want to check output)
    # Check that print was called at least 3 times as in the original logic
    assert mock_print.call_count >= 3
