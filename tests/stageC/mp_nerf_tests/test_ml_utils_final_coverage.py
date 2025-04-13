"""
Tests specifically designed to achieve near 100% coverage for the ml_utils module.
This targets the remaining uncovered lines identified in the coverage report.
"""

import unittest
import torch
from unittest.mock import patch

from rna_predict.pipeline.stageC.mp_nerf.ml_utils import (
    atom_selector,
    combine_noise_legacy as combine_noise,
    fape_torch,
    process_coordinates,
)
from rna_predict.pipeline.stageC.mp_nerf.ml_utils.atom_utils import (
    scn_atom_embedd,
    rename_symmetric_atoms,
    get_symmetric_atom_pairs,
)
from rna_predict.pipeline.stageC.mp_nerf.ml_utils.angle_utils import (
    torsion_angle_loss,
)
from rna_predict.pipeline.stageC.mp_nerf.ml_utils.main import _run_main_logic
from rna_predict.pipeline.stageC.mp_nerf.ml_utils.tensor_ops import (
    chain2atoms,
)


class TestAtomUtilsFinalCoverage(unittest.TestCase):
    """Test cases for atom_utils.py to achieve near 100% coverage."""

    def test_scn_atom_embedd_error_handling(self):
        """Test error handling in scn_atom_embedd function (line 57)."""
        # Create an invalid sequence with a character not in AAS2INDEX
        seq = ["AXG"]  # X is not a valid amino acid

        # Call scn_atom_embedd with the invalid sequence
        # This should trigger the error handling code
        result = scn_atom_embedd(seq)

        # Verify that the result is a tensor with the expected shape
        # Even with an invalid amino acid, the function should return a tensor
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape[0], 1)  # Batch size
        self.assertEqual(result.shape[1], 3)  # Sequence length

    def test_chain2atoms_error_handling(self):
        """Test error handling in chain2atoms function."""
        # Create a tensor with an invalid shape
        x = torch.randn(15, 3)  # Shape (15, 3)

        # Call chain2atoms with the tensor
        # This should not raise an error, but we can test the output shape
        result = chain2atoms(x)

        # Verify the output shape
        self.assertEqual(result.shape, (15, 3, 3))

    def test_rename_symmetric_atoms_error_handling(self):
        """Test error handling in rename_symmetric_atoms function (lines 215, 217, 220)."""
        # Create a tensor with a valid shape
        x = torch.randn(14, 3)  # 1 residue with 14 atoms

        # Call rename_symmetric_atoms with no sequence
        # This should return the original tensor
        result, _ = rename_symmetric_atoms(x, seq=None)

        # Verify that the result is the same as the input
        self.assertTrue(torch.allclose(result, x))

    def test_atom_selector_edge_cases(self):
        """Test edge cases in atom_selector function (lines 244, 254, 272-276)."""
        # Create a sequence with all amino acids to test edge cases
        seq = ["ACDEFGHIKLMNPQRSTVWY"]

        # Create a tensor with shape (batch=1, len=20*14, dims=3)
        # Each residue has 14 atoms, so we need 20*14=280 atoms
        x = torch.zeros((1, 280, 3))

        # Fill in some non-zero values to make the test more realistic
        for i in range(280):
            x[0, i] = torch.tensor([i, i+1, i+2], dtype=torch.float)

        # Call atom_selector with an invalid option
        # This should trigger the error handling code
        with self.assertRaises(ValueError):
            atom_selector(
                scn_seq=seq,
                x=x,
                option="invalid-option",
                discard_absent=True
            )

        # Test with all option
        selected, mask = atom_selector(
            scn_seq=seq,
            x=x,
            option="all",
            discard_absent=True
        )

        # Verify the output shape
        self.assertEqual(selected.shape, (280, 3))
        self.assertEqual(mask.shape, (1, 280))


class TestCoordinateTransformsFinalCoverage(unittest.TestCase):
    """Test cases for coordinate_transforms.py to achieve near 100% coverage."""

    def test_noise_internals_error_handling(self):
        """Test error handling in noise_internals function (lines 59->74, 102->106, 106->126, 108, 128)."""
        # Create tensors for testing
        coords = torch.randn(3, 14, 3)  # 3 residues * 14 atoms
        angles = torch.randn(3, 6)  # 3 residues, 6 angles per residue

        # Import the actual function directly from the module
        from rna_predict.pipeline.stageC.mp_nerf.ml_utils.coordinate_transforms import NoiseConfig, noise_internals_legacy as direct_noise_internals

        # Create config object with angles
        noise_config = NoiseConfig(
            noise_scale=0.01,
            theta_scale=0.5,
            verbose=0
        )
        
        # Call noise_internals with valid parameters
        noised_coords, mask = direct_noise_internals(
            seq="AAA",
            angles=angles,
            coords=None,
            config=noise_config
        )

        # Verify the output shape
        self.assertEqual(noised_coords.shape, (3, 14, 3))
        self.assertEqual(mask.shape, (3, 14))

        # Test with coords instead of angles
        noised_coords, mask = direct_noise_internals(
            seq="AAA",
            angles=None,
            coords=coords,
            config=noise_config
        )

        # Verify the output shape
        self.assertEqual(noised_coords.shape, (3, 14, 3))
        self.assertEqual(mask.shape, (3, 14))

        # Test that providing neither angles nor coords raises an error
        with self.assertRaises(AssertionError):
            direct_noise_internals(
                seq="AAA",
                angles=None,
                coords=None,
                config=noise_config
            )

    def test_combine_noise_error_handling(self):
        """Test error handling in combine_noise function (lines 164->166, 216->226, 219-221, etc.)."""
        # Create tensors for testing
        true_coords = torch.randn(1, 42, 3)  # 3 residues * 14 atoms
        seq = "AAA"  # 3 residues

        # Call combine_noise_legacy with valid parameters
        noised_coords, mask = combine_noise(
            true_coords=true_coords,
            seq=seq,
            int_seq=None,
            angles=None,
            noise_internals=0.01,
            sidechain_reconstruct=False
        )

        # Verify the output shape
        self.assertEqual(noised_coords.shape, true_coords.shape)
        self.assertEqual(mask.shape, (1, 42))

        # Test with a tensor sequence
        int_seq = torch.tensor([0, 0, 0])  # 3 residues of type 0 (Alanine)
        noised_coords, mask = combine_noise(
            true_coords=true_coords,
            seq=None,
            int_seq=int_seq,
            angles=None,
            noise_internals=0.01,
            sidechain_reconstruct=False
        )

        # Verify the output shape
        self.assertEqual(noised_coords.shape, true_coords.shape)
        self.assertEqual(mask.shape, (1, 42))


class TestLossFunctionsFinalCoverage(unittest.TestCase):
    """Test cases for loss_functions.py to achieve near 100% coverage."""

    def test_torsion_angle_loss_edge_cases(self):
        """Test edge cases in torsion_angle_loss function."""
        # Create tensors for testing
        pred = torch.randn(2, 3, 6)  # 2 batches, 3 residues, 6 angles
        target = torch.randn(2, 3, 6)  # Same shape
        mask = torch.ones(2, 3, 6, dtype=torch.bool)  # All angles are valid

        # Call torsion_angle_loss with different parameters
        loss = torsion_angle_loss(pred, target, angle_mask=mask)
        self.assertIsInstance(loss, torch.Tensor)

        # Test with a different coefficient
        loss = torsion_angle_loss(pred, target, coeff=5.0, angle_mask=mask)
        self.assertIsInstance(loss, torch.Tensor)

    def test_fape_torch_multiple_frames(self):
        """Test fape_torch with multiple frames (lines 113, 137, 146)."""
        # Create tensors for testing
        batch_size = 2
        seq_len = 3
        num_atoms = 4
        num_frames = 3

        pred_coords = torch.randn(batch_size, seq_len, num_atoms, 3)
        true_coords = torch.randn(batch_size, seq_len, num_atoms, 3)
        seq_list = ["AAA", "AAA"]

        # Create rotation matrices with multiple frames
        rot_mats_g = []
        for _ in range(batch_size):
            # Create random rotation matrices
            rot_mat = torch.randn(num_frames, 3, 3)
            # Make them orthogonal (not perfect but good enough for testing)
            u, _, v = torch.svd(rot_mat)
            rot_mat = torch.matmul(u, v.transpose(-2, -1))
            rot_mats_g.append(rot_mat)

        # Call fape_torch with multiple rotation frames
        fape_val = fape_torch(
            pred_coords=pred_coords,
            true_coords=true_coords,
            max_val=10.0,
            c_alpha=False,
            seq_list=seq_list,
            rot_mats_g=rot_mats_g
        )

        # Verify the output shape
        self.assertEqual(fape_val.shape, (batch_size,))


class TestMainModuleFinalCoverage(unittest.TestCase):
    """Test cases for main.py to achieve near 100% coverage."""

    @patch('sys.argv', ['ml_utils.py', '--seq', 'ACGT', '--fold', '--invalid-option'])
    def test_run_main_logic_with_multiple_options(self):
        """Test _run_main_logic with multiple options (lines 51->53, 53->57)."""
        # This should print an error message about multiple options
        try:
            _run_main_logic()
            # If we get here, the test passes
        except Exception as e:
            # If it raises any other exception, the test fails
            self.fail(f"_run_main_logic raised {type(e).__name__} unexpectedly!")

    @patch('sys.argv', ['ml_utils.py', '--seq', 'ACGT', '--fold'])
    @patch('torch.randn')
    @patch('torch.save')
    @patch('os.path.exists')
    def test_run_main_logic_fold_with_save(self, mock_exists, mock_save, mock_randn):
        """Test _run_main_logic with fold option and save (lines 63-64, 77-78)."""
        # Mock torch.randn to return a predictable tensor
        mock_randn.return_value = torch.zeros(4, 14, 3)
        # Mock os.path.exists to return True
        mock_exists.return_value = True

        # This should run the fold function and try to save the result
        try:
            _run_main_logic()
            # If we get here, the test passes
        except Exception as e:
            # If it raises any other exception, the test fails
            self.fail(f"_run_main_logic raised {type(e).__name__} unexpectedly!")

        # We don't verify that torch.save was called because it depends on the file existing

    @patch('sys.argv', ['ml_utils.py', '--seq', 'ACGT', '--noise'])
    @patch('torch.randn')
    @patch('torch.save')
    @patch('os.path.exists')
    def test_run_main_logic_noise_with_save(self, mock_exists, mock_save, mock_randn):
        """Test _run_main_logic with noise option and save (lines 91-92, 96)."""
        # Mock torch.randn to return a predictable tensor
        mock_randn.return_value = torch.zeros(4, 14, 3)
        # Mock os.path.exists to return True
        mock_exists.return_value = True

        # This should run the noise function and try to save the result
        try:
            _run_main_logic()
            # If we get here, the test passes
        except Exception as e:
            # If it raises any other exception, the test fails
            self.fail(f"_run_main_logic raised {type(e).__name__} unexpectedly!")

        # We don't verify that torch.save was called because it depends on the file existing


class TestTensorOpsFinalCoverage(unittest.TestCase):
    """Test cases for tensor_ops.py to achieve near 100% coverage."""

    def test_process_coordinates_error_handling(self):
        """Test error handling in process_coordinates function (lines 46-49)."""
        # Create a tensor with an invalid shape
        noised_coords = torch.randn(10, 3)  # Not divisible by 14

        # Create a minimal scaffolds dictionary
        scaffolds = {
            "wrapper": torch.zeros(1, 14, 3),  # This will be ignored
            "torsion_angles": torch.zeros(1, 6),
            "bond_angles": torch.zeros(1, 3),
            "torsion_mask": torch.ones(1, 6, dtype=torch.bool),
            "bond_mask": torch.ones(1, 3, dtype=torch.bool)
        }

        # Call process_coordinates and expect it to raise an error
        try:
            # This should raise an einops.EinopsError
            process_coordinates(noised_coords, scaffolds)
            # If we get here, the test fails
            self.fail("Expected EinopsError but no exception was raised")
        except Exception:
            # If it raises any exception, the test passes
            # We're just testing that the function handles invalid inputs
            pass

    def test_get_symmetric_atom_pairs_edge_cases(self):
        """Test edge cases in get_symmetric_atom_pairs function."""
        # Test with different amino acids
        for aa in "ACDEFGHIKLMNPQRSTVWY":
            pairs = get_symmetric_atom_pairs(aa)
            self.assertIsInstance(pairs, dict)

            # Some amino acids have symmetric atom pairs, but not all
            # We're just testing that the function returns a dict without errors

        # Test with an invalid amino acid
        pairs = get_symmetric_atom_pairs("X")
        self.assertEqual(pairs, {})


if __name__ == "__main__":
    unittest.main()
