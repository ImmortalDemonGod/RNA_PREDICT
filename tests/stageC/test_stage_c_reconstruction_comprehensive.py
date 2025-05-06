"""
Comprehensive tests for stage_c_reconstruction.py to improve test coverage.
"""

import torch
import unittest
from unittest.mock import patch
from omegaconf import OmegaConf
from hypothesis import given, settings, strategies as st

from rna_predict.pipeline.stageC.stage_c_reconstruction import (
    StageCReconstruction,
    run_stageC_rna_mpnerf,
    run_stageC,
    hydra_main
)


class TestStageCReconstructionComprehensive(unittest.TestCase):
    def setUp(self):
        # Set random seed for reproducibility
        torch.manual_seed(42)

        # Create test data
        self.sequence = "ACGU"
        self.torsion_angles = torch.randn(len(self.sequence), 7)  # 7 torsion angles per residue

        # Create a mock Hydra config with the correct structure
        self.cfg = OmegaConf.create({
            "model": {
                "stageC": {
                    "enabled": True,
                    "method": "mp_nerf",
                    "device": "cpu",
                    "do_ring_closure": False,
                    "place_bases": True,
                    "sugar_pucker": "C3'-endo",
                    "angle_representation": "cartesian",
                    "use_metadata": False,
                    "use_memory_efficient_kernel": False,
                    "use_deepspeed_evo_attention": False,
                    "use_lma": False,
                    "inplace_safe": True,
                    "debug_logging": True  # Enable debug logging for tests
                }
            },
            "test_data": {
                "sequence": "ACGUACGU",
                "torsion_angle_dim": 7
            }
        })

        # Create a mock config for legacy method
        self.legacy_cfg = OmegaConf.create({
            "model": {
                "stageC": {
                    "enabled": True,
                    "method": "legacy",
                    "device": "cpu",
                    "do_ring_closure": False,
                    "place_bases": True,
                    "sugar_pucker": "C3'-endo",
                    "angle_representation": "cartesian",
                    "use_metadata": False,
                    "use_memory_efficient_kernel": False,
                    "use_deepspeed_evo_attention": False,
                    "use_lma": False,
                    "inplace_safe": True,
                    "debug_logging": False
                }
            }
        })

    def test_stage_c_reconstruction_legacy(self):
        """
        Tests that StageCReconstruction produces correct output using the legacy configuration.
        
        Verifies that the output contains "coords" and "atom_count" keys, and that their values
        have the expected shapes and counts based on the input torsion angles.
        """
        # Initialize the legacy reconstruction class with a configuration object
        legacy_reconstruction = StageCReconstruction(self.legacy_cfg)

        # Call the class
        result = legacy_reconstruction(self.torsion_angles)

        # Check the result
        self.assertIn("coords", result)
        self.assertIn("atom_count", result)
        self.assertEqual(result["coords"].shape, (len(self.torsion_angles) * 3, 3))
        self.assertEqual(result["atom_count"], len(self.torsion_angles) * 3)

    def test_run_stageC_rna_mpnerf(self):
        """Test the run_stageC_rna_mpnerf function."""
        # Mock the entire run_stageC_rna_mpnerf function
        with patch('rna_predict.pipeline.stageC.stage_c_reconstruction.run_stageC_rna_mpnerf', autospec=True) as mock_run:
            # Set up a mock return value
            mock_result = {
                "coords": torch.ones((len(self.sequence) * 10, 3)),  # [N*atoms, 3]
                "coords_3d": torch.ones((len(self.sequence), 10, 3)),  # [N, atoms_per_residue, 3]
                "atom_count": len(self.sequence) * 10,
                "atom_metadata": {
                    "atom_names": ["C1'"] * (len(self.sequence) * 10),
                    "residue_indices": [i // 10 for i in range(len(self.sequence) * 10)]
                }
            }
            mock_run.return_value = mock_result

            # Call the function
            result = run_stageC(
                cfg=self.cfg,
                sequence=self.sequence,
                torsion_angles=self.torsion_angles
            )

            # Check that the mock was called with the correct arguments
            mock_run.assert_called_once_with(
                cfg=self.cfg,
                sequence=self.sequence,
                predicted_torsions=self.torsion_angles
            )

            # Check the result
            self.assertIn("coords", result)
            self.assertIn("atom_count", result)
            self.assertEqual(result["coords"].shape, (len(self.sequence) * 10, 3))
            self.assertEqual(result["atom_count"], len(self.sequence) * 10)

    def test_run_stageC_rna_mpnerf_without_place_bases(self):
        """Test the run_stageC_rna_mpnerf function without placing bases."""
        # Create config with place_bases=False
        cfg_no_bases = OmegaConf.create({
            "model": {
                "stageC": {
                    "enabled": True,
                    "method": "mp_nerf",
                    "device": "cpu",
                    "do_ring_closure": False,
                    "place_bases": False,
                    "sugar_pucker": "C3'-endo",
                    "angle_representation": "cartesian",
                    "use_metadata": False,
                    "use_memory_efficient_kernel": False,
                    "use_deepspeed_evo_attention": False,
                    "use_lma": False,
                    "inplace_safe": True,
                    "debug_logging": False
                }
            }
        })

        # Mock the entire run_stageC_rna_mpnerf function
        with patch('rna_predict.pipeline.stageC.stage_c_reconstruction.run_stageC_rna_mpnerf', autospec=True) as mock_run:
            # Set up a mock return value
            mock_result = {
                "coords": torch.ones((len(self.sequence) * 10, 3)),  # [N*atoms, 3]
                "coords_3d": torch.ones((len(self.sequence), 10, 3)),  # [N, atoms_per_residue, 3]
                "atom_count": len(self.sequence) * 10,
                "atom_metadata": {
                    "atom_names": ["C1'"] * (len(self.sequence) * 10),
                    "residue_indices": [i // 10 for i in range(len(self.sequence) * 10)]
                }
            }
            mock_run.return_value = mock_result

            # Call the function
            result = run_stageC(
                cfg=cfg_no_bases,
                sequence=self.sequence,
                torsion_angles=self.torsion_angles
            )

            # Check that the mock was called with the correct arguments
            mock_run.assert_called_once_with(
                cfg=cfg_no_bases,
                sequence=self.sequence,
                predicted_torsions=self.torsion_angles
            )

            # Check the result
            self.assertIn("coords", result)
            self.assertIn("atom_count", result)
            self.assertEqual(result["coords"].shape, (len(self.sequence) * 10, 3))
            self.assertEqual(result["atom_count"], len(self.sequence) * 10)

    @given(
        sequence=st.text(alphabet="ACGU", min_size=1, max_size=10),
        extra_angles=st.integers(min_value=8, max_value=20)
    )
    @settings(deadline=None)
    def test_run_stageC_rna_mpnerf_too_many_angles(self, sequence, extra_angles):
        """Property-based test: run_stageC_rna_mpnerf should slice torsion angles to 7 columns if more are provided.

        This test verifies that when torsion angles with more than 7 columns are provided,
        the function correctly slices them to the first 7 columns before passing them to build_scaffolds_rna_from_torsions.

        Args:
            sequence: RNA sequence to test with
            extra_angles: Number of torsion angles to provide (> 7)
        """
        # Create torsion angles with extra_angles columns
        torsion_angles_extra = torch.randn(len(sequence), extra_angles)

        # Create a config with the correct structure
        test_cfg = OmegaConf.create({
            "model": {
                "stageC": {
                    "enabled": True,
                    "method": "mp_nerf",
                    "device": "cpu",
                    "do_ring_closure": False,
                    "place_bases": True,
                    "sugar_pucker": "C3'-endo",
                    "angle_representation": "cartesian",
                    "use_metadata": False,
                    "use_memory_efficient_kernel": False,
                    "use_deepspeed_evo_attention": False,
                    "use_lma": False,
                    "inplace_safe": True,
                    "debug_logging": False
                }
            }
        })

        # Mock the build_scaffolds_rna_from_torsions function to check that it's called with sliced torsions
        with patch('rna_predict.pipeline.stageC.mp_nerf.rna.build_scaffolds_rna_from_torsions') as mock_build:
            # Set up a mock return value for build_scaffolds_rna_from_torsions
            valid_atom_mask = torch.ones(len(sequence) * 10, dtype=torch.bool)
            mock_scaffolds = {
                "angles_mask": torch.ones(len(sequence)),
                "valid_atom_mask": valid_atom_mask
            }
            mock_build.return_value = mock_scaffolds

            # Mock the other functions to avoid actual computation
            with patch('rna_predict.pipeline.stageC.mp_nerf.rna.skip_missing_atoms', return_value=mock_scaffolds):
                with patch('rna_predict.pipeline.stageC.mp_nerf.rna.handle_mods', return_value=mock_scaffolds):
                    with patch('rna_predict.pipeline.stageC.mp_nerf.rna.rna_fold', return_value=torch.ones((len(sequence), 10, 3))):
                        with patch('rna_predict.pipeline.stageC.mp_nerf.rna.place_rna_bases', return_value=torch.ones((len(sequence), 10, 3))):
                            # Mock the STANDARD_RNA_ATOMS dictionary to avoid KeyError
                            with patch('rna_predict.utils.tensor_utils.types.STANDARD_RNA_ATOMS', autospec=True) as mock_atoms:
                                # Set up the mock to return a list of atom names for each residue
                                mock_atoms.__getitem__.side_effect = lambda res: [f"ATOM_{i}" for i in range(10)]

                                # Call the function
                                run_stageC_rna_mpnerf(
                                    cfg=test_cfg,
                                    sequence=sequence,
                                    predicted_torsions=torsion_angles_extra
                                )

                                # Check that build_scaffolds was called with sliced torsion angles
                                mock_build.assert_called_once()
                                # Get the arguments passed to the mock
                                _, kwargs = mock_build.call_args
                                # Check that the torsions were sliced to 7 columns
                                self.assertEqual(kwargs['torsions'].shape, (len(sequence), 7),
                                                f"[UniqueErrorID-TorsionSlicing] Torsions should be sliced to 7 columns, but got {kwargs['torsions'].shape}")

    @given(
        sequence=st.text(alphabet="ACGU", min_size=1, max_size=10),
        few_angles=st.integers(min_value=1, max_value=6)
    )
    @settings(deadline=None)
    def test_run_stageC_rna_mpnerf_too_few_angles(self, sequence, few_angles):
        """Property-based test: run_stageC_rna_mpnerf should raise ValueError if fewer than 7 torsion angles are provided.

        This test verifies that when torsion angles with fewer than 7 columns are provided,
        the function correctly raises a ValueError with an appropriate error message.

        Args:
            sequence: RNA sequence to test with
            few_angles: Number of torsion angles to provide (< 7)
        """
        # Skip empty sequences
        if not sequence:
            return

        # Create torsion angles with few_angles columns
        torsion_angles_few = torch.randn(len(sequence), few_angles)

        # Create a config with the correct structure
        test_cfg = OmegaConf.create({
            "model": {
                "stageC": {
                    "enabled": True,
                    "method": "mp_nerf",
                    "device": "cpu",
                    "do_ring_closure": False,
                    "place_bases": True,
                    "sugar_pucker": "C3'-endo",
                    "angle_representation": "cartesian",
                    "use_metadata": False,
                    "use_memory_efficient_kernel": False,
                    "use_deepspeed_evo_attention": False,
                    "use_lma": False,
                    "inplace_safe": True,
                    "debug_logging": False
                }
            }
        })

        # Check that ValueError is raised
        with self.assertRaises(ValueError) as context:
            run_stageC_rna_mpnerf(
                cfg=test_cfg,
                sequence=sequence,
                predicted_torsions=torsion_angles_few
            )

        self.assertIn("Not enough angles for Stage C", str(context.exception),
                     f"[UniqueErrorID-TorsionTooFew] Expected error message 'Not enough angles for Stage C' but got '{str(context.exception)}'")

    @patch('rna_predict.pipeline.stageC.stage_c_reconstruction.run_stageC_rna_mpnerf')
    def test_run_stageC_with_hydra_config(self, mock_run_stageC_rna_mpnerf):
        """Test run_stageC with Hydra config."""
        # Setup mock
        mock_result = {"coords": torch.randn(40, 1, 3), "atom_count": 40}
        mock_run_stageC_rna_mpnerf.return_value = mock_result

        # Call the function
        result = run_stageC(
            sequence=self.sequence,
            torsion_angles=self.torsion_angles,
            cfg=self.cfg
        )

        # Check that run_stageC_rna_mpnerf was called with the correct arguments
        mock_run_stageC_rna_mpnerf.assert_called_once_with(
            cfg=self.cfg,
            sequence=self.sequence,
            predicted_torsions=self.torsion_angles
        )

        # Check the result
        self.assertEqual(result, mock_result)

    def test_run_stageC_with_legacy_method(self):
        """Test run_stageC with legacy method."""
        # Call the function with legacy method
        result = run_stageC(
            sequence=self.sequence,
            torsion_angles=self.torsion_angles,
            cfg=self.legacy_cfg
        )

        # Check the result
        self.assertIn("coords", result)
        self.assertIn("atom_count", result)
        self.assertEqual(result["coords"].shape, (len(self.torsion_angles) * 3, 3))
        self.assertEqual(result["atom_count"], len(self.torsion_angles) * 3)

    def test_run_stageC_with_direct_parameters(self):
        """Test run_stageC with direct parameters instead of config."""
        # Patch run_stageC_rna_mpnerf to avoid actual computation
        with patch('rna_predict.pipeline.stageC.stage_c_reconstruction.run_stageC_rna_mpnerf') as mock_run:
            mock_result = {"coords": torch.randn(40, 1, 3), "atom_count": 40}
            mock_run.return_value = mock_result

            # Call the function with direct parameters
            result = run_stageC(
                sequence=self.sequence,
                torsion_angles=self.torsion_angles,
                method="mp_nerf",
                device="cpu",
                do_ring_closure=True,
                place_bases=True,
                sugar_pucker="C3'-endo"
            )

            # Check that run_stageC_rna_mpnerf was called
            mock_run.assert_called_once()

            # Check the result
            self.assertEqual(result, mock_result)

    def test_hydra_main(self):
        """Test the hydra_main function."""
        # Patch the logger to capture log messages
        with patch('rna_predict.pipeline.stageC.stage_c_reconstruction.logger') as mock_logger:
            # Patch run_stageC to avoid actual computation
            with patch('rna_predict.pipeline.stageC.stage_c_reconstruction.run_stageC') as mock_run:
                mock_result = {
                    "coords": torch.randn(80, 3),  # [N*atoms, 3]
                    "coords_3d": torch.randn(8, 10, 3),  # [N, atoms_per_residue, 3]
                    "atom_count": 80,
                    "atom_metadata": {
                        "atom_names": ["C1'"] * 80,
                        "residue_indices": [i // 10 for i in range(80)]
                    }
                }
                mock_run.return_value = mock_result

                # Run hydra_main with our config
                hydra_main(self.cfg)

                # Check that the logger was called with the expected messages
                mock_logger.info.assert_any_call("Running Stage C with Hydra configuration:")
                mock_logger.debug.assert_any_call("Using standardized test sequence: ACGUACGU with 7 torsion angles")

                # Check that the logger was called with the expected debug messages
                mock_logger.debug.assert_any_call("\nRunning Stage C for sequence: ACGUACGU")

                # Verify that the mock_run was called
                mock_run.assert_called_once()


if __name__ == '__main__':
    unittest.main()
