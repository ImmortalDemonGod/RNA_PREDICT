"""
Comprehensive tests for stage_c_reconstruction.py to improve test coverage.
"""

import torch
import unittest
from unittest.mock import patch, MagicMock
import sys
import io
from omegaconf import DictConfig, OmegaConf

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

        # Create a mock Hydra config
        self.cfg = OmegaConf.create({
            "stageC": {
                "method": "mp_nerf",
                "device": "cpu",
                "do_ring_closure": False,
                "place_bases": True,
                "sugar_pucker": "C3'-endo"
            }
        })

        # Create a mock config for legacy method
        self.legacy_cfg = OmegaConf.create({
            "stageC": {
                "method": "legacy",
                "device": "cpu",
                "do_ring_closure": False,
                "place_bases": True,
                "sugar_pucker": "C3'-endo"
            }
        })

    def test_stage_c_reconstruction_legacy(self):
        """Test the StageCReconstruction class (legacy approach)."""
        # Initialize the legacy reconstruction class
        legacy_reconstruction = StageCReconstruction()

        # Call the class
        result = legacy_reconstruction(self.torsion_angles)

        # Check the result
        self.assertIn("coords", result)
        self.assertIn("atom_count", result)
        self.assertEqual(result["coords"].shape, (len(self.torsion_angles) * 3, 3))
        self.assertEqual(result["atom_count"], len(self.torsion_angles) * 3)

    def test_run_stageC_rna_mpnerf(self):
        """Test the run_stageC_rna_mpnerf function."""
        # Use more targeted patches to avoid actual computation
        with patch('rna_predict.pipeline.stageC.mp_nerf.rna.build_scaffolds_rna_from_torsions') as mock_build_scaffolds:
            with patch('rna_predict.pipeline.stageC.mp_nerf.rna.skip_missing_atoms') as mock_skip_missing_atoms:
                with patch('rna_predict.pipeline.stageC.mp_nerf.rna.handle_mods') as mock_handle_mods:
                    with patch('rna_predict.pipeline.stageC.mp_nerf.rna.rna_fold') as mock_rna_fold:
                        with patch('rna_predict.pipeline.stageC.mp_nerf.rna.place_rna_bases') as mock_place_rna_bases:
                            # Setup mocks
                            mock_scaffolds = {
                                "angles_mask": torch.ones(len(self.sequence))
                            }
                            mock_build_scaffolds.return_value = mock_scaffolds
                            mock_skip_missing_atoms.return_value = mock_scaffolds
                            mock_handle_mods.return_value = mock_scaffolds

                            # Mock rna_fold to return a 2D tensor
                            mock_coords_bb = torch.randn(len(self.sequence) * 5, 3)  # 5 atoms per residue for backbone
                            mock_rna_fold.return_value = mock_coords_bb

                            # Mock place_rna_bases to return a 3D tensor
                            mock_coords_full = torch.randn(len(self.sequence) * 10, 1, 3)  # 10 atoms per residue including bases
                            mock_place_rna_bases.return_value = mock_coords_full

                            # Call the function
                            result = run_stageC_rna_mpnerf(
                                cfg=self.cfg,
                                sequence=self.sequence,
                                predicted_torsions=self.torsion_angles
                            )

                            # Check that the mocks were called with the correct arguments
                            mock_build_scaffolds.assert_called_once_with(
                                seq=self.sequence,
                                torsions=self.torsion_angles,
                                device="cpu",
                                sugar_pucker="C3'-endo"
                            )
                            mock_skip_missing_atoms.assert_called_once_with(self.sequence, mock_scaffolds)
                            mock_handle_mods.assert_called_once_with(self.sequence, mock_scaffolds)
                            mock_rna_fold.assert_called_once_with(mock_scaffolds, device="cpu", do_ring_closure=False)
                            mock_place_rna_bases.assert_called_once_with(
                                mock_coords_bb, self.sequence, mock_scaffolds["angles_mask"], device="cpu"
                            )

                            # Check the result
                            self.assertIn("coords", result)
                            self.assertIn("atom_count", result)
                            self.assertEqual(result["coords"].shape, mock_coords_full.shape)
                            self.assertEqual(result["atom_count"], mock_coords_full.shape[0] * mock_coords_full.shape[1])

    def test_run_stageC_rna_mpnerf_without_place_bases(self):
        """Test the run_stageC_rna_mpnerf function without placing bases."""
        # Use more targeted patches to avoid actual computation
        with patch('rna_predict.pipeline.stageC.mp_nerf.rna.build_scaffolds_rna_from_torsions') as mock_build_scaffolds:
            with patch('rna_predict.pipeline.stageC.mp_nerf.rna.skip_missing_atoms') as mock_skip_missing_atoms:
                with patch('rna_predict.pipeline.stageC.mp_nerf.rna.handle_mods') as mock_handle_mods:
                    with patch('rna_predict.pipeline.stageC.mp_nerf.rna.rna_fold') as mock_rna_fold:
                        # Setup mocks
                        mock_scaffolds = {
                            "angles_mask": torch.ones(len(self.sequence))
                        }
                        mock_build_scaffolds.return_value = mock_scaffolds
                        mock_skip_missing_atoms.return_value = mock_scaffolds
                        mock_handle_mods.return_value = mock_scaffolds

                        # Mock rna_fold to return a 2D tensor
                        mock_coords_bb = torch.randn(len(self.sequence) * 5, 3)  # 5 atoms per residue for backbone
                        mock_rna_fold.return_value = mock_coords_bb

                        # Create config with place_bases=False
                        cfg_no_bases = OmegaConf.create({
                            "stageC": {
                                "method": "mp_nerf",
                                "device": "cpu",
                                "do_ring_closure": False,
                                "place_bases": False,
                                "sugar_pucker": "C3'-endo"
                            }
                        })

                        # Call the function
                        result = run_stageC_rna_mpnerf(
                            cfg=cfg_no_bases,
                            sequence=self.sequence,
                            predicted_torsions=self.torsion_angles
                        )

                        # Check the result
                        self.assertIn("coords", result)
                        self.assertIn("atom_count", result)
                        # Should be 3D tensor after unsqueeze
                        self.assertEqual(result["coords"].dim(), 3)

    def test_run_stageC_rna_mpnerf_too_many_angles(self):
        """Test run_stageC_rna_mpnerf with more than 7 torsion angles."""
        # Create torsion angles with 10 columns
        torsion_angles_extra = torch.randn(len(self.sequence), 10)

        # Patch the build_scaffolds function to avoid actual computation
        with patch('rna_predict.pipeline.stageC.mp_nerf.rna.build_scaffolds_rna_from_torsions') as mock_build:
            with patch('rna_predict.pipeline.stageC.mp_nerf.rna.skip_missing_atoms') as mock_skip:
                with patch('rna_predict.pipeline.stageC.mp_nerf.rna.handle_mods') as mock_handle:
                    with patch('rna_predict.pipeline.stageC.mp_nerf.rna.rna_fold') as mock_fold:
                        with patch('rna_predict.pipeline.stageC.mp_nerf.rna.place_rna_bases') as mock_place:
                            # Setup mocks to return appropriate values
                            mock_scaffolds = {
                                "angles_mask": torch.ones(len(self.sequence))
                            }
                            mock_build.return_value = mock_scaffolds
                            mock_skip.return_value = mock_scaffolds
                            mock_handle.return_value = mock_scaffolds

                            # Mock rna_fold to return a 2D tensor
                            mock_coords_bb = torch.randn(len(self.sequence) * 5, 3)  # 5 atoms per residue for backbone
                            mock_fold.return_value = mock_coords_bb

                            # Mock place_rna_bases to return a 3D tensor
                            mock_coords_full = torch.randn(len(self.sequence) * 10, 1, 3)  # 10 atoms per residue including bases
                            mock_place.return_value = mock_coords_full

                            # Call the function
                            run_stageC_rna_mpnerf(
                                cfg=self.cfg,
                                sequence=self.sequence,
                                predicted_torsions=torsion_angles_extra
                            )

                            # Check that build_scaffolds was called with sliced torsion angles
                            mock_build.assert_called_once()
                            # Get the arguments passed to the mock
                            _, kwargs = mock_build.call_args
                            # Check that the torsions were sliced to 7 columns
                            self.assertEqual(kwargs['torsions'].shape, (len(self.sequence), 7))

    def test_run_stageC_rna_mpnerf_too_few_angles(self):
        """Test run_stageC_rna_mpnerf with fewer than 7 torsion angles."""
        # Create torsion angles with only 5 columns
        torsion_angles_few = torch.randn(len(self.sequence), 5)

        # Check that ValueError is raised
        with self.assertRaises(ValueError) as context:
            run_stageC_rna_mpnerf(
                cfg=self.cfg,
                sequence=self.sequence,
                predicted_torsions=torsion_angles_few
            )

        self.assertIn("Not enough angles for Stage C", str(context.exception))

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
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            # Patch run_stageC to avoid actual computation
            with patch('rna_predict.pipeline.stageC.stage_c_reconstruction.run_stageC') as mock_run:
                mock_result = {"coords": torch.randn(80, 1, 3), "atom_count": 80}
                mock_run.return_value = mock_result

                # Run hydra_main with our config
                hydra_main(self.cfg)

                # Check that output contains expected strings
                output = captured_output.getvalue()
                self.assertIn("Running Stage C with Hydra configuration:", output)
                self.assertIn("Running Stage C for sequence: ACGUACGU", output)
                self.assertIn("Using dummy torsions shape:", output)
                self.assertIn("Stage C Output:", output)
                self.assertIn("Coords shape:", output)
                self.assertIn("Atom count:", output)
                self.assertIn("Output device:", output)
        finally:
            # Reset stdout
            sys.stdout = sys.__stdout__


if __name__ == '__main__':
    unittest.main()
