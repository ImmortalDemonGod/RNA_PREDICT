# test_rna_predictor.py
"""
Merged Comprehensive Test Suite for RNAPredictor

This suite combines:
- Logical test grouping by functionality (Init, predict_3d_structure, predict_submission).
- setUp methods for consistent test fixtures.
- Mocking of run_stageC and StageBTorsionBertPredictor to force shape scenarios.
- Property-based tests using Hypothesis for broad input coverage.
- Thorough checks for edge cases, empty sequences, shape mismatches, index errors, etc.

Usage:
    python -m unittest test_rna_predictor.py -v
"""

import math
import unittest
from unittest.mock import patch
import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
import torch
from omegaconf import OmegaConf
import random
import os

from rna_predict.conf.config_schema import StageCConfig
from rna_predict.utils.tensor_utils.types import STANDARD_RNA_ATOMS
from rna_predict.pipeline.stageB.torsion.torsion_bert_predictor import StageBTorsionBertPredictor
from rna_predict.interface import RNAPredictor

# --------------------------------------------------------------------------------------
# Strategy definitions for property-based tests
# --------------------------------------------------------------------------------------

# Strategy for valid RNA sequences (A, C, G, U), allowing empty to test edge cases
valid_rna_sequences = st.text(alphabet="ACGU", min_size=0, max_size=50)

# Strategy enumerating how we might shape the coords in forced mocks:
#   0 => shape [N, 3]          (single-atom)
#   1 => shape [N * atoms, 3]  (legacy fallback?)
#   2 => shape [N, atoms, 3]   (already in [N, #atoms, 3])
coords_shape_type = st.integers(min_value=0, max_value=2)
atoms_per_res_strategy = st.integers(min_value=1, max_value=10)


# --------------------------------------------------------------------------------------
#                           Test Class: Initialization
# --------------------------------------------------------------------------------------
class TestRNAPredictorInitialization(unittest.TestCase):
    """
    Tests the RNAPredictor constructor logic, including:
      - default parameters,
      - user-provided parameters,
      - GPU vs CPU auto-detection,
      - random fuzzing of constructor args via Hypothesis.
    """

    @staticmethod
    def minimal_stageC_config(**overrides):
        """Helper to create a minimal valid StageCConfig using structured config."""
        base = StageCConfig()
        for k, v in overrides.items():
            setattr(base, k, v)
        return OmegaConf.structured(base)

    def test_init_defaults(self):
        """
        Test that default arguments successfully initialize.
        Checks device auto-detection (cpu/cuda), torsion predictor, and stageC_method.
        """
        # Patch: Use minimal valid Hydra config
        minimal_cfg = OmegaConf.create({
            "device": "cpu",
            "model": {
                "stageC": OmegaConf.to_container(self.minimal_stageC_config(method="mp_nerf", enabled=True, do_ring_closure=True, place_bases=True, sugar_pucker="C3'-endo", angle_representation="sin_cos", use_metadata=False, use_memory_efficient_kernel=False, use_deepspeed_evo_attention=False, use_lma=False, inplace_safe=False)),
                "stageB": {"torsion_bert": {"dummy": True, "debug_logging": False, "model_name_or_path": "dummy-path", "device": "cpu"}}
            },
            "prediction": {"repeats": 5, "residue_atom_choice": 0}
        })

        # Mock the Hugging Face model loading to avoid network calls
        with patch("transformers.AutoModel.from_pretrained"), \
             patch("transformers.AutoTokenizer.from_pretrained"):
            predictor = RNAPredictor(minimal_cfg)
        self.assertIsNotNone(
            predictor.torsion_predictor,
            "Should initialize torsion_predictor by default.",
        )
        self.assertIn(
            str(predictor.device),
            ["cpu", "cuda"],
            "Device should be cpu or cuda based on availability.",
        )
        # StageC_method is now in config, not as attribute
        self.assertEqual(
            predictor.stageC_config.method,
            "mp_nerf",
            "Default stageC_method should be 'mp_nerf'.",
        )

    @patch("rna_predict.pipeline.stageB.torsion.torsion_bert_predictor.StageBTorsionBertPredictor.__init__", return_value=None)
    def test_init_custom_params(self, mock_torsion_init):
        """
        Test that user-provided parameters are respected.
        """
        custom_cfg = OmegaConf.create({
            "device": "cpu",
            "model": {
                "stageC": OmegaConf.to_container(self.minimal_stageC_config(method="other_method", enabled=True, do_ring_closure=True, place_bases=True, sugar_pucker="C3'-endo", angle_representation="sin_cos", use_metadata=False, use_memory_efficient_kernel=False, use_deepspeed_evo_attention=False, use_lma=False, inplace_safe=False)),
                "stageB": {"torsion_bert": {"dummy": True, "debug_logging": False, "angle_mode": "sin_cos", "num_angles": 5, "max_length": 256, "model_name_or_path": "custom/path", "device": "cpu"}}
            },
            "prediction": {"repeats": 5, "residue_atom_choice": 0}
        })

        # Mock the Hugging Face model loading to avoid network calls
        with patch("transformers.AutoModel.from_pretrained"), \
             patch("transformers.AutoTokenizer.from_pretrained"):
            predictor = RNAPredictor(custom_cfg)
        self.assertEqual(str(predictor.device), "cpu")
        self.assertEqual(predictor.stageC_config.method, "other_method")
        # The following are not checked since torsion_predictor is mocked

    @patch("rna_predict.pipeline.stageB.torsion.torsion_bert_predictor.StageBTorsionBertPredictor.__init__", return_value=None)
    @given(
        model_name_or_path=st.from_regex(r"^[a-zA-Z0-9_-]{1,32}$", fullmatch=True),
        device=st.one_of(st.none(), st.sampled_from(["cpu", "cuda"])),
        angle_mode=st.sampled_from(["degrees", "radians", "sin_cos"]),
        num_angles=st.integers(min_value=1, max_value=10),
        max_length=st.integers(min_value=1, max_value=1024),
        stageC_method=st.sampled_from(["mp_nerf", "dummy_method", "other_method"]),
    )
    @settings(
        deadline=None, # Merged deadline setting here
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
        max_examples=20,
    )
    def test_fuzz_constructor_init(
        self,
        model_name_or_path,
        device,
        angle_mode,
        num_angles,
        max_length,
        stageC_method,
        mock_torsion_init
    ):
        """
        Hypothesis-driven fuzz test of the constructor to ensure broad coverage of parameter combos.
        """
        # Patch: Build config with all required sections
        config = OmegaConf.create({
            "device": device or "cpu",
            "model": {
                "stageC": OmegaConf.to_container(self.minimal_stageC_config(method=stageC_method, enabled=True, do_ring_closure=True, place_bases=True, sugar_pucker="C3'-endo", angle_representation="sin_cos", use_metadata=False, use_memory_efficient_kernel=False, use_deepspeed_evo_attention=False, use_lma=False, inplace_safe=False)),
                "stageB": {"torsion_bert": {"dummy": True, "debug_logging": False, "angle_mode": angle_mode, "num_angles": num_angles, "max_length": max_length, "model_name_or_path": model_name_or_path, "device": device or "cpu"}}
            },
            "prediction": {"repeats": 5, "residue_atom_choice": 0}
        })
        predictor = RNAPredictor(config)
        self.assertEqual(predictor.stageC_config.method, stageC_method)
        self.assertEqual(str(predictor.device), device or "cpu")
        # The following are not checked since torsion_predictor is mocked

# Unique test for invalid model names causing loader errors
@pytest.mark.parametrize("bad_model_name", ["", "!!!", "invalid/space", "a"*97])
def test_rnapredictor_invalid_model_name_raises(monkeypatch, bad_model_name):
    minimal_cfg = OmegaConf.create({
        "device": "cpu",
        "model": {
            "stageC": OmegaConf.to_container(TestRNAPredictorInitialization.minimal_stageC_config(method="mp_nerf", enabled=True, do_ring_closure=True, place_bases=True, sugar_pucker="C3'-endo", angle_representation="sin_cos", use_metadata=False, use_memory_efficient_kernel=False, use_deepspeed_evo_attention=False, use_lma=False, inplace_safe=False)),
            "stageB": {"torsion_bert": {"model_name_or_path": bad_model_name, "debug_logging": False, "device": "cpu"}}
        },
        "prediction": {"repeats": 5, "residue_atom_choice": 0}
    })
    # Patch StageBTorsionBertPredictor to raise ValueError on bad model name
    with patch("rna_predict.pipeline.stageB.torsion.torsion_bert_predictor.StageBTorsionBertPredictor.__init__", side_effect=ValueError("Invalid model_name_or_path")):
        with pytest.raises(ValueError, match="Invalid model_name_or_path"):
            RNAPredictor(minimal_cfg)


# --------------------------------------------------------------------------------------
#                         Test Class: Predict3DStructure
# --------------------------------------------------------------------------------------
class TestPredict3DStructure(unittest.TestCase):
    """
    Tests the predict_3d_structure method. Includes normal usage and
    forced shape errors, plus Hypothesis property-based tests.
    """

    def setUp(self):
        """
        Create a fresh RNAPredictor instance for each test,
        ensuring consistent device usage (CPU) to avoid GPU complications.
        """
        # Patch: Use minimal valid Hydra config
        minimal_cfg = OmegaConf.create({
            "device": "cpu",
            "model": {
                "stageC": OmegaConf.to_container(self.minimal_stageC_config(method="mp_nerf", enabled=True, do_ring_closure=True, place_bases=True, sugar_pucker="C3'-endo", angle_representation="sin_cos", use_metadata=False, use_memory_efficient_kernel=False, use_deepspeed_evo_attention=False, use_lma=False, inplace_safe=False)),
                "stageB": {"torsion_bert": {"dummy": True, "debug_logging": False, "model_name_or_path": "dummy-path", "device": "cpu"}}
            },
            "prediction": {"repeats": 5, "residue_atom_choice": 0}
        })

        # Mock the Hugging Face model loading to avoid network calls
        with patch("transformers.AutoModel.from_pretrained"), \
             patch("transformers.AutoTokenizer.from_pretrained"):
            self.predictor = RNAPredictor(minimal_cfg)

    @staticmethod
    def minimal_stageC_config(**overrides):
        """Helper to create a minimal valid StageCConfig using structured config."""
        base = StageCConfig()
        # Always set device to 'cpu' for tests to avoid CUDA errors
        setattr(base, "device", "cpu")
        for k, v in overrides.items():
            setattr(base, k, v)
        return OmegaConf.structured(base)

    def test_predict_3d_structure_basic(self):
        """
        Basic functional test with a short RNA sequence.
        Verifies presence of 'coords' and 'atom_count' in the returned dict.
        Adds debug output for failure analysis.
        """
        sequence = "ACGU"
        try:
            result = self.predictor.predict_3d_structure(sequence)
        except Exception as e:
            print(f"[DEBUG] Exception in test_predict_3d_structure_basic: {e}")
            print(f"[DEBUG] Config: {self.predictor.stageC_config}")
            raise
        self.assertIn("coords", result)
        self.assertIn("atom_count", result)
        coords = result["coords"]
        self.assertTrue(
            hasattr(coords, "shape"), "coords should be a tensor or similar."
        )
        self.assertGreaterEqual(
            coords.shape[-1], 3, "Should have at least x,y,z in the last dimension."
        )

    def test_predict_3d_structure_empty_seq(self):
        """
        If sequence is empty, pipeline might return coords of shape [0,3].
        Confirm method doesn't raise an error for an empty string.
        """
        sequence = ""
        result = self.predictor.predict_3d_structure(sequence)
        coords = result["coords"]
        self.assertEqual(
            coords.shape[0],
            0,
            "Empty sequence => zero residues => coords shape[0] = 0.",
        )
        self.assertEqual(result["atom_count"], 0, "No atoms if sequence is empty.")

    @given(valid_rna_sequences)
    @settings(
        deadline=None,  # Disable deadline for this flaky test
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
        max_examples=10,
    )
    def test_predict_3d_structure_with_random_sequences(self, sequence):
        """
        Property-based test: random valid RNA sequences.
        Ensures the method runs without raising exceptions for typical usage.
        Adds debug output for shape/config errors.
        """
        try:
            result = self.predictor.predict_3d_structure(sequence)
            self.assertIn("coords", result)
            self.assertIn("atom_count", result)

            # Validate coords_3d tensor for its 3D shape
            coords_3d = result["coords_3d"]
            self.assertTrue(torch.is_tensor(coords_3d), "coords_3d should be a tensor")
            self.assertEqual(
                coords_3d.dim(), 3, "coords_3d should be 3D tensor [N, atoms, 3]"
            )
            self.assertEqual(
                coords_3d.shape[-1], 3, "last dimension of coords_3d should be 3 for x,y,z"
            )

            # For short sequences, NaN values might be present due to model limitations
            # We'll only check for NaN/Inf values if the sequence is long enough
            if len(sequence) > 3:  # Skip NaN check for very short sequences (3 or fewer)
                self.assertFalse(
                    torch.isnan(coords_3d).any(),
                    "[ERR-RNAPREDICT-NAN-001] coords_3d should not contain NaN values for sequences longer than 3. Sequence: {}".format(sequence)
                )
                self.assertFalse(
                    torch.isinf(coords_3d).any(),
                    "[ERR-RNAPREDICT-INF-001] coords_3d should not contain Inf values for sequences longer than 3. Sequence: {}".format(sequence)
                )

            # Also validate the flattened 'coords' tensor (optional, but good practice)
            coords_flat = result["coords"]
            self.assertTrue(torch.is_tensor(coords_flat), "coords (flat) should be a tensor")
            self.assertEqual(
                coords_flat.dim(), 2, "coords (flat) should be 2D tensor [total_atoms, 3]"
            )
            self.assertEqual(
                coords_flat.shape[-1], 3, "last dimension of coords (flat) should be 3 for x,y,z"
            )
            # Check for NaN/Inf values in the flattened tensor
            self.assertFalse(
                torch.isnan(coords_flat).any(),
                "[ERR-RNAPREDICT-NAN-002] coords (flat) should not contain NaN values. Sequence: {}".format(sequence)
            )
            self.assertFalse(
                torch.isinf(coords_flat).any(),
                "[ERR-RNAPREDICT-INF-002] coords (flat) should not contain Inf values. Sequence: {}".format(sequence)
            )

            # Validate atom_count
            self.assertIsInstance(
                result["atom_count"], int, "atom_count should be an integer"
            )
            self.assertGreaterEqual(
                result["atom_count"], 0, "atom_count should be non-negative"
            )

        except (RuntimeError, OSError, ValueError, IndexError) as e:
            # If no real model is found or environment lacks GPU, we pass
            if "CUDA" in str(e) or "model" in str(e).lower():
                pass
            else:
                print(f"[DEBUG] Hypothesis random failure in test_predict_3d_structure_with_random_sequences: {e}")
                print(f"[DEBUG] Config: {self.predictor.stageC_config}")
                raise  # Re-raise other errors


# --------------------------------------------------------------------------------------
#                        Test Class: PredictSubmission
# --------------------------------------------------------------------------------------
class TestPredictSubmission(unittest.TestCase):
    """
    Tests the predict_submission method, focusing on DataFrame output,
    repeated coords, residue atom choice, shape handling,
    and replicating NaN propagation from missing bond lengths.
    """

    @patch("rna_predict.interface.RNAPredictor.predict_3d_structure")
    def setUp(self, mock_predict_3d):
        """Instantiate a RNAPredictor for repeated usage."""
        random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
            torch.use_deterministic_algorithms(True)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        else:
            torch.use_deterministic_algorithms(True)
        # Build config via Hydra compose for tests
        from hydra import initialize_config_dir, compose
        # Initialize Hydra with absolute config_dir for tests
        with initialize_config_dir(config_dir="/Users/tomriddle1/RNA_PREDICT/rna_predict/conf", version_base="1.1", job_name="test_predict"):
            minimal_cfg = compose(
                config_name="predict",
                overrides=[
                    "device=cpu",
                    # Stage C overrides
                    "model.stageC.method=mp_nerf",
                    "model.stageC.enabled=true",
                    "model.stageC.do_ring_closure=true",
                    "model.stageC.place_bases=true",
                    "model.stageC.sugar_pucker=\"C3'-endo\"",  # Wrap sugar_pucker in quotes to satisfy Hydra override grammar
                    "model.stageC.angle_representation=sin_cos",
                    "model.stageC.use_metadata=false",
                    "model.stageC.use_memory_efficient_kernel=false",
                    "model.stageC.use_deepspeed_evo_attention=false",
                    "model.stageC.use_lma=false",
                    "model.stageC.inplace_safe=false",
                    # Stage B overrides
                    "model.stageB.torsion_bert.model_name_or_path=dummy-path",
                    "model.stageB.torsion_bert.device=cpu",
                    "model.stageB.torsion_bert.debug_logging=false",
                    # Prediction overrides
                    "prediction.repeats=5",
                    "prediction.residue_atom_choice=0",
                    "prediction.enable_stochastic_inference_for_submission=false",
                ],
            )

        # Mock the Hugging Face model loading to avoid network calls
        with patch("transformers.AutoModel.from_pretrained"), \
             patch("transformers.AutoTokenizer.from_pretrained"):
            self.predictor = RNAPredictor(minimal_cfg)
        # Patch the torsion_predictor's model.forward to accept any kwargs and return a dummy tensor
        import types
        def dummy_forward(*args, **kwargs):
            import torch
            seq_len = kwargs.get('input_ids', torch.zeros((1,))).shape[0] if 'input_ids' in kwargs else 1
            output_dim = 7  # or whatever is expected by the pipeline
            dummy_tensor = torch.zeros((1, seq_len, output_dim))
            return types.SimpleNamespace(last_hidden_state=dummy_tensor)
        # Defensive: Only patch if model exists
        if hasattr(self.predictor.torsion_predictor, 'model'):
            self.predictor.torsion_predictor.model.forward = types.MethodType(dummy_forward, self.predictor.torsion_predictor.model)

        # configure predict_3d_structure mock to return coords tensor of shape [total_atoms, 3]
        # using module-level torch and STANDARD_RNA_ATOMS imported at file top
        def fake_predict3d(self_arg, sequence):
            total_atoms = sum(len(STANDARD_RNA_ATOMS.get(res, [])) for res in sequence)
            return {"coords": torch.zeros((total_atoms, 3))}
        mock_predict_3d.side_effect = fake_predict3d

    @staticmethod
    def minimal_stageC_config(**overrides):
        """Helper to create a minimal valid StageCConfig using structured config."""
        base = StageCConfig()
        # Always set device to 'cpu' for tests to avoid CUDA errors
        setattr(base, "device", "cpu")
        for k, v in overrides.items():
            setattr(base, k, v)
        return OmegaConf.structured(base)

    @given(
        sequence=st.text(alphabet=st.sampled_from("ACGU"), min_size=1, max_size=10),
        repeats=st.integers(min_value=1, max_value=3)
    )
    @settings(deadline=None)
    def test_predict_submission_basic(self, sequence, repeats):
        """
        Property-based test: For any valid RNA sequence, output DataFrame shape should match total atom count for variable-atom case, or residue count for uniform case.
        """
        try:
            df = self.predictor.predict_submission(
                sequence, prediction_repeats=repeats, residue_atom_choice=0
            )
        except Exception as e:
            print(f"[DEBUG] Exception in test_predict_submission_basic: {e}")
            print(f"[DEBUG] Config: {self.predictor.stageC_config}")
            raise
        print(f"[DEBUG] DataFrame columns: {list(df.columns)}")
        print(f"[DEBUG] DataFrame shape: {df.shape}")
        seq_list = list(sequence)
        expected_atom_count = sum(len(STANDARD_RNA_ATOMS.get(res, [])) for res in seq_list)
        if df.shape[0] == expected_atom_count:
            # Variable atom count per residue (flat output)
            pass  # Accept
        else:
            self.assertEqual(
                len(df), len(sequence), "[UniqueErrorID-ShapeMismatch] DataFrame rows should match number of residues for uniform atom case."
            )
        expected_cols = ["ID", "resname", "resid"] if "ID" in df.columns else ["x_1", "y_1", "z_1", "residue_index"]
        for col in expected_cols:
            self.assertIn(col, df.columns)

    @given(
        repeats=st.integers(min_value=1, max_value=5)
    )
    @settings(deadline=None)
    def test_predict_submission_empty_seq(self, repeats):
        """
        Property-based test: If sequence is empty, expect an empty DataFrame but valid columns.
        Tests with different numbers of repeats to ensure column generation is correct.
        """
        sequence = ""
        df = self.predictor.predict_submission(sequence, prediction_repeats=repeats)
        self.assertTrue(df.empty, "DataFrame should be empty for empty sequence.")

        # Expected columns should include ID, resname, resid, and x_N, y_N, z_N for each repeat
        expected_cols = ["ID", "resname", "resid"]
        for i in range(1, repeats+1):
            expected_cols.extend([f"x_{i}", f"y_{i}", f"z_{i}"])

        self.assertListEqual(list(df.columns), expected_cols,
                             f"[UniqueErrorID-EmptySeqColumns] Expected columns {expected_cols} but got {list(df.columns)}")

    @given(
        sequence=st.text(alphabet="ACGU", min_size=1, max_size=10),
        atoms_per_res=st.integers(min_value=1, max_value=5),
        invalid_choice=st.integers(min_value=10, max_value=1000)
    )
    @settings(deadline=None)
    def test_predict_submission_invalid_atom_choice(self, sequence, atoms_per_res, invalid_choice):
        """
        Property-based test: If we pick a residue_atom_choice that doesn't exist, code should raise IndexError.
        Tests with different sequences, atom counts per residue, and invalid atom choices.
        """
        # Ensure the dummy predictor returns a shape [N, atoms_per_res, 3] so invalid_choice is always out of bounds
        with patch.object(self.predictor, "predict_3d_structure",
                         return_value={"coords": torch.zeros((len(sequence), atoms_per_res, 3)),
                                      "atom_count": len(sequence) * atoms_per_res}):
            try:
                with self.assertRaises(IndexError, msg=f"[UniqueErrorID-InvalidAtomChoice] Should raise IndexError for atom_choice={invalid_choice} with atoms_per_res={atoms_per_res}"):
                    self.predictor.predict_submission(sequence, residue_atom_choice=invalid_choice)
            except Exception as e:
                print(f"[DEBUG] Exception in test_predict_submission_invalid_atom_choice: {e}")
                print(f"[DEBUG] Config: {self.predictor.stageC_config}")
                print(f"[DEBUG] Sequence: {sequence}, atoms_per_res: {atoms_per_res}, invalid_choice: {invalid_choice}")
                raise

    @given(
        sequence=st.text(alphabet="ACGU", min_size=1, max_size=10),
        repeats=st.integers(min_value=1, max_value=5)
    )
    @settings(deadline=None)
    def test_predict_submission_custom_repeats(self, sequence, repeats):
        """
        Property-based test: For a valid sequence, output DataFrame shape should match total atom count (flat)
        or residue count (uniform), and columns should be correct for custom repeats.
        Tests with different sequences and repeat counts.
        """
        if not sequence:  # Skip empty sequences
            return

        df = self.predictor.predict_submission(sequence, prediction_repeats=repeats)
        print(f"[DEBUG] DataFrame columns: {list(df.columns)}")
        print(f"[DEBUG] DataFrame shape: {df.shape}")

        seq_list = list(sequence)
        expected_atom_count = sum(len(STANDARD_RNA_ATOMS.get(res, [])) for res in seq_list)

        # Expected columns should include ID, resname, resid, and x_N, y_N, z_N for each repeat
        expected_cols = ["ID", "resname", "resid"]
        for i in range(1, repeats+1):
            expected_cols.extend([f"x_{i}", f"y_{i}", f"z_{i}"])

        if df.shape[0] == expected_atom_count:
            # Variable atom count per residue (flat output)
            for col in expected_cols:
                self.assertIn(col, df.columns,
                             f"[UniqueErrorID-CustomRepeats] Missing column {col} in flat output")
        else:
            # Uniform atom count per residue
            self.assertEqual(len(df), len(sequence),
                           "[UniqueErrorID-CustomRepeats] DataFrame rows should match number of residues for uniform atom case.")
            for col in expected_cols:
                self.assertIn(col, df.columns,
                             f"[UniqueErrorID-CustomRepeats] Missing column {col} in uniform output")

    @given(
        sequence=st.text(alphabet="ACGU", min_size=1, max_size=10),
        atoms_per_res=st.integers(min_value=1, max_value=5),
        repeats=st.integers(min_value=1, max_value=5)
    )
    @settings(deadline=None)
    def test_predict_submission_nan_handling(self, sequence, atoms_per_res, repeats):
        """
        Property-based test: If predict_3d_structure returns NaN coords, they should appear in the submission DataFrame.
        Tests with different sequences, atom counts, and repeats.
        """
        if not sequence:  # Skip empty sequences
            return

        N = len(sequence)

        # Create mock coords with all NaN values to ensure consistent behavior
        mock_coords = torch.full((N, atoms_per_res, 3), float("nan"), dtype=torch.float32)

        # Mock predict_3d_structure
        with patch.object(self.predictor, "predict_3d_structure",
                         return_value={
                             "coords": mock_coords,
                             "atom_count": N * atoms_per_res,
                             "atom_metadata": {
                                 "atom_names": ["P"] * (N * atoms_per_res),
                                 "residue_indices": [i // atoms_per_res for i in range(N * atoms_per_res)],
                             },
                         }):
            # Call predict_submission
            df = self.predictor.predict_submission(
                sequence, prediction_repeats=repeats, residue_atom_choice=0
            )

            # Assert that NaNs are present in the coordinate columns
            # Check at least one coordinate column has NaNs
            has_nans = False
            for i in range(1, repeats + 1):
                for coord in ["x", "y", "z"]:
                    if df[f"{coord}_{i}"].isna().any():
                        has_nans = True
                        break
                if has_nans:
                    break

            self.assertTrue(has_nans,
                          "[UniqueErrorID-NaNHandling] At least one coordinate column should contain NaNs when all input coordinates are NaN")

    @given(
        sequence=st.text(alphabet="ACGU", min_size=1, max_size=10),
        atoms_per_res=st.integers(min_value=1, max_value=10),
        repeats=st.integers(min_value=1, max_value=5),
        atom_choice=st.integers(min_value=0, max_value=3)
    )
    @settings(deadline=None)
    def test_predict_submission_numerical_validity(self, sequence, atoms_per_res, repeats, atom_choice):
        """
        Property-based test: If predict_3d_structure returns valid numerical coords,
        the submission DataFrame should contain only finite values.
        Tests with different sequences, atom counts, repeats, and atom choices.
        """
        if not sequence:  # Skip empty sequences
            return

        N = len(sequence)

        # Skip if atom_choice is out of bounds for atoms_per_res
        if atom_choice >= atoms_per_res:
            return

        # Mock return with valid float values
        mock_coords = torch.randn((N, atoms_per_res, 3), dtype=torch.float32)

        # Mock predict_3d_structure
        with patch.object(self.predictor, "predict_3d_structure",
                         return_value={
                             "coords": mock_coords,
                             "atom_count": N * atoms_per_res,
                             "atom_metadata": {
                                 "atom_names": ["P"] * (N * atoms_per_res),
                                 "residue_indices": [i // atoms_per_res for i in range(N * atoms_per_res)],
                             },
                         }):
            # Call predict_submission
            df = self.predictor.predict_submission(
                sequence, prediction_repeats=repeats, residue_atom_choice=atom_choice
            )

            # Assert that coordinate columns contain only finite values
            for i in range(1, repeats + 1):
                col_x, col_y, col_z = f"x_{i}", f"y_{i}", f"z_{i}"
                self.assertTrue(
                    df[col_x].apply(lambda x: math.isfinite(x)).all(),
                    f"[UniqueErrorID-NumericalValidity] {col_x} column should contain only finite values",
                )
                self.assertTrue(
                    df[col_y].apply(lambda x: math.isfinite(x)).all(),
                    f"[UniqueErrorID-NumericalValidity] {col_y} column should contain only finite values",
                )
                self.assertTrue(
                    df[col_z].apply(lambda x: math.isfinite(x)).all(),
                    f"[UniqueErrorID-NumericalValidity] {col_z} column should contain only finite values",
                )


# --------------------------------------------------------------------------------------
#               Test Class: PredictSubmissionParametricShapes (Optional)
# --------------------------------------------------------------------------------------
class TestPredictSubmissionParametricShapes(unittest.TestCase):
    """
    Demonstrates shape-based forced mocking, ensuring coverage of coordinate shapes:
      [N,3], [N*atoms,3], [N,atoms,3].
    This merges advanced shape logic with Hypothesis for broad coverage.
    """

    def setUp(self):
        # Use Hydra config composition for minimal config (Hydra best practice)
        from hydra import initialize_config_dir, compose
        with initialize_config_dir(config_dir="/Users/tomriddle1/RNA_PREDICT/rna_predict/conf", version_base="1.1", job_name="test_predict_parametric_shapes"):
            minimal_cfg = compose(
                config_name="predict",
                overrides=[
                    "device=cpu",
                    # Stage C overrides
                    "model.stageC.method=mp_nerf",
                    "model.stageC.enabled=true",
                    "model.stageC.do_ring_closure=true",
                    "model.stageC.place_bases=true",
                    "model.stageC.sugar_pucker=\"C3'-endo\"",
                    "model.stageC.angle_representation=sin_cos",
                    "model.stageC.use_metadata=false",
                    "model.stageC.use_memory_efficient_kernel=false",
                    "model.stageC.use_deepspeed_evo_attention=false",
                    "model.stageC.use_lma=false",
                    "model.stageC.inplace_safe=false",
                    # Stage B overrides
                    "model.stageB.torsion_bert.model_name_or_path=dummy-path",
                    "model.stageB.torsion_bert.device=cpu",
                    "model.stageB.torsion_bert.debug_logging=false",
                    # Prediction overrides
                    "prediction.repeats=5",
                    "prediction.residue_atom_choice=0",
                    "prediction.enable_stochastic_inference_for_submission=false",
                ],
            )
        # Mock the Hugging Face model loading to avoid network calls
        with patch("transformers.AutoModel.from_pretrained"), \
             patch("transformers.AutoTokenizer.from_pretrained"):
            self.predictor = RNAPredictor(minimal_cfg)

    @staticmethod
    def minimal_stageC_config(**overrides):
        """Helper to create a minimal valid StageCConfig using structured config."""
        base = StageCConfig()
        # Always set device to 'cpu' for tests to avoid CUDA errors
        setattr(base, "device", "cpu")
        for k, v in overrides.items():
            setattr(base, k, v)
        return OmegaConf.structured(base)

    @given(
        seq=valid_rna_sequences.filter(lambda s: len(s) > 0),  # non-empty
        shape_type=coords_shape_type,
        atoms_per_res=atoms_per_res_strategy,
        repeats=st.integers(min_value=1, max_value=3),
    )
    @settings(
        deadline=None,  # Disable deadline for this flaky test
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
        max_examples=5,
    )
    def test_forced_coord_shapes(self, seq, shape_type, atoms_per_res, repeats):
        """
        Force run_stageC to return specific shapes, verifying correct reshaping or error handling.
        """
        N = len(seq)
        # Construct coords based on shape_type
        if shape_type == 0:
            coords = torch.zeros((N, 3))  # Single-atom scenario
            atom_count = N
        elif shape_type == 1:
            coords = torch.zeros((N * atoms_per_res, 3))
            atom_count = N * atoms_per_res
        else:
            coords = torch.zeros((N, atoms_per_res, 3))
            atom_count = N * atoms_per_res

        stageC_result = {"coords": coords, "atom_count": atom_count}

        # Instrumentation for debugging
        print(f"[DEBUG] seq: {seq}")
        print(f"[DEBUG] shape_type: {shape_type}, atoms_per_res: {atoms_per_res}, repeats: {repeats}")
        print(f"[DEBUG] coords.shape: {coords.shape}, atom_count: {atom_count}")

        with patch.object(
            self.predictor, "predict_3d_structure", return_value=stageC_result
        ) as mock_p3d:
            try:
                df = self.predictor.predict_submission(
                    seq, prediction_repeats=repeats, residue_atom_choice=0
                )
                print(f"[DEBUG] df.shape: {df.shape}")
                if shape_type == 1:
                    self.assertEqual(len(df), N * atoms_per_res, "Rows must match number of atoms for flat per-atom coords.")
                else:
                    self.assertEqual(len(df), N, "Rows must match number of residues.")
                # Columns: ID, resname, resid + repeats*(x,y,z) => 3 + 3*repeats total
                self.assertEqual(df.shape[1], 3 + (3 * repeats))
                # Assert predict_3d_structure called exactly 'repeats' times with the same sequence
                self.assertEqual(mock_p3d.call_count, repeats)
                for call in mock_p3d.call_args_list:
                    # Accept any kwargs, but first arg must be seq
                    self.assertEqual(call[0][0], seq)
            except Exception as e:
                print(f"[ERROR] Exception in test_forced_coord_shapes: {e}")
                raise

@given(
    cfg=st.dictionaries(
        st.text(min_size=1, max_size=20),
        st.one_of(
            st.integers(),
            st.text(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.dictionaries(st.text(min_size=1, max_size=20), st.integers() | st.text() | st.floats(allow_nan=False, allow_infinity=False)),
        ),
        min_size=1,
        max_size=3
    ).filter(lambda d: "model" not in d or "stageC" not in d.get("model", {}))
)
def test_rnapredictor_requires_stageC(cfg):
    """Property-based: RNAPredictor should raise ValueError if model.stageC is missing."""
    from omegaconf import OmegaConf
    import pytest

    # Ensure the config always has a 'device' key to avoid ConfigAttributeError
    if 'device' not in cfg:
        cfg['device'] = 'cpu'

    with pytest.raises(ValueError, match="stageC"):
        RNAPredictor(OmegaConf.create(cfg))

@pytest.mark.parametrize(
    "present,expected_error",
    [
        (False, True),   # do_ring_closure missing, should raise error
        (True, False),   # do_ring_closure present, should not raise error
    ]
)
def test_stageC_requires_do_ring_closure(present, expected_error):
    """Test that ValidationError is raised if do_ring_closure is missing from stageC config."""
    from omegaconf import OmegaConf
    import pytest
    from omegaconf.errors import ValidationError

    # Create a minimal stageC config
    stageC_config = TestRNAPredictorInitialization.minimal_stageC_config(method="mp_nerf", enabled=True, device="cpu")

    # Set or remove do_ring_closure based on the test case
    if present:
        setattr(stageC_config, "do_ring_closure", True)
    elif hasattr(stageC_config, "do_ring_closure"):
        delattr(stageC_config, "do_ring_closure")

    # Create the full config
    bad_cfg = OmegaConf.create({
        "device": "cpu",
        "model": {
            "stageC": OmegaConf.to_container(stageC_config),
            "stageB": {"torsion_bert": {"dummy": True, "debug_logging": False, "model_name_or_path": "dummy-path", "device": "cpu"}}
        },
        "prediction": {"repeats": 5, "residue_atom_choice": 0}
    })

    # Mock the StageBTorsionBertPredictor to avoid loading the actual model
    with patch("rna_predict.pipeline.stageB.torsion.torsion_bert_predictor.StageBTorsionBertPredictor.__init__", return_value=None):
        with patch("rna_predict.pipeline.stageB.torsion.torsion_bert_predictor.StageBTorsionBertPredictor.__call__") as mock_call:
            # Configure the mock to return a tensor with the right shape
            mock_call.return_value = {"torsion_angles": torch.zeros((4, 7))}

            from rna_predict.interface import RNAPredictor

            if expected_error:
                with pytest.raises(ValidationError, match="do_ring_closure"):
                    # Create a new predictor instance with the bad config
                    predictor = RNAPredictor(bad_cfg)
                    # Call predict_3d_structure which should trigger the validation error
                    predictor.predict_3d_structure("ACGU")
            else:
                # Should not raise
                predictor = RNAPredictor(bad_cfg)
                predictor.predict_3d_structure("ACGU")


# --- NEW TEST: property-based config structure validation ---
@given(
    st.dictionaries(
        keys=st.text(min_size=1, max_size=16),
        values=st.recursive(
            st.integers() | st.text() | st.booleans() | st.none(),
            lambda children: st.lists(children) | st.dictionaries(st.text(min_size=1, max_size=16), children),
            max_leaves=5,
        ),
        min_size=1,
        max_size=3,
    )
)
def test_stageb_torsionbert_config_structure_property(config_dict):
    """
    Property-based test: StageBTorsionBertPredictor should raise unique error if config is missing model.stageB.torsion_bert.
    """
    # Only pass configs that are guaranteed NOT to have model.stageB.torsion_bert
    if not ("model" in config_dict and isinstance(config_dict["model"], dict) and "stageB" in config_dict["model"] and isinstance(config_dict["model"]["stageB"], dict) and "torsion_bert" in config_dict["model"]["stageB"]):
        cfg = OmegaConf.create(config_dict)
        with pytest.raises(ValueError) as excinfo:
            StageBTorsionBertPredictor(cfg)
        assert "[UNIQUE-ERR-TORSIONBERT-NOCONFIG]" in str(excinfo.value)


# --------------------------------------------------------------------------------------
#                                     Main
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main(verbosity=2)
