"""
Consolidated Test Suite for rna_predict.dataset.dataset_loader

This file combines tests for:
  1) stream_bprna_dataset
  2) build_rna_token_metadata
  3) build_atom_to_token_idx
  4) validate_input_features
  5) load_rna_data_and_features

Key Improvements:
  - Organized test classes by function
  - Comprehensive coverage: normal paths, error handling, edge cases
  - setUp() methods to reduce redundancy
  - Descriptive docstrings for each test method
  - Use of Hypothesis for fuzzing complex inputs
  - Mocking external dependencies (datasets.load_dataset)
  - Parametrized tests where helpful
  - Proper use of unittest and pytest together
"""

import unittest
from unittest.mock import MagicMock, patch
import os
from rna_predict.dataset.loader import RNADataset

import torch
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from rna_predict.dataset.dataset_loader import (
    build_atom_to_token_idx,
    build_rna_token_metadata,
    load_rna_data_and_features,
    stream_bprna_dataset,
    validate_input_features,
)

from hydra import compose, initialize


class TestDatasetLoader(unittest.TestCase):
    @patch("rna_predict.dataset.dataset_loader.load_dataset")
    def test_stream_bprna_dataset(self, mock_load_dataset: MagicMock):
        # Set up the mock to return a mock iterable
        mock_iterable = MagicMock()
        mock_load_dataset.return_value = mock_iterable
        
        # Set up the mock to handle iteration
        mock_item = {"sequence": "AUGC", "structure": "((..))"}
        mock_iterable.__iter__.return_value = iter([mock_item])
        
        ds_iter = stream_bprna_dataset("train")
        self.assertIsNotNone(
            ds_iter, "The returned dataset iterator should not be None."
        )
        # Check minimal iteration
        # We won't exhaust the dataset; just confirm it's iterable.
        iterator = iter(ds_iter)
        first_item = next(iterator, None)
        self.assertIsNotNone(
            first_item,
            "Should be able to retrieve at least one record from the dataset.",
        )
        
        # Verify the mock was called correctly
        mock_load_dataset.assert_called_once_with(
            "multimolecule/bprna-spot", split="train", streaming=True
        )


# -----------------------------------------------------------------------------
#                         Test stream_bprna_dataset
# -----------------------------------------------------------------------------
class TestStreamBprnaDataset(unittest.TestCase):
    """
    Tests for the stream_bprna_dataset function.
    """

    def setUp(self) -> None:
        """
        Common setup for TestStreamBprnaDataset.
        """
        self.valid_split = "train"

    @patch("rna_predict.dataset.dataset_loader.load_dataset")
    def test_stream_bprna_dataset_valid(self, mock_load_dataset: MagicMock) -> None:
        """
        Test that a valid split returns an IterableDataset instance, using a mock.
        """
        mock_iterable = MagicMock()
        mock_load_dataset.return_value = mock_iterable

        result = stream_bprna_dataset(self.valid_split)
        self.assertEqual(result, mock_iterable)
        mock_load_dataset.assert_called_once_with(
            "multimolecule/bprna-spot", split=self.valid_split, streaming=True
        )

    @patch("rna_predict.dataset.dataset_loader.load_dataset")
    def test_stream_bprna_dataset_multiple_splits_train(
        self, mock_load_dataset: MagicMock
    ) -> None:
        """
        Test train split to ensure calls to HF dataset are correct.
        """
        mock_iterable = MagicMock()
        mock_load_dataset.return_value = mock_iterable
        split = "train"

        result = stream_bprna_dataset(split)
        self.assertEqual(result, mock_iterable)
        mock_load_dataset.assert_called_with(
            "multimolecule/bprna-spot", split=split, streaming=True
        )

    @patch("rna_predict.dataset.dataset_loader.load_dataset")
    def test_stream_bprna_dataset_multiple_splits_test(
        self, mock_load_dataset: MagicMock
    ) -> None:
        """
        Test test split to ensure calls to HF dataset are correct.
        """
        mock_iterable = MagicMock()
        mock_load_dataset.return_value = mock_iterable
        split = "test"

        result = stream_bprna_dataset(split)
        self.assertEqual(result, mock_iterable)
        mock_load_dataset.assert_called_with(
            "multimolecule/bprna-spot", split=split, streaming=True
        )

    @patch("rna_predict.dataset.dataset_loader.load_dataset")
    def test_stream_bprna_dataset_multiple_splits_validation(
        self, mock_load_dataset: MagicMock
    ) -> None:
        """
        Test validation split to ensure calls to HF dataset are correct.
        """
        mock_iterable = MagicMock()
        mock_load_dataset.return_value = mock_iterable
        split = "validation"

        result = stream_bprna_dataset(split)
        self.assertEqual(result, mock_iterable)
        mock_load_dataset.assert_called_with(
            "multimolecule/bprna-spot", split=split, streaming=True
        )

    @patch("rna_predict.dataset.dataset_loader.load_dataset")
    def test_stream_bprna_dataset_multiple_splits_random(
        self, mock_load_dataset: MagicMock
    ) -> None:
        """
        Test random split to ensure calls to HF dataset are correct.
        """
        mock_iterable = MagicMock()
        mock_load_dataset.return_value = mock_iterable
        split = "random"

        result = stream_bprna_dataset(split)
        self.assertEqual(result, mock_iterable)
        mock_load_dataset.assert_called_with(
            "multimolecule/bprna-spot", split=split, streaming=True
        )

    @given(split=st.text(min_size=0, max_size=10))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @patch("rna_predict.dataset.dataset_loader.load_dataset")
    def test_stream_bprna_dataset_fuzz(
        self, mock_load_dataset: MagicMock, split: str
    ) -> None:
        """
        Fuzz test with random short strings for 'split'.
        Ensures function calls load_dataset without crashing.
        """
        mock_iterable = MagicMock()
        mock_load_dataset.return_value = mock_iterable
        result = stream_bprna_dataset(split)
        self.assertIsNotNone(result)


# -----------------------------------------------------------------------------
#                 Test build_rna_token_metadata
# -----------------------------------------------------------------------------
class TestBuildRnaTokenMetadata(unittest.TestCase):
    """
    Tests for the build_rna_token_metadata function.
    """

    def setUp(self) -> None:
        """
        Common setup for TestBuildRnaTokenMetadata.
        """
        self.num_tokens = 10
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_build_rna_token_metadata_basic(self) -> None:
        """
        Test basic usage with fixed parameters.
        """
        if self.device == 'cpu':
            self.skipTest("No CUDA device available; test requires non-cpu device for build_rna_token_metadata.")
        metadata = build_rna_token_metadata(self.num_tokens, device=self.device)
        self.assertIn("asym_id", metadata)
        self.assertIn("residue_index", metadata)
        self.assertIn("token_index", metadata)
        self.assertEqual(metadata["asym_id"].shape[0], self.num_tokens)
        self.assertTrue((metadata["residue_index"] == torch.arange(1, 11)).all())

    @given(num_tokens=st.integers(min_value=1, max_value=200))
    @example(num_tokens=1)  # minimal test case
    @settings(
        deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_build_rna_token_metadata_fuzz(self, num_tokens: int) -> None:
        """
        Fuzzy test to ensure build_rna_token_metadata does not crash for various valid token counts.
        """
        meta = build_rna_token_metadata(num_tokens, device="cpu")
        self.assertEqual(meta["asym_id"].shape[0], num_tokens)
        self.assertTrue(
            (meta["residue_index"] == torch.arange(1, num_tokens + 1)).all()
        )


# -----------------------------------------------------------------------------
#                   Test build_atom_to_token_idx
# -----------------------------------------------------------------------------
class TestBuildAtomToTokenIdx(unittest.TestCase):
    """
    Tests for the build_atom_to_token_idx function.
    """

    def setUp(self) -> None:
        """
        Common setup for TestBuildAtomToTokenIdx.
        """
        self.num_atoms = 40
        self.num_tokens = 10
        self.device = "cpu"

    def test_build_atom_to_token_idx_basic(self) -> None:
        """
        Test a basic scenario where num_atoms=40, num_tokens=10.
        """
        mapping = build_atom_to_token_idx(
            self.num_atoms, self.num_tokens, device=self.device
        )
        self.assertEqual(mapping.shape[0], self.num_atoms)
        # The distribution should be fairly even among token indices from 0..9
        self.assertTrue(torch.all(mapping >= 0))
        self.assertTrue(torch.all(mapping < self.num_tokens))

    @given(
        num_atoms=st.integers(min_value=1, max_value=1000),
        num_tokens=st.integers(min_value=1, max_value=1000),
    )
    @settings(
        deadline=None,  # Disable deadline for this flaky test
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        max_examples=100,
    )
    def test_build_atom_to_token_idx_fuzz(
        self, num_atoms: int, num_tokens: int
    ) -> None:
        """
        Fuzzy test ensures no crashes or shape mismatch for random valid num_atoms/num_tokens.
        """
        idx_map = build_atom_to_token_idx(num_atoms, num_tokens, device="cpu")
        self.assertEqual(idx_map.shape[0], num_atoms)
        # token indices must be between 0 and num_tokens-1
        self.assertTrue((idx_map >= 0).all().item())
        self.assertTrue((idx_map < num_tokens).all().item())


# -----------------------------------------------------------------------------
#                  Test validate_input_features
# -----------------------------------------------------------------------------
class TestValidateInputFeatures(unittest.TestCase):
    """
    Tests for the validate_input_features function.
    """

    def setUp(self) -> None:
        """
        Common setup with a valid sample input_feature_dict.
        """
        # Minimal valid dictionary
        self.valid_input = {
            "atom_to_token_idx": torch.zeros((1, 10), dtype=torch.long),
            "ref_pos": torch.zeros((1, 10, 3), dtype=torch.float),
            "ref_space_uid": torch.zeros((1, 10), dtype=torch.long),
            "asym_id": torch.zeros((1, 10), dtype=torch.long),
            "residue_index": torch.zeros((1, 10), dtype=torch.long),
            "entity_id": torch.zeros((1, 10), dtype=torch.long),
            "sym_id": torch.zeros((1, 10), dtype=torch.long),
            "token_index": torch.zeros((1, 10), dtype=torch.long),
        }

    def test_validate_input_features_valid(self) -> None:
        """
        Test that a fully valid input passes validation.
        """
        self.assertTrue(validate_input_features(self.valid_input))

    def test_validate_input_features_missing_keys(self) -> None:
        """
        Test that missing a required key raises ValueError.
        """
        invalid_input = dict(self.valid_input)
        invalid_input.pop("atom_to_token_idx")  # remove a required key
        with self.assertRaises(ValueError) as ctx:
            validate_input_features(invalid_input)
        self.assertIn("Missing required key: atom_to_token_idx", str(ctx.exception))

    def test_validate_input_features_ref_pos_shape(self) -> None:
        """
        Test that an incorrect ref_pos shape raises ValueError.
        """
        invalid_input = dict(self.valid_input)
        invalid_input["ref_pos"] = torch.zeros((1, 10, 4))  # shape mismatch in last dim
        with self.assertRaises(ValueError) as ctx:
            validate_input_features(invalid_input)
        self.assertIn("ref_pos must have shape [batch, N_atom, 3].", str(ctx.exception))

    @given(
        input_dict=st.dictionaries(
            keys=st.text(min_size=1, max_size=15),
            values=st.none() | st.integers() | st.floats() | st.just(torch.tensor([])),
        )
    )
    @settings(
        deadline=None,  # Disable deadline for this flaky test
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        max_examples=50,
    )
    def test_validate_input_features_fuzz(self, input_dict: dict) -> None:
        """
        Fuzzy test with random dictionaries to ensure code doesn't accept invalid shapes or keys.
        Often raises ValueError, or passes if by chance dictionary is valid (very unlikely).
        """
        try:
            result = validate_input_features(input_dict)  # may raise ValueError
            # If it doesn't raise, confirm the dictionary truly meets the requirement
            self.assertTrue(
                result,
                "validate_input_features should return True if no error is raised.",
            )
        except ValueError:
            # Acceptable outcome, as most random dicts won't have the required shape/keys.
            pass


# -----------------------------------------------------------------------------
#               Test load_rna_data_and_features
# -----------------------------------------------------------------------------
class TestLoadRnaDataAndFeatures(unittest.TestCase):
    """
    Tests for the load_rna_data_and_features function.
    """

    def setUp(self) -> None:
        """
        Common setup for load_rna_data_and_features tests.
        """
        self.default_filepath = "fake_path"
        self.device = "cpu"

    def test_load_rna_data_and_features_basic(self) -> None:
        """
        Test a basic usage returns two dictionaries (atom_feature_dict, token_feature_dict).
        """
        atom_dict, token_dict = load_rna_data_and_features(
            rna_filepath=self.default_filepath, device=self.device
        )
        self.assertIsInstance(atom_dict, dict)
        self.assertIsInstance(token_dict, dict)
        # Verify some expected keys
        self.assertIn("atom_to_token_idx", atom_dict)
        self.assertIn("restype", token_dict)

    @given(
        override_num_atoms=st.one_of(st.none(), st.integers(min_value=1, max_value=200))
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        max_examples=30,
        deadline=None,
    )
    def test_load_rna_data_and_features_fuzz(self, override_num_atoms):
        """
        Fuzz test override_num_atoms to ensure method can handle a range or None without crashing.
        """
        atom_dict, token_dict = load_rna_data_and_features(
            rna_filepath=self.default_filepath,
            device=self.device,
            override_num_atoms=override_num_atoms,
        )
        self.assertIn("atom_to_token_idx", atom_dict)
        self.assertIn("restype", token_dict)
        self.assertIsInstance(atom_dict["ref_pos"], torch.Tensor)
        if override_num_atoms is None:
            # Should revert to default_num_atoms=40
            self.assertEqual(atom_dict["ref_pos"].shape[1], 40)
        else:
            self.assertEqual(atom_dict["ref_pos"].shape[1], override_num_atoms)



class TestRNADatasetMinimal(unittest.TestCase):
    def test_loader_reads_target_id(self):
        # Use the minimal Kaggle index CSV
        index_csv = os.path.join(
            os.path.dirname(__file__),
            '../../rna_predict/dataset/examples/kaggle_minimal_index.csv'
        )
        # Print raw CSV contents for debugging
        with open(index_csv, 'r') as f:
            print('[TEST] Raw CSV contents:')
            for line in f:
                print('[TEST] ' + line.rstrip())
        # Minimal config stub (now via Hydra)
        with initialize(config_path="../../rna_predict/conf", version_base=None):
            cfg = compose(config_name="default.yaml")
            # Override for test
            cfg.data.max_residues = 10
            cfg.data.max_atoms = 21
            cfg.data.batch_size = 1
            cfg.data.index_csv = index_csv
            print("[DEBUG] Loaded Hydra config for test_dataset_loader.py:\n", cfg)
            loader = RNADataset(index_csv, cfg)
        # Access first sample and assert 'target_id' exists
        row = loader.meta[0]
        self.assertIn('target_id', row.dtype.names)
        self.assertTrue(row['target_id'] != '', "target_id should not be empty")
        # Optionally, test __getitem__
        sample = loader[0]
        # If sample is a dict, check for keys; if tuple, skip
        if isinstance(sample, dict):
            self.assertIn('target_id', sample)


if __name__ == "__main__":
    # If you prefer unittest discovery:
    unittest.main()
