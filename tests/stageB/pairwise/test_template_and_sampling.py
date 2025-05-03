"""
Tests for TemplateEmbedder and MSA sampling functionality.
"""

import unittest
import pytest
import numpy as np
import torch

from rna_predict.pipeline.stageB.pairwise.pairformer import (
    TemplateEmbedder,
    sample_msa_feature_dict_random_without_replacement,
)
from tests.stageB.pairwise.test_utils import create_feature_dict_for_sampling
from rna_predict.conf.config_schema import TemplateEmbedderConfig

# Define unique error identifiers
ERROR_TEMPLATE_MISSING_CONFIG = "ERR_TEMPLATE_001"
ERROR_SAMPLE_EMPTY_DICT = "ERR_SAMPLE_001"
ERROR_SAMPLE_MISSING_MSA = "ERR_SAMPLE_002"
ERROR_SAMPLE_SHAPE_MISMATCH = "ERR_SAMPLE_003"
ERROR_SAMPLE_NONE_DELETION = "ERR_SAMPLE_004"

class TestTemplateEmbedder(unittest.TestCase):
    """
    Tests for TemplateEmbedder, verifying:
        • n_blocks=0 => returns 0
        • missing 'template_restype' => returns 0
        • presence of 'template_restype' but code is not fully implemented => also returns 0
    """

    def test_instantiate_basic(self):
        # Use minimal required config values
        minimal_cfg = TemplateEmbedderConfig(
            n_blocks=2,
            c=8,
            c_z=16,
            dropout=0.0
        )
        te = TemplateEmbedder(minimal_cfg)
        self.assertIsInstance(te, TemplateEmbedder)

    def test_forward_no_template_restype(self):
        cfg = TemplateEmbedderConfig(n_blocks=2, c=8, c_z=16, dropout=0.0)
        embedder = TemplateEmbedder(cfg)
        z_in = torch.randn((1, 4, 4, 16), dtype=torch.float32)
        pair_mask = torch.ones((1, 4, 4), dtype=torch.bool)
        out = embedder.forward({}, z_in, pair_mask=pair_mask)
        self.assertTrue(torch.equal(out, torch.zeros_like(z_in)))

    def test_forward_nblocks_zero(self):
        cfg = TemplateEmbedderConfig(n_blocks=0, c=8, c_z=16, dropout=0.0)
        embedder = TemplateEmbedder(cfg)
        z_in = torch.randn((1, 4, 4, 16), dtype=torch.float32)
        out = embedder.forward({"template_restype": torch.zeros((1, 4))}, z_in)
        self.assertTrue(torch.equal(out, torch.zeros_like(z_in)))

    def test_forward_template_present(self):
        """
        Even if 'template_restype' is present, the current logic returns 0
        unless there's a deeper implementation. Checking coverage only.
        """
        cfg = TemplateEmbedderConfig(n_blocks=2, c=8, c_z=16, dropout=0.0)
        embedder = TemplateEmbedder(cfg)
        input_dict = {"template_restype": torch.zeros((1, 4))}
        z_in = torch.randn((1, 4, 4, 16), dtype=torch.float32)
        out = embedder.forward(input_dict, z_in)
        self.assertTrue(torch.equal(out, torch.zeros_like(z_in)))


def test_sample_empty_dict():
    """
    Test sampling with an empty feature dictionary.
    Covers line 46 (if not feature_dict).
    Expects the original empty dictionary to be returned.
    """
    feature_dict = {}
    n_samples = 5
    result = sample_msa_feature_dict_random_without_replacement(feature_dict, n_samples)
    assert result == {}
    # Explicitly check it's the *same* object in this case
    assert id(result) == id(feature_dict)


def test_sample_dict_missing_msa():
    """
    Test sampling with a feature dictionary missing the 'msa' key.
    Covers line 46 ("msa" not in feature_dict).
    Expects the original dictionary to be returned.
    """
    feature_dict = {"other_key": np.array([1])}
    n_samples = 5
    result = sample_msa_feature_dict_random_without_replacement(feature_dict, n_samples)
    assert result == feature_dict
    assert id(result) == id(feature_dict)  # Should return the original dict


@pytest.mark.parametrize(
    "n_seqs, n_samples",
    [
        (10, 10),  # n_seqs == n_samples
        (5, 10),  # n_seqs < n_samples
    ],
)
def test_sample_n_samples_ge_n_seqs(n_seqs: int, n_samples: int):
    """
    Test sampling when n_samples is >= number of sequences in MSA.
    Covers lines 52-53.
    Expects the original feature dictionary to be returned.
    """
    n_res = 20
    feature_dict = create_feature_dict_for_sampling(n_seqs, n_res)
    # --- ADDED ASSERTIONS ---
    assert feature_dict["msa"] is not None
    assert feature_dict["has_deletion"] is not None
    assert feature_dict["deletion_value"] is not None
    assert feature_dict["other_data"] is not None
    # --- END ADDED ASSERTIONS ---
    original_msa_id = id(feature_dict["msa"])

    result = sample_msa_feature_dict_random_without_replacement(feature_dict, n_samples)

    # Check the whole dictionary is returned (deep comparison might be needed for arrays)
    assert result.keys() == feature_dict.keys()
    # --- ADDED ASSERTIONS ---
    assert result["msa"] is not None
    assert result["has_deletion"] is not None
    assert result["deletion_value"] is not None
    assert result["other_data"] is not None
    # --- END ADDED ASSERTIONS ---
    np.testing.assert_array_equal(result["msa"], feature_dict["msa"])
    np.testing.assert_array_equal(result["has_deletion"], feature_dict["has_deletion"])
    np.testing.assert_array_equal(
        result["deletion_value"], feature_dict["deletion_value"]
    )
    np.testing.assert_array_equal(result["other_data"], feature_dict["other_data"])

    # Importantly, check it returns the *original* dictionary object
    assert id(result) == id(feature_dict)
    assert id(result["msa"]) == original_msa_id


def test_sample_successful_sampling_all_keys():
    """
    Test successful sampling when n_samples < n_seqs.
    Covers lines 56-69, including specific handling for 'msa',
    'has_deletion', 'deletion_value', and other keys.
    Expects a new dictionary with sampled arrays.
    """
    n_seqs = 10
    n_res = 20
    n_samples = 5
    assert n_seqs > n_samples  # Precondition for this test path

    feature_dict = create_feature_dict_for_sampling(
        n_seqs, n_res, include_deletions=True, other_key=True
    )
    original_other_data = feature_dict["other_data"]  # Keep a reference

    result = sample_msa_feature_dict_random_without_replacement(feature_dict, n_samples)

    # Check it's a new dictionary
    assert id(result) != id(feature_dict)

    # Check keys are preserved
    assert result.keys() == feature_dict.keys()

    # Check shapes of sampled arrays
    assert result["msa"].shape[0] == n_samples
    assert (
        result["msa"].shape[1:] == feature_dict["msa"].shape[1:]
    )  # Other dims unchanged
    assert result["has_deletion"].shape[0] == n_samples
    assert result["has_deletion"].shape[1:] == feature_dict["has_deletion"].shape[1:]
    assert result["deletion_value"].shape[0] == n_samples
    assert (
        result["deletion_value"].shape[1:] == feature_dict["deletion_value"].shape[1:]
    )

    # Check other data is untouched (same object ID)
    assert id(result["other_data"]) == id(original_other_data)
    np.testing.assert_array_equal(result["other_data"], original_other_data)

    # Optional: Verify sampled data comes from original (more complex)
    # This is hard due to randomness, but we can check if rows exist in the original
    original_msa_rows = [row.tobytes() for row in feature_dict["msa"]]
    sampled_msa_rows = [row.tobytes() for row in result["msa"]]
    assert len(set(sampled_msa_rows)) == n_samples  # Ensure unique rows were sampled
    for row_bytes in sampled_msa_rows:
        assert row_bytes in original_msa_rows


def test_sample_successful_sampling_none_deletion():
    """
    Test successful sampling when deletion-related keys are None or absent.
    Covers lines 56-69, specifically line 64 (value is None).
    Expects a new dictionary with sampled MSA and untouched other keys.
    The key 'has_deletion' should NOT be present if its input value was None.
    """
    n_seqs = 10
    n_res = 20
    n_samples = 5
    assert n_seqs > n_samples

    # Create dict where 'has_deletion' is None, 'deletion_value' might be absent
    feature_dict = create_feature_dict_for_sampling(
        n_seqs, n_res, include_deletions=False, other_key=True
    )
    assert feature_dict["has_deletion"] is None
    assert "deletion_value" not in feature_dict  # Based on helper function logic

    original_other_data = feature_dict["other_data"]

    result = sample_msa_feature_dict_random_without_replacement(feature_dict, n_samples)

    # Check it's a new dictionary
    assert id(result) != id(feature_dict)

    # Check keys are preserved (or handled correctly if None/absent)
    assert "msa" in result
    # --- MODIFIED ASSERTION ---
    # If the input value for 'has_deletion' was None, it should NOT be in the output
    assert "has_deletion" not in result
    # --- END MODIFIED ASSERTION ---
    assert "other_data" in result
    # 'deletion_value' was absent in input, should remain absent
    assert "deletion_value" not in result

    # Check shapes of sampled arrays
    assert result["msa"].shape[0] == n_samples
    assert result["msa"].shape[1:] == feature_dict["msa"].shape[1:]

    # Check other data is untouched
    assert id(result["other_data"]) == id(original_other_data)
    np.testing.assert_array_equal(result["other_data"], original_other_data)


if __name__ == "__main__":
    unittest.main()