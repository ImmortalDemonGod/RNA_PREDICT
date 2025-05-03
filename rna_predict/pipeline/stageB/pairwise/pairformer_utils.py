"""
Utility functions for the Pairformer module.
"""

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as np_strategies


def float_arrays(shape, min_value=-1.0, max_value=1.0):
    """
    A Hypothesis strategy for creating float32 NumPy arrays of a given shape
    within [min_value, max_value].

    Fixed to avoid subnormal float issues and float32 precision problems.
    """
    return np_strategies.arrays(
        dtype=np.float32,
        shape=shape,
        elements=st.floats(
            min_value=min_value,
            max_value=max_value,
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=False,
            width=32,
        ),
    )


def float_mask_arrays(shape):
    """A Hypothesis strategy for float32 arrays of 0.0 or 1.0."""
    return np_strategies.arrays(
        dtype=np.float32, shape=shape, elements=st.sampled_from([0.0, 1.0])
    )


def sample_msa_feature_dict_random_without_replacement(feature_dict, n_samples):
    """
    Sample n_samples sequences from the MSA feature dict without replacement.

    Args:
        feature_dict: Dictionary containing MSA features
        n_samples: Number of sequences to sample

    Returns:
        Dictionary with sampled sequences
    """
    if not feature_dict or "msa" not in feature_dict:
        return feature_dict

    msa = feature_dict["msa"]
    n_seqs = msa.shape[0]

    if n_seqs <= n_samples:
        return feature_dict

    # Sample indices without replacement
    sampled_indices = np.random.choice(n_seqs, n_samples, replace=False)

    # Create new feature dict with sampled sequences
    sampled_dict = {}
    for key, value in feature_dict.items():
        if key == "msa":
            sampled = np.take(value, sampled_indices, axis=0)
            # Guarantee shape preservation even for singleton axes
            sampled = np.reshape(sampled, (n_samples,) + value.shape[1:])
            sampled_dict[key] = sampled
        elif key in ["has_deletion", "deletion_value"]:
            if value is not None:
                sampled = np.take(value, sampled_indices, axis=0)
                # Guarantee shape preservation even for singleton axes
                sampled = np.reshape(sampled, (n_samples,) + value.shape[1:])
                sampled_dict[key] = sampled
        else:
            sampled_dict[key] = value

    return sampled_dict
