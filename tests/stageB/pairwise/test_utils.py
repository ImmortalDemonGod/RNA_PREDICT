"""
Test utilities and shared functions for pairformer tests.
"""

import os
import psutil
import numpy as np
import torch
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as np_strategies
from typing import Dict, Any, Optional

# ------------------------------------------------------------------------
# HELPER STRATEGIES & FUNCTIONS
# ------------------------------------------------------------------------

# To avoid repeated warnings about large or slow generation:
settings.register_profile(
    "extended",
    suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow],
    deadline=None,
)
settings.load_profile("extended")


def bool_arrays(shape):
    """A Hypothesis strategy for boolean arrays of a given shape."""
    return np_strategies.arrays(dtype=np.bool_, shape=shape)


def float_arrays(shape):
    """A Hypothesis strategy for float arrays of a given shape."""
    return np_strategies.arrays(dtype=np.float32, shape=shape)


def float_mask_arrays(shape):
    """A Hypothesis strategy for float mask arrays of a given shape."""
    return np_strategies.arrays(dtype=np.float32, shape=shape)


@st.composite
def s_z_mask_draw(
    draw, c_s_range=(0, 16), c_z_range=(4, 16), n_token_range=(1, 3), batch_range=(1, 1)
):
    """
    Produces random (s_in, z_in, mask) plus c_s, c_z:
        - s_in shape: (batch, n_token, c_s) or None if c_s=0
        - z_in shape: (batch, n_token, n_token, c_z)
        - mask shape: (batch, n_token, n_token), as float 0.0 or 1.0
    Also ensures if c_s>0 and n_heads=2, c_s is multiple of 2.
    """
    batch = draw(st.integers(*batch_range))
    n_token = draw(st.integers(*n_token_range))
    c_s_candidate = draw(st.integers(*c_s_range))
    if c_s_candidate > 0:
        # Round up to multiple of 2
        c_s_candidate = (c_s_candidate // 2) * 2
        if c_s_candidate == 0:
            c_s_candidate = 2
    c_s = c_s_candidate

    c_z = draw(st.integers(*c_z_range))

    if c_s > 0:
        s_array = draw(float_arrays((batch, n_token, c_s)))
    else:
        s_array = None

    z_array = draw(float_arrays((batch, n_token, n_token, c_z)))
    # Produce mask in [0,1] float
    mask_array = draw(float_mask_arrays((batch, n_token, n_token)))

    s_tensor = torch.from_numpy(s_array) if s_array is not None else None
    z_tensor = torch.from_numpy(z_array)
    mask_tensor = torch.from_numpy(mask_array)
    return s_tensor, z_tensor, mask_tensor, c_s, c_z


def get_memory_usage():
    """Get current memory usage of the Python process."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return {
        "rss": mem_info.rss / 1024 / 1024,  # RSS in MB
        "vms": mem_info.vms / 1024 / 1024,  # VMS in MB
    }


def create_feature_dict_for_sampling(
    n_seqs: int, n_res: int, include_deletions: bool = True, other_key: bool = True
) -> Dict[str, Optional[Any]]:
    """Creates a sample feature dictionary for sampling tests."""
    feature_dict: Dict[str, Optional[Any]] = {
        "msa": np.random.rand(n_seqs, n_res, 10).astype(np.float32)
    }
    if include_deletions:
        feature_dict["has_deletion"] = np.random.randint(
            0, 2, size=(n_seqs, n_res)
        ).astype(np.float32)
        feature_dict["deletion_value"] = np.random.rand(n_seqs, n_res).astype(
            np.float32
        )
    else:
        feature_dict["has_deletion"] = None

    if other_key:
        feature_dict["other_data"] = np.array([1, 2, 3])

    return feature_dict 