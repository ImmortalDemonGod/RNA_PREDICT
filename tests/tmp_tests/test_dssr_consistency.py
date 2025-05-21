import os
import numpy as np
import pytest
from rna_predict.dataset.preprocessing.angles import (
    extract_rna_torsions,
    _extract_rna_torsions_dssr,
)

# Reference project root, then dataset/examples directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
dataset_dir = os.path.join(project_root, 'rna_predict', 'dataset')
EXAMPLES_DIR = os.path.join(dataset_dir, 'examples')
SYN_PDB = os.path.abspath(os.path.join(EXAMPLES_DIR, "synthetic_cppc_0000001.pdb"))
CHAIN = 'B'

@ pytest.mark.parametrize('angle_set,expected_dim', [
    ('canonical', 7),
    ('full', 14),
])
def test_synthetic_pdb_dssr_direct(angle_set, expected_dim):
    """
    Test that DSSR directly extracts angles from synthetic PDB.
    """
    arr = _extract_rna_torsions_dssr(SYN_PDB, CHAIN, angle_set)
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 2
    assert arr.shape[1] == expected_dim

@ pytest.mark.parametrize('angle_set,expected_dim', [
    ('canonical', 7),
    ('full', 14),
])
def test_synthetic_pdb_dssr_matches_mdanalysis(angle_set, expected_dim):
    """
    Test that DSSR output matches MDAnalysis output on synthetic PDB.
    """
    a1 = extract_rna_torsions(SYN_PDB, CHAIN, backend='dssr', angle_set=angle_set)
    a2 = extract_rna_torsions(SYN_PDB, CHAIN, backend='mdanalysis', angle_set=angle_set)
    # Both should produce results
    assert a1 is not None and a2 is not None
    # Shapes should match
    assert a1.shape == a2.shape
    if angle_set == 'canonical':
        # Nan masks should match
        nan_mask = np.isnan(a1) == np.isnan(a2)
        assert nan_mask.all(), "Mismatch in NaN positions between DSSR and MDAnalysis"
        # Numeric consistency within tolerance
        valid = ~np.isnan(a1) & ~np.isnan(a2)
        diffs = np.abs(a1[valid] - a2[valid])
        assert diffs.size > 0
        max_diff = np.nanmax(diffs)
        assert max_diff < 0.5, f"Angle differences too large: max {max_diff}"
    else:
        # For full angles, only check shape and some finite values
        assert np.isfinite(a1).any() and np.isfinite(a2).any(), "No finite angles found"
