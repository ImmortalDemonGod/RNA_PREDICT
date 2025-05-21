import os
import numpy as np
import pytest
from rna_predict.predict import RNAPredictor
from hydra import initialize_config_dir, compose

@pytest.mark.slow
def test_real_rna_prediction_plausibility():
    # Force use of real model (bypass dummy logic)
    """
    Tests that the real RNAPredictor model produces plausible 3D backbone phosphorus atom coordinates for a short RNA sequence.
    
    The test ensures that predicted backbone P atom bond lengths are within a physically reasonable range and that no NaN values are present in the coordinates. It fails if atom metadata is inaccessible or if no backbone P atoms are found.
    """
    os.environ["FORCE_REAL_MODEL"] = "1"
    # Load Hydra config from absolute directory using experimental API
    conf_dir = "/Users/tomriddle1/RNA_PREDICT/rna_predict/conf"
    with initialize_config_dir(config_dir=conf_dir):
        cfg = compose(config_name="predict")
    predictor = RNAPredictor(cfg)
    sequence = "AUGCU"
    df = predictor.predict_submission(sequence, prediction_repeats=1)
    coords = df[["x_1", "y_1", "z_1"]].values

    # Try to access atom_metadata from the predictor's last result
    try:
        result = predictor.predict_3d_structure(sequence)
        atom_metadata = result.get('atom_metadata', {})
        atom_names = atom_metadata.get('atom_names', [])
        residue_indices = atom_metadata.get('residue_indices', [])

        # Filter for backbone P atoms
        backbone_p_indices = [i for i, (atom_name, residue_index) in enumerate(zip(atom_names, residue_indices)) if atom_name == 'P']
        if not backbone_p_indices:
            assert False, "No backbone P atoms found in atom_metadata!"

        backbone_p_coords = coords[backbone_p_indices]
        diffs = np.linalg.norm(backbone_p_coords[1:] - backbone_p_coords[:-1], axis=1)

        # Check for NaNs
        assert not np.isnan(backbone_p_coords).any(), "NaNs in backbone P coordinates!"
        # Check plausible bond lengths (phosphodiester backbone ~1.5-2.0 Å)
        assert np.all((diffs > 0.5) & (diffs < 10.0)), f"Implausible bond lengths: {diffs}"
    except Exception as e:
        assert False, f"Could not access atom_metadata: {e}"
    assert np.all((diffs > 0.5) & (diffs < 10.0)), f"Implausible bond lengths: {diffs}"
