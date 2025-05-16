import os
import numpy as np
import pytest
from rna_predict.predict import RNAPredictor
from hydra import initialize_config_dir, compose

@pytest.mark.slow
def test_real_rna_prediction_plausibility():
    # Force use of real model (bypass dummy logic)
    os.environ["FORCE_REAL_MODEL"] = "1"
    # Load Hydra config from absolute directory using experimental API
    conf_dir = "/Users/tomriddle1/RNA_PREDICT/rna_predict/conf"
    with initialize_config_dir(config_dir=conf_dir):
        cfg = compose(config_name="predict")
    predictor = RNAPredictor(cfg)
    sequence = "AUGCU"
    df = predictor.predict_submission(sequence, prediction_repeats=1)
    coords = df[["x_1", "y_1", "z_1"]].values
    # Check for NaNs
    assert not np.isnan(coords).any(), "NaNs in coordinates!"
    # Check plausible bond lengths (phosphodiester backbone ~1.5-2.0 Å)
    diffs = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
    assert np.all((diffs > 0.5) & (diffs < 10.0)), f"Implausible bond lengths: {diffs}"
