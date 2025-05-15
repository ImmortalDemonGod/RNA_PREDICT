import pytest
import numpy as np
from omegaconf import OmegaConf

from rna_predict.predict import RNAPredictor

@pytest.fixture
def minimal_torsion_cfg():
    # Minimal config for RNAPredictor streamline mode
    return OmegaConf.create({
        'device': 'cpu',
        'model': {
            'stageB': {
                'torsion_bert': {
                    'model_name_or_path': 'sayby/rna_torsionbert',
                    'device': 'cpu',
                    'angle_mode': 'sin_cos',
                    'num_angles': 7,
                    'max_length': 32,
                    'init_from_scratch': True  # Use dummy model for speed
                }
            },
            'stageC': {
                'enabled': True,
                'device': 'cpu',
                'angle_mode': 'sin_cos',
                'num_angles': 7,
                'max_length': 32,
                'init_from_scratch': True
            }
        },
        'default_repeats': 5,
        'default_atom_choice': 0
    })

@pytest.mark.parametrize("seed_mode", [
    (None, "unique"),
    ([42]*5, "identical"),
    ([1,2,3,4,5], "reproducible")
])
def test_stochastic_predict_submission(minimal_torsion_cfg, seed_mode):
    seeds, mode = seed_mode
    predictor = RNAPredictor(minimal_torsion_cfg)
    sequence = "AUGCU"  # Short test sequence
    df = predictor.predict_submission(sequence, prediction_repeats=5, repeat_seeds=seeds)
    coords = np.stack([df[[f"x_{i}", f"y_{i}", f"z_{i}"]].values for i in range(1,6)], axis=0)  # [5, N, 3]
    if mode == "unique":
        # All repeats should be different
        diffs = [np.any(np.not_equal(coords[i], coords[j])) for i in range(5) for j in range(i+1,5)]
        assert all(diffs), "Some repeats are not unique!"
    elif mode == "identical":
        # All repeats should be identical
        for i in range(1,5):
            np.testing.assert_allclose(coords[0], coords[i], err_msg=f"Repeat {i+1} differs with fixed seed!")
    elif mode == "reproducible":
        # Rerun with same seeds and check for exact match
        df2 = predictor.predict_submission(sequence, prediction_repeats=5, repeat_seeds=seeds)
        coords2 = np.stack([df2[[f"x_{i}", f"y_{i}", f"z_{i}"]].values for i in range(1,6)], axis=0)
        np.testing.assert_allclose(coords, coords2, err_msg="Predictions not reproducible with same seeds!")
