import pytest
import numpy as np
from hydra import initialize_config_dir, compose

from rna_predict.predict import RNAPredictor

@pytest.fixture
def minimal_torsion_cfg():
    # Minimal Hydra-composed config for RNAPredictor streamline mode
    with initialize_config_dir(config_dir="/Users/tomriddle1/RNA_PREDICT/rna_predict/conf", version_base="1.1", job_name="test_torsion"):
        cfg = compose(
            config_name="predict",
            overrides=[
                "device=cpu",
                "model.stageB.torsion_bert.model_name_or_path=sayby/rna_torsionbert",
                "model.stageB.torsion_bert.device=cpu",
                "model.stageB.torsion_bert.angle_mode=sin_cos",
                "model.stageB.torsion_bert.num_angles=7",
                "model.stageB.torsion_bert.max_length=32",
                "model.stageB.torsion_bert.init_from_scratch=false",
                "model.stageC.enabled=true",
                "model.stageC.method=mp_nerf",
                "model.stageC.do_ring_closure=false",
                "model.stageC.place_bases=true",
                # Value contains special char, wrap in quotes per Hydra override grammar
                "model.stageC.sugar_pucker=\"C3'-endo\"",
                "model.stageC.device=cpu",
                "model.stageC.debug_logging=true",
                "model.stageC.angle_representation=degrees",
                "model.stageC.use_metadata=false",
                "model.stageC.use_memory_efficient_kernel=false",
                "model.stageC.use_deepspeed_evo_attention=false",
                "model.stageC.use_lma=false",
                "model.stageC.inplace_safe=false",
                "model.stageC.chunk_size=null",
                "run_stageD=false",
            ],
        )
    # Patch StageBTorsionBertPredictor to return random torsion angles for stochastic behavior
    from rna_predict.pipeline.stageB.torsion.torsion_bert_predictor import StageBTorsionBertPredictor
    def dummy_call(self, sequence, stochastic_pass=False, seed=None):
        import torch
        n = len(sequence)
        dim = 14
        if seed is not None:
            torch.manual_seed(seed)
        return {"torsion_angles": torch.randn(n, dim)}
    StageBTorsionBertPredictor.__call__ = dummy_call
    return cfg

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
        print(f"coords.shape: {coords.shape}")
        print(f"diffs: {diffs}")
        for i in range(5):
            for j in range(i+1,5):
                print(f"np.not_equal(coords[{i}], coords[{j}]):", np.not_equal(coords[i], coords[j]).sum())
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
