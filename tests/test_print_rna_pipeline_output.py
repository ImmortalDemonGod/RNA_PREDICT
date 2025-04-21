from rna_predict.print_rna_pipeline_output import print_tensor_example, setup_pipeline
from omegaconf import OmegaConf

def test_print_tensor_example():
    # Test with None tensor
    print_tensor_example("test_none", None)

    # Test with 1D tensor
    import numpy as np

    tensor_1d = np.array([1, 2, 3, 4, 5, 6])
    print_tensor_example("test_1d", tensor_1d)

    # Test with 2D tensor
    tensor_2d = np.array([[1, 2, 3], [4, 5, 6]])
    print_tensor_example("test_2d", tensor_2d)


def test_setup_pipeline():
    # Use a minimal valid config for setup_pipeline
    minimal_cfg = OmegaConf.create({
        "device": "cpu",
        "enable_stageC": True,
        "merge_latent": True,
        "init_z_from_adjacency": True,
        "model": {
            "stageA": {},
            "stageB": {},
            "stageC": {},
            "stageD": {},
        }
    })
    config, device = setup_pipeline(minimal_cfg)
    assert device == "cpu"
    assert "stageA_predictor" in config
    assert "torsion_bert_model" in config
    assert "pairformer_model" in config
    assert "merger" in config
    assert config["enable_stageC"] is True
    assert config["merge_latent"] is True
    assert config["init_z_from_adjacency"] is True
