from rna_predict.print_rna_pipeline_output import print_tensor_example, setup_pipeline


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
    config, device = setup_pipeline()
    assert device == "cpu"
    assert "stageA_predictor" in config
    assert "torsion_bert_model" in config
    assert "pairformer_model" in config
    assert "merger" in config
    assert config["enable_stageC"] is True
    assert config["merge_latent"] is True
    assert config["init_z_from_adjacency"] is True
