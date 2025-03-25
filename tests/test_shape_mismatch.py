import pytest
import torch

@pytest.mark.xfail(reason="Shape mismatch bug: c_token=832 vs leftover normalized_shape=833")
def test_shape_mismatch_c_token_832_vs_833():
    """
    Reproduces the 'RuntimeError: Given normalized_shape=[833], 
    expected input with shape [*, 833], but got input of size [1, 10, 1664].'

    Steps:
      1) diffusion_config with c_token=832
      2) partial_coords shape [1, 10, 3]
      3) trunk_embeddings includes 'sing' or 's_trunk' at shape [1, 10, 384]
      4) call run_stageD_diffusion(...) which triggers the mismatch unless patched
    """
    diffusion_config = {
        "c_atom": 128,
        "c_s": 384,
        "c_z": 32,
        "c_token": 832, 
        "transformer": {"n_blocks": 4, "n_heads": 16},
    }

    partial_coords = torch.randn((1, 10, 3))

    trunk_embeddings = {
        "sing": torch.randn((1, 10, 384)),
        "pair": torch.randn((1, 10, 10, 32)),
    }

    from rna_predict.pipeline.stageD.run_stageD import run_stageD_diffusion

    with pytest.raises(RuntimeError, match=r"normalized_shape=\[833\].*got input of size.*1664"):
        run_stageD_diffusion(
            partial_coords,
            trunk_embeddings,
            diffusion_config,
            mode="inference",
            device="cpu"
        )

    # Once the bridging fix + dynamic layernorm are in place, 
    # remove @pytest.mark.xfail to verify it passes successfully.