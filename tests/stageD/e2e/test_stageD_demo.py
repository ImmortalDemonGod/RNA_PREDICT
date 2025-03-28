import pytest
from rna_predict.pipeline.stageB.pairwise.main import demo_run_diffusion

def test_demo_run_diffusion_config():
    """
    Ensures the corrected diffusion_config does not raise the 'unexpected keyword' TypeError.
    """
    try:
        demo_run_diffusion()
    except TypeError as exc:
        pytest.fail(f"Unexpected TypeError after fixing config: {exc}")