import logging
import pytest
from rna_predict.pipeline.stageA.input_embedding.current.transformer.atom_attention.components.feature_processing import FeatureProcessor

def test_feature_processor_logging_capture(caplog):
    caplog.set_level(logging.DEBUG)
    fp = FeatureProcessor(
        c_atom=8,
        c_atompair=8,
        c_s=4,
        c_z=4,
        c_ref_element=16,
        debug_logging=True,
    )
    # Instantiation should trigger unconditional and conditional logs
    logs = caplog.text
    # Check for unconditional TEST log
    assert "TEST: FeatureProcessor constructed" in logs or "TEST: FeatureProcessor constructed (forced handler)" in logs
    # Check for debug_logging value
    assert "[FeatureProcessor] __init__ debug_logging=True" in logs
    # Check for the debug message about ref_element
    assert "[DEBUG][FeatureProcessor] ref_element expected dim: 16" in logs
    # Check for warning/error logs
    assert "TEST: FeatureProcessor WARNING" in logs or "TEST: FeatureProcessor WARNING (forced handler)" in logs
    assert "TEST: FeatureProcessor ERROR" in logs or "TEST: FeatureProcessor ERROR (forced handler)" in logs
