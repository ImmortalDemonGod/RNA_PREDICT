import subprocess
import sys
import os
import tempfile
import textwrap
import pytest

def test_feature_processor_logging_subprocess():
    # Create a temp python script that instantiates FeatureProcessor and logs
    script = textwrap.dedent('''
        import logging
        from rna_predict.pipeline.stageA.input_embedding.current.transformer.atom_attention.components.feature_processing import FeatureProcessor
        import sys
        logger = logging.getLogger("rna_predict.pipeline.stageA.input_embedding.current.transformer.atom_attention.components.feature_processing")
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('[%(levelname)s][%(name)s] %(message)s')
        handler.setFormatter(formatter)
        if not logger.hasHandlers():
            logger.addHandler(handler)
        logger.propagate = True
        fp = FeatureProcessor(
            c_atom=8,
            c_atompair=8,
            c_s=4,
            c_z=4,
            c_ref_element=16,
            debug_logging=True,
        )
    ''')
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
        tf.write(script)
        tf.flush()
        tfname = tf.name
    # Run the script in a subprocess using uv
    result = subprocess.run([
        sys.executable, "-m", "uv", "run", tfname
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    os.unlink(tfname)
    # Check that logger output appears in stdout
    stdout = result.stdout
    stderr = result.stderr
    assert "FeatureProcessor constructed" in stdout or "FeatureProcessor constructed" in stderr, f"No logger output in subprocess.\nstdout:\n{stdout}\nstderr:\n{stderr}"
    assert "ref_element expected dim: 16" in stdout or "ref_element expected dim: 16" in stderr, f"No conditional debug log in subprocess.\nstdout:\n{stdout}\nstderr:\n{stderr}"
