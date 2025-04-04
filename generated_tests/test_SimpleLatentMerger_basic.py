import run_full_pipeline
import unittest
from hypothesis import given, strategies as st

class TestFuzzSimplelatentmerger(unittest.TestCase):

    @given(dim_angles=st.integers(), dim_s=st.integers(), dim_z=st.integers(), dim_out=st.integers())
    def test_fuzz_SimpleLatentMerger(self, dim_angles: int, dim_s: int, dim_z: int, dim_out: int) -> None:
        run_full_pipeline.SimpleLatentMerger(dim_angles=dim_angles, dim_s=dim_s, dim_z=dim_z, dim_out=dim_out)