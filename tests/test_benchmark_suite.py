import unittest
from unittest.mock import patch
from typing import List, Dict
import torch
import torch.nn as nn

# Hypothesis imports
from hypothesis import given, strategies as st, settings, example, HealthCheck
from hypothesis.strategies import integers, lists, booleans

# Import the module under test (assumes `benchmark.py` is in the same folder).
import rna_predict.benchmarks.benchmark as benchmark

##############################################################################
# Consolidated, Refactored Test Suite for benchmark.py
#
# This test file merges and reorganizes tests originally spread across multiple
# “fuzz” or “basic” test files into a single, coherent unittest suite.
# 
# Key improvements:
#  1. Logical grouping of tests by function or class under test
#  2. Clear docstrings for each test class and test method
#  3. setUp methods to reduce redundancy, create shared resources
#  4. Robust assertions and meaningful coverage
#  5. Hypothesis used in a focused, effective way (instead of random fuzz)
#  6. Simple mock usage where helpful (e.g. torch.cuda.is_available)
##############################################################################


class TestBenchmarkConfig(unittest.TestCase):
    """
    Test the BenchmarkConfig dataclass that holds common benchmarking parameters.
    Verifies that values are correctly stored and defaults behave as expected.
    """

    def test_constructor_defaults(self):
        """Check default values are correctly set when using no parameters."""
        config = benchmark.BenchmarkConfig()
        self.assertEqual(config.N_atom_list, [128, 256, 512])
        self.assertEqual(config.N_token_list, [32, 64, 128])
        self.assertEqual(config.block_size, 16)
        self.assertEqual(config.device, "cuda")
        self.assertEqual(config.num_warmup, 5)
        self.assertEqual(config.num_iters, 10)
        self.assertFalse(config.use_optimized)

    @given(
        N_atom_list=lists(integers(min_value=1, max_value=1024), min_size=1, max_size=5),
        N_token_list=lists(integers(min_value=1, max_value=1024), min_size=1, max_size=5),
        block_size=integers(min_value=1, max_value=32),
        device=st.sampled_from(["cuda", "cpu"]),
        num_warmup=integers(min_value=1, max_value=20),
        num_iters=integers(min_value=1, max_value=50),
        use_optimized=booleans(),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_constructor_hypothesis(self, N_atom_list, N_token_list, block_size,
                                    device, num_warmup, num_iters, use_optimized):
        """
        Hypothesis-based test to ensure the config stores arbitrary valid inputs.
        """
        config = benchmark.BenchmarkConfig(
            N_atom_list=N_atom_list,
            N_token_list=N_token_list,
            block_size=block_size,
            device=device,
            num_warmup=num_warmup,
            num_iters=num_iters,
            use_optimized=use_optimized,
        )
        self.assertEqual(config.N_atom_list, N_atom_list)
        self.assertEqual(config.N_token_list, N_token_list)
        self.assertEqual(config.block_size, block_size)
        self.assertEqual(config.device, device)
        self.assertEqual(config.num_warmup, num_warmup)
        self.assertEqual(config.num_iters, num_iters)
        self.assertEqual(config.use_optimized, use_optimized)


class TestResolveDevice(unittest.TestCase):
    """
    Tests the resolve_device function to ensure it properly switches to 'cpu'
    if CUDA is not available, and leaves 'cpu' as is.
    """

    def test_resolve_device_cpu(self):
        """If device is 'cpu', it remains 'cpu' regardless of CUDA availability."""
        self.assertEqual(benchmark.resolve_device("cpu"), "cpu")

    @patch("torch.cuda.is_available", return_value=False)
    def test_resolve_device_cuda_not_available(self, mock_cuda):
        """
        If CUDA is requested but not available, it should return 'cpu'
        and issue a warning.
        """
        self.assertEqual(benchmark.resolve_device("cuda"), "cpu")

    @patch("torch.cuda.is_available", return_value=True)
    def test_resolve_device_cuda_available(self, mock_cuda):
        """
        If CUDA is requested and available, it should remain 'cuda'.
        """
        self.assertEqual(benchmark.resolve_device("cuda"), "cuda")


class TestCreateEmbedder(unittest.TestCase):
    """
    Tests create_embedder to ensure it returns an instance of nn.Module,
    and the device is correctly set based on availability.
    """

    def setUp(self):
        self.default_args = {
            "device": "cpu",
            "use_optimized": False,
            "c_token": 128,
            "restype_dim": 16,
            "profile_dim": 16,
            "c_atom": 64,
            "c_pair": 32,
            "num_heads": 2,
            "num_layers": 1,
        }

    def test_create_embedder_cpu(self):
        """Test embedder creation on CPU, verifying type and device change if needed."""
        embedder, dev = benchmark.create_embedder(**self.default_args)
        self.assertIsInstance(embedder, nn.Module)
        self.assertEqual(dev, "cpu")

    @patch("torch.cuda.is_available", return_value=False)
    def test_create_embedder_cuda_not_available(self, mock_cuda):
        """
        Even if 'device' is 'cuda', if CUDA is not available, should fall back to 'cpu'.
        """
        args = dict(self.default_args)
        args["device"] = "cuda"
        embedder, dev = benchmark.create_embedder(**args)
        self.assertIsInstance(embedder, nn.Module)
        self.assertEqual(dev, "cpu")

    @patch("torch.cuda.is_available", return_value=True)
    def test_create_embedder_cuda_available(self, mock_cuda):
        """
        If CUDA is available, creating with device='cuda' should keep device='cuda'.
        """
        args = dict(self.default_args)
        args["device"] = "cuda"
        embedder, dev = benchmark.create_embedder(**args)
        self.assertIsInstance(embedder, nn.Module)
        self.assertEqual(dev, "cuda")


class TestGenerateSyntheticFeatures(unittest.TestCase):
    """
    Tests the generate_synthetic_features function, ensuring output dict
    has correct shapes and is placed on the proper device.
    """

    @given(
        st.integers(min_value=1, max_value=256),
        st.integers(min_value=1, max_value=128),
        st.sampled_from(["cpu", "cuda"])
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=5)
    def test_generate_synthetic_features_shape(self, N_atom, N_token, device):
        """
        Uses Hypothesis to generate random but reasonable N_atom, N_token,
        verifying shapes of synthetic features.
        """
        # If CUDA not available, fallback to CPU in test
        actual_device = device
        if device == "cuda" and not torch.cuda.is_available():
            actual_device = "cpu"

        f = benchmark.generate_synthetic_features(N_atom, N_token, actual_device)
        self.assertIn("ref_pos", f)
        self.assertEqual(f["ref_pos"].shape, (N_atom, 3))
        self.assertIn("ref_charge", f)
        self.assertEqual(f["ref_charge"].shape, (N_atom,))
        self.assertIn("ref_element", f)
        self.assertEqual(f["ref_element"].shape, (N_atom, 128))
        self.assertIn("ref_atom_name_chars", f)
        self.assertEqual(f["ref_atom_name_chars"].shape, (N_atom, 16))
        self.assertIn("atom_to_token", f)
        self.assertEqual(f["atom_to_token"].shape, (N_atom,))
        self.assertIn("restype", f)
        self.assertEqual(f["restype"].shape, (N_token, 32))
        self.assertIn("profile", f)
        self.assertEqual(f["profile"].shape, (N_token, 32))
        self.assertIn("deletion_mean", f)
        self.assertEqual(f["deletion_mean"].shape, (N_token,))


class TestWarmupInference(unittest.TestCase):
    """
    Tests the warmup_inference function to ensure it runs multiple passes,
    does not raise errors, and calls the embedder as expected.
    """

    def setUp(self):
        # Minimal embedder creation
        self.embedder, _ = benchmark.create_embedder(device="cpu")
        self.f = benchmark.generate_synthetic_features(16, 4, "cpu")
        self.block_index = torch.randint(0, 16, (16, 4))

    def test_warmup_inference_runs(self):
        """Simple check that warmup_inference runs without error."""
        benchmark.warmup_inference(
            embedder=self.embedder,
            f=self.f,
            block_index=self.block_index,
            device="cpu",
            num_warmup=2
        )
        # If no exceptions, we consider it passed.


class TestMeasureInferenceTimeAndMemory(unittest.TestCase):
    """
    Tests measure_inference_time_and_memory for correct function calls
    and returns a float representing average forward pass time.
    """

    def setUp(self):
        self.embedder, _ = benchmark.create_embedder(device="cpu")
        self.f = benchmark.generate_synthetic_features(8, 2, "cpu")
        self.block_index = torch.randint(0, 8, (8, 2))

    def test_measure_inference_time_and_memory(self):
        """Check it returns a float >= 0."""
        avg_fwd = benchmark.measure_inference_time_and_memory(
            embedder=self.embedder,
            f=self.f,
            block_index=self.block_index,
            device="cpu",
            num_iters=2
        )
        self.assertIsInstance(avg_fwd, float)
        self.assertGreaterEqual(avg_fwd, 0.0)


class TestBenchmarkDecodingLatencyAndMemory(unittest.TestCase):
    """
    Basic check for benchmark_decoding_latency_and_memory to ensure
    it runs multiple iterations without error on small synthetic data.
    """

    def test_benchmark_decoding_latency_and_memory_runs(self):
        """Smoke test to ensure the function completes without throwing exceptions."""
        # Use smaller lists to shorten test time
        benchmark.benchmark_decoding_latency_and_memory(
            N_atom_list=[4, 8],
            N_token_list=[2, 4],
            block_size=2,
            device="cpu",
            num_warmup=1,
            num_iters=2,
        )
        # If it completes, we consider it passed. Checking logs or prints is optional.


class TestBenchmarkInputEmbedding(unittest.TestCase):
    """
    Checks the benchmark_input_embedding function to ensure no errors
    occur with forward/backward pass on small input. 
    """

    def test_benchmark_input_embedding_runs(self):
        """Smoke test: runs the function with small data and doesn't crash."""
        benchmark.benchmark_input_embedding(
            N_atom_list=[4],
            N_token_list=[2],
            block_size=2,
            device="cpu",
            num_warmup=1,
            num_iters=1,
            use_optimized=False,
        )


class TestTimeInputEmbedding(unittest.TestCase):
    """
    Tests time_input_embedding to measure forward/backward times for a small embedder.
    """

    def setUp(self):
        self.embedder, _ = benchmark.create_embedder(device="cpu")
        self.f = benchmark.generate_synthetic_features(8, 2, "cpu")
        self.block_index = torch.randint(0, 8, (8, 2))
        self.criterion = nn.MSELoss()

    def test_time_input_embedding_runs(self):
        """Check it returns a tuple of floats for (avg_fwd, avg_bwd)."""
        avg_fwd, avg_bwd = benchmark.time_input_embedding(
            embedder=self.embedder,
            f=self.f,
            block_index=self.block_index,
            device="cpu",
            num_iters=2,
            criterion=self.criterion,
        )
        self.assertIsInstance(avg_fwd, float)
        self.assertIsInstance(avg_bwd, float)
        self.assertGreaterEqual(avg_fwd, 0.0)
        self.assertGreaterEqual(avg_bwd, 0.0)


class TestWarmupInputEmbedding(unittest.TestCase):
    """
    Tests warmup_input_embedding to ensure forward/backward passes run
    multiple times without errors.
    """

    def setUp(self):
        self.embedder, _ = benchmark.create_embedder(device="cpu")
        self.f = benchmark.generate_synthetic_features(8, 2, "cpu")
        self.block_index = torch.randint(0, 8, (8, 2))
        self.criterion = nn.MSELoss()

    def test_warmup_input_embedding_runs(self):
        """Should not raise exceptions during multiple forward/backward passes."""
        benchmark.warmup_input_embedding(
            embedder=self.embedder,
            f=self.f,
            block_index=self.block_index,
            device="cpu",
            num_warmup=2,
            criterion=self.criterion,
        )
        # No exception => pass.


class TestTimedDecoding(unittest.TestCase):
    """
    Tests the timed_decoding function which calls measure_inference_time_and_memory 
    under the hood for a number of iterations.
    """

    def setUp(self):
        self.embedder, _ = benchmark.create_embedder(device="cpu")
        self.f = benchmark.generate_synthetic_features(8, 2, "cpu")
        self.block_index = torch.randint(0, 8, (8, 2))

    def test_timed_decoding_runs(self):
        """Check that timed_decoding returns a float and doesn't error out."""
        result = benchmark.timed_decoding(
            embedder=self.embedder,
            f=self.f,
            block_index=self.block_index,
            device="cpu",
            iters=2
        )
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)


if __name__ == "__main__":
    unittest.main()