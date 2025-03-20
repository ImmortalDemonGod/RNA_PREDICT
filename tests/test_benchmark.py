import unittest
import torch
import time
from unittest.mock import patch

from rna_predict.benchmarks.benchmark import (
    BenchmarkConfig,
    resolve_device,
    generate_synthetic_features,
    warmup_decoding,
    timed_decoding,
    warmup_embedding,
    timed_embedding,
    benchmark_decoding_latency_and_memory,
    benchmark_input_embedding,
)
from rna_predict.models.encoder.input_feature_embedding import InputFeatureEmbedder
import torch.nn as nn

class TestBenchmarkConfigs(unittest.TestCase):
    def test_decoding_benchmark_config_defaults(self):
        config = BenchmarkConfig()
        self.assertEqual(config.N_atom_list, [128, 256, 512])
        self.assertEqual(config.N_token_list, [32, 64, 128])
        self.assertEqual(config.block_size, 16)
        self.assertEqual(config.device, "cuda")
        self.assertEqual(config.num_warmup, 5)
        self.assertEqual(config.num_iters, 10)

    def test_embedding_benchmark_config_defaults(self):
        config = BenchmarkConfig()
        self.assertEqual(config.N_atom_list, [128, 256, 512])
        self.assertEqual(config.N_token_list, [32, 64, 128])
        self.assertEqual(config.block_size, 16)
        self.assertEqual(config.device, "cuda")
        self.assertEqual(config.num_warmup, 5)
        self.assertEqual(config.num_iters, 10)
        self.assertFalse(config.use_optimized)

    def test_decoding_benchmark_config_custom(self):
        config = BenchmarkConfig(
            N_atom_list=[1],
            N_token_list=[2],
            block_size=3,
            device="cpu",
            num_warmup=0,
            num_iters=1
        )
        self.assertEqual(config.N_atom_list, [1])
        self.assertEqual(config.N_token_list, [2])
        self.assertEqual(config.block_size, 3)
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.num_warmup, 0)
        self.assertEqual(config.num_iters, 1)

    def test_embedding_benchmark_config_custom(self):
        config = BenchmarkConfig(
            N_atom_list=[10, 20],
            N_token_list=[5],
            block_size=4,
            device="cpu",
            num_warmup=2,
            num_iters=2,
            use_optimized=True
        )
        self.assertEqual(config.N_atom_list, [10, 20])
        self.assertEqual(config.N_token_list, [5])
        self.assertEqual(config.block_size, 4)
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.num_warmup, 2)
        self.assertEqual(config.num_iters, 2)
        self.assertTrue(config.use_optimized)

class TestBenchmarkHelpers(unittest.TestCase):
    def test_resolve_device_cpu(self):
        self.assertEqual(resolve_device("cpu"), "cpu")

    @patch("torch.cuda.is_available", return_value=False)
    def test_resolve_device_cuda_unavailable(self, mock_is_avail):
        self.assertEqual(resolve_device("cuda"), "cpu")

    def test_generate_synthetic_features_shapes(self):
        N_atom, N_token = 4, 3
        dev = "cpu"
        features = generate_synthetic_features(N_atom, N_token, dev)
        self.assertIn("ref_pos", features)
        self.assertEqual(features["ref_pos"].shape, (N_atom, 3))
        self.assertEqual(features["ref_charge"].shape, (N_atom,))
        self.assertEqual(features["ref_element"].shape, (N_atom, 128))
        self.assertEqual(features["ref_atom_name_chars"].shape, (N_atom, 16))
        self.assertEqual(features["atom_to_token"].shape, (N_atom,))
        self.assertEqual(features["restype"].shape, (N_token, 32))
        self.assertEqual(features["profile"].shape, (N_token, 32))
        self.assertEqual(features["deletion_mean"].shape, (N_token,))

    def test_warmup_decoding(self):
        embedder = InputFeatureEmbedder()
        device = "cpu"
        embedder.to(device)
        f = generate_synthetic_features(2, 1, device)
        block_index = torch.randint(0, 2, (2, 1), device=device)
        warmup_decoding(embedder, f, block_index, device, num_warmup=1)
        # If no error, we pass.

    def test_timed_decoding(self):
        embedder = InputFeatureEmbedder()
        device = "cpu"
        embedder.to(device)
        f = generate_synthetic_features(2, 1, device)
        block_index = torch.randint(0, 2, (2, 1), device=device)
        avg_time = timed_decoding(embedder, f, block_index, device, iters=1)
        self.assertIsInstance(avg_time, float)

    def test_warmup_embedding(self):
        embedder = InputFeatureEmbedder()
        device = "cpu"
        embedder.to(device)
        f = generate_synthetic_features(2, 1, device)
        block_index = torch.randint(0, 2, (2, 1), device=device)
        criterion = nn.MSELoss()
        warmup_embedding(embedder, f, block_index, device, num_warmup=1, criterion=criterion)
        # Check no errors

    def test_timed_embedding(self):
        embedder = InputFeatureEmbedder()
        device = "cpu"
        embedder.to(device)
        f = generate_synthetic_features(2, 1, device)
        block_index = torch.randint(0, 2, (2, 1), device=device)
        criterion = nn.MSELoss()
        avg_fwd, avg_bwd = timed_embedding(embedder, f, block_index, device, iters=1, criterion=criterion)
        self.assertIsInstance(avg_fwd, float)
        self.assertIsInstance(avg_bwd, float)

class TestBenchmarkEntryPoints(unittest.TestCase):
    def test_benchmark_decoding_latency_and_memory_small(self):
        """
        Run with small sets to ensure it doesn't blow up runtime.
        Check no exceptions raised.
        """
        benchmark_decoding_latency_and_memory(
            N_atom_list=[2],
            N_token_list=[2],
            block_size=1,
            device="cpu",
            num_warmup=1,
            num_iters=1
        )
        # If we reach here without error, it's a pass.

    def test_benchmark_input_embedding_small(self):
        """
        Similarly run with small sets, 1 iteration.
        """
        benchmark_input_embedding(
            N_atom_list=[2],
            N_token_list=[2],
            block_size=1,
            device="cpu",
            num_warmup=1,
            num_iters=1,
            use_optimized=False
        )
        # If no error, we pass.

    def test_benchmark_input_embedding_optimized_small(self):
        """
        Test use_optimized=True path with small sets, 1 iteration.
        """
        benchmark_input_embedding(
            N_atom_list=[2],
            N_token_list=[2],
            block_size=1,
            device="cpu",
            num_warmup=1,
            num_iters=1,
            use_optimized=True
        )

if __name__ == "__main__":
    unittest.main()