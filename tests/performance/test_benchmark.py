import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from rna_predict.benchmarks.benchmark import (
    BenchmarkConfig,
    benchmark_decoding_latency_and_memory,
    benchmark_input_embedding,
    create_embedder,
    generate_synthetic_features,
    measure_inference_time_and_memory,
    resolve_device,
    time_input_embedding,
    timed_decoding,
    timed_embedding,
    warmup_decoding,
    warmup_embedding,
    warmup_inference,
    warmup_input_embedding,
)
from rna_predict.pipeline.stageA.input_embedding.current.embedders import (
    InputFeatureEmbedder,
)


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
            num_iters=1,
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
            use_optimized=True,
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
        self.assertEqual(features["ref_pos"].shape, (1, N_atom, 3))
        self.assertEqual(features["ref_charge"].shape, (1, N_atom, 1))
        self.assertEqual(features["ref_element"].shape, (1, N_atom, 128))
        self.assertEqual(features["ref_atom_name_chars"].shape, (1, N_atom, 256))
        self.assertEqual(features["atom_to_token"].shape, (1, N_atom))
        self.assertEqual(features["restype"].shape, (1, N_token, 32))
        self.assertEqual(features["profile"].shape, (1, N_token, 32))
        self.assertEqual(features["deletion_mean"].shape, (1, N_token))

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
        warmup_embedding(
            embedder, f, block_index, device, num_warmup=1, criterion=criterion
        )
        # Check no errors

    def test_timed_embedding(self):
        embedder = InputFeatureEmbedder()
        device = "cpu"
        embedder.to(device)
        f = generate_synthetic_features(2, 1, device)
        block_index = torch.randint(0, 2, (2, 1), device=device)
        criterion = nn.MSELoss()
        avg_fwd, avg_bwd = timed_embedding(
            embedder, f, block_index, device, num_iters=1, criterion=criterion
        )
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
            num_iters=1,
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
            use_optimized=False,
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
            use_optimized=True,
        )


class TestBenchmark(unittest.TestCase):
    def setUp(self):
        # Set up common test parameters
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.N_atom = 128
        self.N_token = 32
        self.block_size = 16
        self.num_warmup = 2
        self.num_iters = 2

    def test_resolve_device(self):
        # Test CUDA available case
        with patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(resolve_device("cuda"), "cuda")

        # Test CUDA unavailable case
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_device("cuda"), "cpu")

        # Test CPU case
        self.assertEqual(resolve_device("cpu"), "cpu")

    def test_benchmark_config(self):
        config = BenchmarkConfig()
        self.assertEqual(config.N_atom_list, [128, 256, 512])
        self.assertEqual(config.N_token_list, [32, 64, 128])
        self.assertEqual(config.block_size, 16)
        self.assertEqual(config.device, "cuda")
        self.assertEqual(config.num_warmup, 5)
        self.assertEqual(config.num_iters, 10)
        self.assertEqual(config.use_optimized, False)

    def test_create_embedder(self):
        embedder, actual_device = create_embedder(
            self.device,
            use_optimized=False,
            c_token=384,
            restype_dim=32,
            profile_dim=32,
            c_atom=128,
            c_pair=32,
            num_heads=4,
            num_layers=3,
        )

        self.assertIsNotNone(embedder)
        self.assertEqual(actual_device, self.device)
        self.assertEqual(embedder.c_token, 384)
        self.assertEqual(embedder.restype_dim, 32)
        self.assertEqual(embedder.profile_dim, 32)
        self.assertEqual(embedder.c_atom, 128)
        self.assertEqual(embedder.c_pair, 32)
        self.assertEqual(embedder.num_heads, 4)
        self.assertEqual(embedder.num_layers, 3)

    def test_generate_synthetic_features(self):
        features = generate_synthetic_features(self.N_atom, self.N_token, self.device)

        # Check all required features are present
        required_features = [
            "ref_pos",
            "ref_charge",
            "ref_element",
            "ref_atom_name_chars",
            "atom_to_token",
            "atom_to_token_idx",
            "ref_space_uid",
            "ref_mask",
            "restype",
            "profile",
            "deletion_mean",
        ]
        for feature in required_features:
            self.assertIn(feature, features)

        # Check shapes
        self.assertEqual(features["ref_pos"].shape, (1, self.N_atom, 3))
        self.assertEqual(features["ref_charge"].shape, (1, self.N_atom, 1))
        self.assertEqual(features["ref_element"].shape, (1, self.N_atom, 128))
        self.assertEqual(features["ref_atom_name_chars"].shape, (1, self.N_atom, 256))
        self.assertEqual(features["atom_to_token"].shape, (1, self.N_atom))
        self.assertEqual(features["restype"].shape, (1, self.N_token, 32))
        self.assertEqual(features["profile"].shape, (1, self.N_token, 32))
        self.assertEqual(features["deletion_mean"].shape, (1, self.N_token))

    def test_warmup_inference(self):
        embedder, device = create_embedder(self.device)
        features = generate_synthetic_features(self.N_atom, self.N_token, device)
        block_index = torch.randint(
            0, self.N_atom, (self.N_atom, self.block_size), device=device
        )

        # Test with CPU
        warmup_inference(embedder, features, block_index, "cpu", self.num_warmup)

        # Test with CUDA if available
        if torch.cuda.is_available():
            with patch("torch.cuda.synchronize") as mock_sync:
                warmup_inference(
                    embedder, features, block_index, "cuda", self.num_warmup
                )
                self.assertEqual(mock_sync.call_count, self.num_warmup)

    def test_measure_inference_time_and_memory(self):
        embedder, device = create_embedder(self.device)
        features = generate_synthetic_features(self.N_atom, self.N_token, device)
        block_index = torch.randint(
            0, self.N_atom, (self.N_atom, self.block_size), device=device
        )

        # Test with CPU
        avg_time_cpu = measure_inference_time_and_memory(
            embedder, features, block_index, "cpu", self.num_iters
        )
        self.assertIsInstance(avg_time_cpu, float)
        self.assertGreater(avg_time_cpu, 0)

        # Test with CUDA if available
        if torch.cuda.is_available():
            with (
                patch("torch.cuda.synchronize") as mock_sync,
                patch("torch.cuda.reset_peak_memory_stats") as mock_reset,
                patch(
                    "torch.cuda.max_memory_allocated", return_value=1024 * 1024
                ) as mock_max_mem,
            ):
                avg_time_cuda = measure_inference_time_and_memory(
                    embedder, features, block_index, "cuda", self.num_iters
                )
                self.assertIsInstance(avg_time_cuda, float)
                self.assertGreater(avg_time_cuda, 0)
                self.assertEqual(mock_sync.call_count, self.num_iters)
                self.assertEqual(mock_reset.call_count, self.num_iters)
                self.assertEqual(mock_max_mem.call_count, self.num_iters)

    def test_warmup_input_embedding(self):
        embedder, device = create_embedder(self.device)
        features = generate_synthetic_features(self.N_atom, self.N_token, device)
        block_index = torch.randint(
            0, self.N_atom, (self.N_atom, self.block_size), device=device
        )
        criterion = torch.nn.MSELoss().to(device)

        # Test with CPU
        warmup_input_embedding(
            embedder, features, block_index, "cpu", self.num_warmup, criterion
        )

        # Test with CUDA if available
        if torch.cuda.is_available():
            with patch("torch.cuda.synchronize") as mock_sync:
                warmup_input_embedding(
                    embedder, features, block_index, "cuda", self.num_warmup, criterion
                )
                self.assertEqual(mock_sync.call_count, self.num_warmup)

    def test_time_input_embedding(self):
        embedder, device = create_embedder(self.device)
        features = generate_synthetic_features(self.N_atom, self.N_token, device)
        block_index = torch.randint(
            0, self.N_atom, (self.N_atom, self.block_size), device=device
        )
        criterion = torch.nn.MSELoss().to(device)

        # Test with CPU
        fwd_time_cpu, bwd_time_cpu = time_input_embedding(
            embedder, features, block_index, "cpu", self.num_iters, criterion
        )
        self.assertIsInstance(fwd_time_cpu, float)
        self.assertIsInstance(bwd_time_cpu, float)
        self.assertGreater(fwd_time_cpu, 0)
        self.assertGreater(bwd_time_cpu, 0)

        # Test with CUDA if available
        if torch.cuda.is_available():
            with patch("torch.cuda.synchronize") as mock_sync:
                fwd_time_cuda, bwd_time_cuda = time_input_embedding(
                    embedder, features, block_index, "cuda", self.num_iters, criterion
                )
                self.assertIsInstance(fwd_time_cuda, float)
                self.assertIsInstance(bwd_time_cuda, float)
                self.assertGreater(fwd_time_cuda, 0)
                self.assertGreater(bwd_time_cuda, 0)
                # Should be called twice per iteration (once for forward, once for backward)
                self.assertEqual(mock_sync.call_count, 2 * self.num_iters)

    def test_benchmark_decoding_latency_and_memory_small(self):
        # Test with small parameters to keep test time reasonable
        benchmark_decoding_latency_and_memory(
            N_atom_list=[128],
            N_token_list=[32],
            block_size=16,
            device=self.device,
            num_warmup=1,
            num_iters=1,
        )

    def test_benchmark_input_embedding_small(self):
        # Test with small parameters to keep test time reasonable
        benchmark_input_embedding(
            N_atom_list=[128],
            N_token_list=[32],
            block_size=16,
            device=self.device,
            num_warmup=1,
            num_iters=1,
            use_optimized=False,
        )

    def test_benchmark_input_embedding_optimized_small(self):
        # Test with small parameters to keep test time reasonable
        benchmark_input_embedding(
            N_atom_list=[128],
            N_token_list=[32],
            block_size=16,
            device=self.device,
            num_warmup=1,
            num_iters=1,
            use_optimized=True,
        )

    def test_main_execution(self):
        # Test the main execution path
        with (
            patch("sys.argv", ["benchmark.py"]),
            patch(
                "rna_predict.benchmarks.benchmark.benchmark_input_embedding"
            ) as mock_benchmark_input,
            patch(
                "rna_predict.benchmarks.benchmark.benchmark_decoding_latency_and_memory"
            ) as mock_benchmark_decoding,
        ):
            if __name__ == "__main__":
                import rna_predict.benchmarks.benchmark

                rna_predict.benchmarks.benchmark.main()

                # Verify both benchmark functions were called
                mock_benchmark_input.assert_any_call(use_optimized=False)
                mock_benchmark_input.assert_any_call(use_optimized=True)
                mock_benchmark_decoding.assert_called_once()


if __name__ == "__main__":
    unittest.main()
