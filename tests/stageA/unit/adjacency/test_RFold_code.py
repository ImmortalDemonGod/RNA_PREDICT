# test_refactored_RFold_code.py
"""
Consolidated and refactored unittest suite for the rna_predict.pipeline.stageA.RFold_code module.

This suite:
1. Organizes related tests into logical classes.
2. Uses docstrings to describe each test's purpose.
3. Employs setUp methods for shared initialization.
4. Combines or removes redundant tests, focusing on meaningful coverage.
5. Uses Hypothesis for property-based testing where beneficial.
6. Demonstrates robust assertions and error-case testing.
7. Is intended to be run with: python -m unittest test_refactored_RFold_code.py
"""

import os
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# Adjust this import according to your project layout, e.g.:
# from rna_predict.pipeline.stageA import RFold_code as RC
import rna_predict.pipeline.stageA.adjacency.RFold_code as RC


class TestRFoldUtilities(unittest.TestCase):
    """
    Tests utility functions such as set_seed, print_log, output_namespace, check_dir, and cuda.
    """

    def setUp(self):
        """Common setup for utility tests."""
        self.temp_dir_name = "temp_test_dir"

    def tearDown(self):
        """Clean up any temp directories created during testing."""
        if os.path.exists(self.temp_dir_name):
            import shutil

            shutil.rmtree(self.temp_dir_name, ignore_errors=True)

    @given(seed=st.integers(min_value=0, max_value=2**32 - 1))
    @settings(
        deadline=None
    )  # Disable deadline since this test can be slow on first run
    def test_set_seed(self, seed: int):
        """
        Ensure set_seed initializes random seeds consistently.
        We'll generate some random numbers after setting the seed
        to confirm they match across repeated calls.
        """
        RC.set_seed(seed)
        first_rand = random.random()
        # Re-seed and compare
        RC.set_seed(seed)
        second_rand = random.random()
        self.assertAlmostEqual(first_rand, second_rand, places=7)

    def test_print_log(self):
        """
        Check that print_log prints and logs the given message.
        We'll patch the module's logger to confirm it's called.
        """
        import rna_predict.pipeline.stageA.adjacency.RFold_code as RC_mod
        with patch.object(RC_mod.logger, "info") as mock_info:
            RC.print_log("Hello World", debug_logging=True)
            mock_info.assert_called_once_with("Hello World")

        # Also check that logger.info is NOT called if debug_logging is False
        with patch.object(RC_mod.logger, "info") as mock_info:
            RC.print_log("Hello World", debug_logging=False)
            mock_info.assert_not_called()

    def test_output_namespace(self):
        """
        Verify output_namespace returns the expected string from a namespace object.
        """

        class NamespaceObj:
            def __init__(self):
                self.alpha = 1
                self.beta = "test"

        ns = NamespaceObj()
        out_str = RC.output_namespace(ns)
        self.assertIn("alpha: \t1", out_str)
        self.assertIn("beta: \ttest", out_str)

    def test_check_dir_creates_directory(self):
        """
        Confirm check_dir creates a directory if it doesn't exist.
        """
        self.assertFalse(os.path.exists(self.temp_dir_name))
        RC.check_dir(self.temp_dir_name)
        self.assertTrue(os.path.exists(self.temp_dir_name))

    def test_cuda_raises_on_unsupported_type(self):
        """
        Passing an unsupported Python type to cuda should raise a TypeError.
        """
        with self.assertRaises(TypeError):
            RC.cuda({"a_set": {1, 2, 3}})  # sets are not handled by RC.cuda


class TestMatrixOps(unittest.TestCase):
    """
    Tests base_matrix, constraint_matrix, row_col_softmax, row_col_argmax,
    and sequence2onehot functionalities.
    """

    def setUp(self):
        # Use CPU for test simplicity
        self.device = torch.device("cpu")

    @given(n=st.integers(min_value=1, max_value=64))
    @settings(deadline=None)  # Disable deadline for this flaky test
    def test_base_matrix(self, n: int):
        """
        base_matrix(n, device) should produce an n x n matrix with 1s except
        in a small diagonal band of 0s. We test random n in [1..64].
        """
        mat = RC.base_matrix(n, self.device)
        self.assertEqual(mat.shape, (n, n))
        for i in range(n):
            # Check the "diagonal band" of zeros within +/- 3 positions
            for j in range(max(0, i - 3), min(n, i + 4)):
                self.assertEqual(mat[i, j].item(), 0.0)
            # Values outside that band should be 1.0
            if i - 4 >= 0:
                self.assertEqual(mat[i, i - 4].item(), 1.0)
            if i + 4 < n:
                self.assertEqual(mat[i, i + 4].item(), 1.0)

    def test_constraint_matrix(self):
        """
        constraint_matrix(x) returns base-pair constraints for AU, CG, UG.
        We'll feed a small one-hot batch of shape [B=1, L=4, 4].
        """
        # Single batch, length=4 => one-hot for "AUCG"
        one_hot = torch.tensor(
            [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]],
            dtype=torch.float32,
            device=self.device,
        )
        cm = RC.constraint_matrix(one_hot)
        self.assertEqual(cm.shape, (1, 4, 4))
        # Constraint for A<->U at positions (0,1) & (1,0)
        self.assertEqual(cm[0, 0, 1].item(), 1.0)
        self.assertEqual(cm[0, 1, 0].item(), 1.0)
        # Constraint for C<->G at positions (2,3) & (3,2)
        self.assertEqual(cm[0, 2, 3].item(), 1.0)
        self.assertEqual(cm[0, 3, 2].item(), 1.0)

    def test_row_col_softmax_and_argmax(self):
        """
        row_col_softmax should average rowwise and columnwise softmax.
        row_col_argmax uses the result to produce a discrete matrix of 1-hot positions.
        """
        mat = torch.tensor(
            [[[0.1, 0.2, 0.3], [0.5, 0.4, 0.2], [0.0, 0.1, 0.9]]], dtype=torch.float32
        )
        # shape => [B=1, L=3, L=3]
        softmaxed = RC.row_col_softmax(mat)
        self.assertEqual(softmaxed.shape, (1, 3, 3))

        one_hot = RC.row_col_argmax(mat)
        self.assertEqual(one_hot.shape, (1, 3, 3))
        # For quick check: each row/col can have only one "1" where it is max
        row_sums = one_hot.sum(dim=-1)
        col_sums = one_hot.sum(dim=-2)
        self.assertTrue(torch.all(row_sums <= 1))
        self.assertTrue(torch.all(col_sums <= 1))

    def test_sequence2onehot(self):
        """
        sequence2onehot(seq, device) transforms e.g. "AUGC" -> [ [1,0,0,0], [0,1,0,0], ... ].
        """
        seq = "AUGC"
        oh = RC.sequence2onehot(seq, self.device)
        # oh shape => [1, 4], with numeric labels
        # But we want to confirm each index matches the known dict:
        # "A": 0, "U": 1, "C": 2, "G": 3
        # Actually, the code maps "A"->0, "U"->1, "C"->2, "G"->3
        # So oh => [ [0,1,3,2] ] if we read them directly, but let's just check length and contents.
        self.assertEqual(oh.shape, (1, 4))
        self.assertListEqual(oh[0].tolist(), [0, 1, 3, 2])


class TestNNModules(unittest.TestCase):
    """
    Tests for conv_block, up_conv, OffsetScale, and Attn modules,
    plus brief checks on Encoder, Decoder, and Seq2Map.
    """

    def test_conv_block_basic(self):
        """
        conv_block(ch_in, ch_out, residual=False) is a small 2-layer CNN block.
        We'll run a forward pass with random input and check shape.
        """
        block = RC.conv_block(ch_in=8, ch_out=8, residual=False)
        x = torch.randn(2, 8, 16, 16)  # batch=2, channels=8, H=16, W=16
        out = block(x)
        self.assertEqual(out.shape, (2, 8, 16, 16))

    def test_conv_block_residual(self):
        """
        If residual=True, conv_block should skip-add the original input.
        We'll confirm shape and that output differs from non-residual pass.
        """
        block = RC.conv_block(ch_in=8, ch_out=8, residual=True)
        x = torch.randn(2, 8, 16, 16)
        out = block(x)
        self.assertEqual(out.shape, (2, 8, 16, 16))
        # Check a subset of elements to see if skip-add likely occurred:
        self.assertFalse(torch.allclose(out, block.conv(x)))

    def test_up_conv_basic(self):
        """
        up_conv(ch_in, ch_out) performs 2x upsample + convolution.
        We'll confirm shape changes accordingly.
        """
        up = RC.up_conv(ch_in=8, ch_out=4)
        x = torch.randn(2, 8, 8, 8)
        out = up(x)
        # shape => batch=2, channels=4, 16x16
        self.assertEqual(out.shape, (2, 4, 16, 16))

    def test_offset_scale(self):
        """
        OffsetScale(dim, heads=1) modifies each element with learned scale & offset.
        We'll pass random input and confirm shape remains the same except for the heads dimension.
        """
        dim = 8
        module = RC.OffsetScale(dim=dim, heads=2)
        x = torch.randn(3, 10, dim)  # shape => [B=3, L=10, D=8]
        # forward returns 2 separate Tensors (unbound in last dimension)
        out_q, out_k = module(x)
        self.assertEqual(out_q.shape, (3, 10, dim))
        self.assertEqual(out_k.shape, (3, 10, dim))

    @settings(deadline=1000, suppress_health_check=[HealthCheck.too_slow])
    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        seq_len=st.integers(min_value=1, max_value=16),
        dim=st.integers(min_value=4, max_value=32),
    )
    def test_attn_with_hypothesis(self, batch_size: int, seq_len: int, dim: int):
        """
        Attn(...) takes a [B, L, D] input and returns [B, L, L] attention map.
        The output is non-negative (due to ReLU) and squared, but not normalized to sum to 1.
        """
        # Set seeds for all random operations
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        # Create input tensor with fixed values for deterministic testing
        x = torch.randn(batch_size, seq_len, dim)

        attn_module = RC.Attn(
            dim=dim, query_key_dim=dim, expansion_factor=2.0, dropout=0.0
        )

        # Run forward pass
        attn_output = attn_module(x)

        # Verify output properties
        self.assertEqual(attn_output.shape, (batch_size, seq_len, seq_len))
        self.assertTrue(torch.all(attn_output >= 0))  # Non-negative due to ReLU
        self.assertTrue(torch.all(attn_output <= 1))  # Squared values should be <= 1

    def test_encoder_decoder_seq2map_smoke(self):
        """
        Smoke test for Encoder, Decoder, and Seq2Map in a minimal forward pass.
        Confirm shapes are consistent and no exceptions raised.
        """
        encoder = RC.Encoder(C_lst=[4, 8, 16])
        decoder = RC.Decoder(C_lst=[32, 16, 8])
        seqmap = RC.Seq2Map(
            input_dim=4, num_hidden=8, dropout=0.0, device=torch.device("cpu")
        )

        # 1) seqmap => produce a [batch, seq_len, seq_len] attention
        src = torch.randint(0, 4, (2, 10))  # batch=2, seq_len=10
        attention = seqmap(src)
        self.assertEqual(attention.shape, (2, 10, 10))

        # 2) pass into encoder => expects [B, C, H, W], so unsqueeze
        x = (attention * torch.sigmoid(attention)).unsqueeze(
            1
        )  # shape => [2, 1, 10, 10]
        latent, skips = encoder(x)
        # 3) decoder => reversed skips
        out = decoder(latent, skips)
        self.assertIsInstance(out, torch.Tensor)


class TestRFoldModel(unittest.TestCase):
    """
    Tests for the RFoldModel class, ensuring forward pass shape correctness and
    verifying minimal usage with 'args' namespace.
    """

    def setUp(self):
        class MockArgs:
            num_hidden = 16
            dropout = 0.0
            use_gpu = False
            device = 'cpu'  # Added to fix ValueError in RFoldModel

        self.args = MockArgs()
        self.model = RC.RFoldModel(self.args)
        self.model.eval()

    def test_rfold_model_forward_basic(self):
        """
        A minimal forward pass with a small batch of sequences. Verifies shape correctness.
        """
        seqs = torch.randint(0, 4, (2, 12))  # batch=2, length=12
        out = self.model(seqs)
        # out shape => [2, 12, 12] from final adjacency-likematrix
        self.assertEqual(out.shape, (2, 12, 12))


class TestIOFunctions(unittest.TestCase):
    """
    Tests for IO-like functionality: ct_file_output, seq2dot, save_ct.
    We'll use temporary directories and check expected outputs.
    """

    def setUp(self):
        self.temp_dir = "test_io_temp"
        RC.check_dir(self.temp_dir)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_seq2dot(self):
        """
        For a numeric array seq, seq2dot interprets positions where seq[i] < i => ')',
        seq[i] > i => '(', else '.'.
        We'll create a small numeric seq and check the resulting string.
        """
        seq = np.array([2, 0, 3, 0])  # shape=4
        # index => [1,2,3,4]
        # seq[0]=2 => 2>1 => '('
        # seq[1]=0 => 0<2 => ')'
        # seq[2]=3 => 3>3 => false => '.'? Actually 3>3 is false => '.'
        # seq[3]=0 => 0<4 => ')'
        dot_str = RC.seq2dot(seq)
        self.assertEqual(dot_str, "(.))")

    def test_ct_file_output_and_save_ct(self):
        """
        Round-trip test:
         1) Manually create a small adjacency or pairing
         2) Use ct_file_output
         3) Check that the .ct file was created
        We'll also test save_ct with a minimal mock predict_matrix and confirm no exception.
        """
        # 1) ct_file_output
        pairs = [(1, 3), (3, 1)]  # positions 1<->3
        seq = "AGC"
        RC.ct_file_output(pairs, seq, "test_seq", self.temp_dir)
        ct_path = os.path.join(self.temp_dir, "test_seq.ct")
        self.assertTrue(os.path.exists(ct_path), "CT file not created.")

        # 2) save_ct test
        predict_matrix = torch.tensor(
            [[[0.1, 0.8, 0.1], [0.6, 0.1, 0.3], [0.2, 0.0, 0.8]]]
        )
        # shape => [1,3,3]
        seq_ori = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
        RC.save_ct(predict_matrix, seq_ori, "predict_test")


class TestMiscFunctions(unittest.TestCase):
    """
    Tests for get_cut_len, process_seqs, and visual_get_bases.
    """

    def test_get_cut_len(self):
        """
        get_cut_len should round length up to the nearest multiple of 16.
        """
        self.assertEqual(RC.get_cut_len(1), 16)
        self.assertEqual(RC.get_cut_len(15), 16)
        self.assertEqual(RC.get_cut_len(16), 16)
        self.assertEqual(RC.get_cut_len(17), 32)

    def test_process_seqs(self):
        """
        process_seqs(seq, device) -> (nseq, nseq_one_hot, seq_len).
        We'll confirm shapes match expectations and no error is raised.
        """
        seq = "AUCG"
        nseq, nseq_one_hot, length = RC.process_seqs(seq, torch.device("cpu"))
        self.assertEqual(length, 4)
        self.assertEqual(nseq.shape, (1, 16))  # nearest multiple of 16 is 16
        self.assertEqual(nseq_one_hot.shape, (1, 16, 4))

    def test_visual_get_bases(self):
        """
        Return a tuple of strings listing indexes of A, U, C, G bases.
        """
        a_bases, u_bases, c_bases, g_bases = RC.visual_get_bases("AUGCAUGG")
        # Indices start from 1
        self.assertEqual(a_bases, "1,5")
        self.assertEqual(u_bases, "2,6")
        self.assertEqual(c_bases, "3")
        self.assertEqual(g_bases, "4,7,8")


if __name__ == "__main__":
    unittest.main()
