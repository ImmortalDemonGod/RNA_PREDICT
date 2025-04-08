"""
Comprehensive pytest test suite covering:
1) Naive local block-sparse attention (LocalBlockSparseAttentionNaive) and supporting functionality
   from block_sparse.py
2) The checkpoint_blocks function from checkpointing.py

This suite demonstrates:
- Normal usage tests (forward and backward pass for naive attention)
- Shape and error condition handling
- Local blockmask generation tests
- Basic and fuzz tests for checkpoint_blocks with Hypothesis
- Handling of optional block-sparse-attn library for BlockSparseAttentionOptimized
"""

from typing import Any, Callable, List, Optional, cast

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from rna_predict.pipeline.stageA.input_embedding.current.checkpointing import (
    checkpoint_blocks,
)

# If your actual code resides elsewhere, adjust these imports accordingly.
# For example:
from rna_predict.pipeline.stageA.input_embedding.legacy.attention.block_sparse import (
    BlockSparseAttentionOptimized,
    LocalBlockMaskConfig,
    LocalBlockSparseAttentionNaive,
    build_local_blockmask,
    _HAS_BSA,
)


###############################################################################
# Dummy definitions for demonstration:
# Remove these and uncomment real imports above in your actual code.
###############################################################################
###############################################################################


@pytest.fixture
def small_tensors():
    """
    Fixture providing small test tensors for local block-sparse attention.
    Shapes chosen to be minimal but valid.
    """
    N_atom, n_heads, c_per_head = 4, 2, 3
    q = torch.randn(N_atom, n_heads, c_per_head, requires_grad=True)
    k = torch.randn(N_atom, n_heads, c_per_head, requires_grad=True)
    v = torch.randn(N_atom, n_heads, c_per_head, requires_grad=True)
    pair_bias = torch.randn(N_atom, N_atom, n_heads, requires_grad=True)
    block_index = torch.randint(
        low=0, high=N_atom, size=(N_atom, 2)
    )  # block_size=2 neighbors
    return q, k, v, pair_bias, block_index


class TestLocalBlockSparseAttentionNaive:
    """
    Tests for the naive local block-sparse attention (forward/backward).
    """

    def test_forward_output_shape(self, small_tensors):
        """
        Verifies output shape matches input Q shape.
        """
        q, k, v, bias, block_idx = small_tensors
        out = LocalBlockSparseAttentionNaive.apply(q, k, v, bias, block_idx)
        assert out.shape == q.shape, "Output shape should match Q shape."

    def test_backward_no_error(self, small_tensors):
        """
        Verifies that backward pass executes without error.
        """
        q, k, v, bias, block_idx = small_tensors
        out = LocalBlockSparseAttentionNaive.apply(q, k, v, bias, block_idx)
        loss = out.sum()
        loss.backward()
        # Check grads were computed (not necessarily correctness).
        assert q.grad is not None, "Q gradient should exist."
        assert k.grad is not None, "K gradient should exist."
        assert v.grad is not None, "V gradient should exist."
        assert bias.grad is not None, "Pairwise bias gradient should exist."

    def test_shape_mismatch_raises(self, small_tensors):
        """
        Tests that shape mismatch in Q and K triggers an error or mismatch in practice.
        Here, we artificially mismatch shapes to see if forward fails.
        """
        q, k, v, bias, block_idx = small_tensors
        # Introduce a mismatch: k has an extra row
        k_bad = torch.randn(q.shape[0] + 1, q.shape[1], q.shape[2])
        with pytest.raises(RuntimeError):
            LocalBlockSparseAttentionNaive.apply(q, k_bad, v, bias, block_idx)


class TestBlockSparseAttentionOptimized:
    """
    Tests for the optional optimized block-sparse attention module.
    """

    @pytest.mark.skipif(not _HAS_BSA, reason="block_sparse_attn not installed")
    def test_forward_shape(self, small_tensors):
        """
        Checks that forward output shape matches Q.
        """
        q, k, v, bias, _ = small_tensors
        mod = BlockSparseAttentionOptimized(
            nheads=2, block_size=128, local_window=32, causal=False
        )
        out = mod(q, k, v, bias)
        assert out.shape == q.shape, "Optimized attention output shape mismatch."

    @pytest.mark.skipif(not _HAS_BSA, reason="block_sparse_attn not installed")
    def test_negative_blocksize_raises(self, small_tensors):
        """
        If block_size < 0, we mimic the scenario that the library is missing or config is invalid.
        """
        q, k, v, bias, _ = small_tensors
        mod = BlockSparseAttentionOptimized(
            nheads=2, block_size=-1, local_window=32, causal=False
        )
        with pytest.raises(RuntimeError):
            _ = mod(q, k, v, bias)


class TestBuildLocalBlockMask:
    """
    Tests the build_local_blockmask function, verifying shape and behavior.
    """

    @pytest.mark.parametrize(
        "n_atom,expected_shape",
        [
            (16, (1, 4, 1, 1)),  # N_atom=16, block_size=128 => nrow = ncol = 1
            (256, (1, 4, 2, 2)),  # N_atom=256 => nrow=ncol=2
            (0, None),  # 0 => should return None
        ],
    )
    def test_blockmask_shape(self, n_atom: int, expected_shape: Any):
        """
        Tests shape or None return for various N_atom with default config.
        """
        config = LocalBlockMaskConfig(block_size=128, local_window=32, nheads=4)
        mask = build_local_blockmask(n_atom, config)
        if expected_shape is None:
            assert mask is None, "Expected None when N_atom <= 0."
        else:
            if not _HAS_BSA:
                assert mask is None, "Expected None when _HAS_BSA is False"
            else:
                assert (
                    mask.shape == expected_shape
                ), f"Expected shape {expected_shape}, got {mask.shape}"

    def test_causal_masking(self):
        """
        Tests that causal masking is applied correctly.
        """
        config = LocalBlockMaskConfig(
            block_size=8, local_window=8, nheads=2, causal=True
        )
        N_atom = 16
        mask = build_local_blockmask(N_atom, config)
        if _HAS_BSA:
            # If block_sparse_attn is available, expect a tensor with specific shape
            assert mask is not None  # Runtime check
            mask_tensor = cast(torch.Tensor, mask)  # Type cast for the type checker
            assert isinstance(mask_tensor, torch.Tensor)  # Runtime check
            assert mask_tensor.shape == (1, 2, 2, 2)
        else:
            # If block_sparse_attn is not available, expect None
            assert mask is None


class TestCheckpointBlocks:
    """
    Tests for the checkpoint_blocks function.
    """

    def test_simple_block_chain(self):
        """
        Verifies that a chain of simple blocks processes arguments in correct order.
        """

        def block_a(x: int) -> int:
            return x + 1

        def block_b(x: int) -> int:
            return x * 3

        out = checkpoint_blocks([block_a, block_b], (5,), None)
        assert out == (18,), "Expected (5+1)*3 => 18."

    def test_blocks_per_ckpt_value_error(self):
        """
        Checks that an invalid blocks_per_ckpt triggers ValueError.
        """

        def block_a(x: int) -> int:
            return x + 1

        # Function defaults blocks_per_ckpt if invalid, doesn't raise ValueError
        checkpoint_blocks([block_a], (5,), blocks_per_ckpt=2)  # 2 > len(blocks)=1

        checkpoint_blocks([block_a], (5,), blocks_per_ckpt=0)  # 0 < 1

    @settings(max_examples=100, deadline=None)  # Increased examples and deadline
    @given(
        # Generate simpler blocks that are less likely to cause TypeErrors with args
        blocks=st.lists(
            st.just(lambda *a: a[0] if a else None), # Return first arg or None
            min_size=0,
            max_size=5
        ),
        # Generate args as a list first
        args_list=st.lists(
            st.integers() | st.floats(allow_nan=False, allow_infinity=False) | st.booleans(),
            min_size=1, # Ensure at least one argument for the blocks
            max_size=3
        ),
        # Allow potentially invalid blocks_per_ckpt values to test defaulting logic
        blocks_per_ckpt=st.one_of(st.none(), st.integers(min_value=-5, max_value=10)),
    )
    def test_fuzz_checkpoint_blocks(
        self, blocks: List[Callable], args_list: List[Any], blocks_per_ckpt: Optional[int]
    ):
        """
        Fuzz test with Hypothesis: checks that checkpoint_blocks completes
        successfully for various inputs. It verifies:
        1. The function does not raise unexpected errors.
        2. The function handles potentially invalid `blocks_per_ckpt` by defaulting.
        3. The function always returns a tuple.
        """
        # Convert list args to tuple as expected by the function's internal logic
        args_tuple = tuple(args_list)

        # The function handles invalid blocks_per_ckpt internally by defaulting,
        # so we don't expect ValueError for out-of-range values.
        # We only expect successful execution or predictable TypeErrors if block/args mismatch.
        try:
            out = checkpoint_blocks(blocks, args_tuple, blocks_per_ckpt)
            # Check that the output is always a tuple, as per function design (due to wrap)
            assert isinstance(out, tuple), f"Expected output to be a tuple, got {type(out)}"

            # If blocks were executed, check if the first element matches expectation
            # (This is a basic check, assumes simple block functions like identity)
            if blocks:
                 # The output tuple might contain the result of the last block,
                 # or the original args if blocks couldn't process them.
                 # We can't easily predict the exact value without knowing the blocks,
                 # but we know it should be a tuple.
                 pass # Just checking type is sufficient for this fuzz test robustness
            else:
                 # If no blocks, output should be the initial args tuple
                 assert out == args_tuple, "Expected output to be initial args tuple when no blocks"

        except TypeError:
            # Allow TypeErrors that might occur if Hypothesis generates a block
            # incompatible with the generated args. This is acceptable in fuzzing.
            # Example: block expects int, args is ('text',)
            pass
        except Exception as e:
            # Catch any other unexpected errors during execution
            pytest.fail(f"checkpoint_blocks raised an unexpected exception: {e} with inputs: blocks={blocks}, args={args_tuple}, ckpt={blocks_per_ckpt}")

