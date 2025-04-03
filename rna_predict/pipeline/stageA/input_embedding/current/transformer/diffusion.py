"""
Diffusion transformer modules for RNA structure prediction.
"""

import logging
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn

from rna_predict.pipeline.stageA.input_embedding.current.checkpointing import (
    checkpoint_blocks,
)
from rna_predict.pipeline.stageA.input_embedding.current.transformer.attention import (
    AttentionPairBias,
)
from rna_predict.pipeline.stageA.input_embedding.current.transformer.common import (
    make_typed_partial,
)
from rna_predict.pipeline.stageA.input_embedding.current.transformer.transition import (
    ConditionedTransitionBlock,
)


class DiffusionTransformerBlock(nn.Module):
    """
    Single block of the diffusion transformer architecture.
    Implements Algorithm 23[Line2-Line3] in AlphaFold3.
    """

    def __init__(
        self,
        c_a: int,  # atom embedding dimension
        c_s: int,  # style embedding dimension
        c_z: int,  # pair embedding dimension
        n_heads: int,  # number of attention heads
        biasinit: float = -2.0,
    ) -> None:
        """
        Initialize the DiffusionTransformerBlock.

        Args:
            c_a: Single embedding dimension for atom features
            c_s: Single embedding dimension for style features
            c_z: Pair embedding dimension
            n_heads: Number of attention heads
            biasinit: Bias initialization value
        """
        super(DiffusionTransformerBlock, self).__init__()
        self.n_heads = n_heads
        self.c_a = c_a
        self.c_s = c_s
        self.c_z = c_z

        # Create attention module with pair bias
        self.attention_pair_bias = AttentionPairBias(
            has_s=True, n_heads=n_heads, c_a=c_a, c_s=c_s, c_z=c_z, biasinit=biasinit
        )

        # Create feed-forward network with conditioning
        self.conditioned_transition_block = ConditionedTransitionBlock(
            n=2, c_a=c_a, c_s=c_s, biasinit=biasinit
        )

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        n_queries: Optional[int] = None,
        n_keys: Optional[int] = None,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process input through the diffusion transformer block.

        Args:
            a: Single feature aggregate per-atom representation [..., N, c_a]
            s: Single embedding [..., N, c_s]
            z: Pair embedding [..., N, N, c_z] or [..., n_block, n_queries, n_keys, c_z]
            n_queries: Local window size of query tensor. If not None, will perform local attention.
            n_keys: Local window size of key tensor.
            inplace_safe: Whether it is safe to use inplace operations.
            chunk_size: Chunk size for memory-efficient operations.

        Returns:
            Tuple containing:
                - Updated representation (out_a)
                - Original single embedding (s) - preserved for checkpointing
                - Original pair embedding (z) - preserved for checkpointing
        """
        # Apply attention with pair bias
        attn_out = self.attention_pair_bias(
            a=a,
            s=s,
            z=z,
            n_queries=n_queries,
            n_keys=n_keys,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )

        # Residual connection after attention
        if inplace_safe:
            attn_out += a
        else:
            attn_out = attn_out + a

        # Apply feed-forward network
        ff_out = self.conditioned_transition_block(a=attn_out, s=s)

        # Residual connection after feed-forward
        out_a = ff_out + attn_out

        # Return tuple including s and z to avoid deletion by torch.utils.checkpoint
        return out_a, s, z


class DiffusionTransformer(nn.Module):
    """
    Multi-block diffusion transformer architecture.
    Implements Algorithm 23 in AlphaFold3.
    """

    def __init__(
        self,
        c_a: int,  # atom embedding dimension
        c_s: int,  # style embedding dimension
        c_z: int,  # pair embedding dimension
        n_blocks: int,  # number of transformer blocks
        n_heads: int,  # number of attention heads
        blocks_per_ckpt: Optional[int] = None,
    ) -> None:
        """
        Initialize the DiffusionTransformer.

        Args:
            c_a: Dimension of single embedding (atom features)
            c_s: Dimension of single embedding (style/conditioning)
            c_z: Dimension of pair embedding
            n_blocks: Number of transformer blocks
            n_heads: Number of attention heads
            blocks_per_ckpt: Number of blocks per checkpoint for memory efficiency

        Raises:
            ValueError: If any dimension is invalid
        """
        super(DiffusionTransformer, self).__init__()

        # Validate input dimensions
        if c_a <= 0 or c_s <= 0 or c_z <= 0:
            raise ValueError(
                f"All embedding dimensions must be positive. "
                f"Got c_a={c_a}, c_s={c_s}, c_z={c_z}"
            )

        if n_blocks <= 0:
            raise ValueError(
                f"Number of blocks must be positive. Got n_blocks={n_blocks}"
            )

        if n_heads <= 0:
            raise ValueError(f"Number of heads must be positive. Got n_heads={n_heads}")

        # Store parameters
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.c_a = c_a
        self.c_s = c_s
        self.c_z = c_z
        self.blocks_per_ckpt = blocks_per_ckpt

        # Create transformer blocks
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            block = DiffusionTransformerBlock(
                n_heads=n_heads, c_a=c_a, c_s=c_s, c_z=c_z
            )
            self.blocks.append(block)

        # Log creation information
        logger = logging.getLogger(__name__)
        logger.debug(
            f"Initialized DiffusionTransformer with {n_blocks} blocks, "
            f"{n_heads} heads, dimensions: c_a={c_a}, c_s={c_s}, c_z={c_z}"
        )

    def _prep_blocks(
        self,
        n_queries: Optional[int] = None,
        n_keys: Optional[int] = None,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        clear_cache_between_blocks: bool = False,
    ) -> List[
        Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ]
    ]:
        """
        Prepare blocks for processing with specific parameters.

        Args:
            n_queries: Number of queries for local attention
            n_keys: Number of keys for local attention
            inplace_safe: Whether inplace operations are safe
            chunk_size: Size of chunks for memory efficiency
            clear_cache_between_blocks: Whether to clear cache between blocks

        Returns:
            List of callable blocks ready for processing
        """
        # Create callable blocks with fixed parameters
        BlockType = Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ]

        blocks: List[BlockType] = []
        for block in self.blocks:
            # Create a typed partial function for each block
            func = make_typed_partial(
                block,
                n_queries=n_queries,
                n_keys=n_keys,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
            blocks.append(func)

        # If cache clearing is requested, wrap each block
        if clear_cache_between_blocks:
            wrapped_blocks = []
            for block_func in blocks:
                # Create cache-clearing wrapper
                def clear_cache_wrapper(
                    b: BlockType, *args: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                    """
                    Wrapper to clear cache before executing a block.

                    Args:
                        b: The block function to execute
                        args: Positional arguments for the block

                    Returns:
                        Result of the block execution
                    """
                    # Could uncomment for actual CUDA implementations:
                    # torch.cuda.empty_cache()
                    return b(*args)

                # Create a typed partial function that wraps the block with cache clearing
                wrapped = make_typed_partial(clear_cache_wrapper, b=block_func)
                wrapped_blocks.append(wrapped)

            blocks = wrapped_blocks

        return blocks

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        n_queries: Optional[int] = None,
        n_keys: Optional[int] = None,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the diffusion transformer.

        Args:
            a: Single feature aggregate per-atom representation [..., N, c_a]
            s: Single embedding [..., N, c_s]
            z: Pair embedding [..., N, N, c_z] or [..., n_blocks, n_queries, n_keys, c_z]
            n_queries: Local window size of query tensor. If not None, will perform local attention.
            n_keys: Local window size of key tensor.
            inplace_safe: Whether it is safe to use inplace operations.
            chunk_size: Chunk size for memory-efficient operations.

        Returns:
            Updated atom representation
        """
        # Determine whether to clear cache between blocks
        clear_cache_between_blocks = False
        if hasattr(z, "shape") and len(z.shape) >= 3:
            if z.shape[-2] > 2000 and not self.training:
                clear_cache_between_blocks = True

        # Prepare blocks with common parameters
        blocks = self._prep_blocks(
            n_queries=n_queries,
            n_keys=n_keys,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            clear_cache_between_blocks=clear_cache_between_blocks,
        )

        # Determine whether to use checkpointing
        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None

        # Process through blocks with optional checkpointing
        a, s, z = checkpoint_blocks(
            blocks, args=(a, s, z), blocks_per_ckpt=blocks_per_ckpt
        )

        # Clean up to free memory
        del s, z

        return a
