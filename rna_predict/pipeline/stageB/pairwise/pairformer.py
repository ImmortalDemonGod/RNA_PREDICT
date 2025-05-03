# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pairformer implementation for RNA_PREDICT Pipeline Stage B.

This module provides the core implementation of the Pairformer model,
integrated with the Hydra configuration system for flexible parameter management.

The module includes several components:
- PairformerBlock: Implements a single block of the Pairformer architecture
- PairformerStack: Stacks multiple PairformerBlocks together
- MSAPairWeightedAveraging: Implements weighted averaging for MSA pairs
- MSAStack: Implements the MSA stack
- MSABlock: Implements a single block of the MSA module
- MSAModule: Implements the full MSA module
- TemplateEmbedder: Implements template embedding
"""

from functools import partial
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig

from protenix.openfold_local.model.dropout import DropoutRowwise
from protenix.openfold_local.model.outer_product_mean import (
    OuterProductMean,  # Alg 9 in AF3
)
from rna_predict.pipeline.stageB.pairwise.triangular_attention import TriangleAttention
from rna_predict.pipeline.stageB.pairwise.triangular_multiplicative import (
    TriangleMultiplicationIncoming,  # Alg 13 in AF3
    TriangleMultiplicationOutgoing,  # Alg 12 in AF3
)

from rna_predict.conf.config_schema import (
    PairformerBlockConfig,
    PairformerStackConfig,
    MSAConfig,
    TemplateEmbedderConfig,
)

from rna_predict.pipeline.stageA.input_embedding.current.checkpointing import (
    checkpoint_blocks,
)
from rna_predict.pipeline.stageA.input_embedding.current.primitives import (
    LayerNorm,
    LinearNoBias,
    Transition,
)
from rna_predict.pipeline.stageA.input_embedding.current.transformer import (
    AttentionPairBias,
)
from rna_predict.pipeline.stageA.input_embedding.current.utils import (
    sample_msa_feature_dict_random_without_replacement,
)


class PairformerBlock(nn.Module):
    """Implements Algorithm 17 [Line2-Line8] in AF3
    c_hidden_mul is set as openfold
    Ref to:
    https://github.com/aqlaboratory/openfold/blob/feb45a521e11af1db241a33d58fb175e207f8ce0/openfold/model/evoformer.py#L123
    """

    def __init__(
        self,
        cfg: Union[DictConfig, PairformerBlockConfig],
    ) -> None:
        """
        Initialize a PairformerBlock with configuration.

        Args:
            cfg: Configuration object containing parameters for the PairformerBlock.
                Can be either a DictConfig from Hydra or a PairformerBlockConfig.
        """
        super(PairformerBlock, self).__init__()

        # Extract parameters from config
        self.n_heads = cfg.n_heads
        self.c_z = cfg.c_z
        self.c_s = cfg.c_s
        self.dropout = cfg.dropout
        c_hidden_mul = cfg.c_hidden_mul
        c_hidden_pair_att = cfg.c_hidden_pair_att
        no_heads_pair = cfg.no_heads_pair

        # Initialize components
        self.tri_mul_out = TriangleMultiplicationOutgoing(
            c_z=self.c_z, c_hidden=c_hidden_mul
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(c_z=self.c_z, c_hidden=c_hidden_mul)
        self.tri_att_start = TriangleAttention(
            c_in=self.c_z,
            c_hidden=c_hidden_pair_att,
            no_heads=no_heads_pair,
        )
        self.tri_att_end = TriangleAttention(
            c_in=self.c_z,
            c_hidden=c_hidden_pair_att,
            no_heads=no_heads_pair,
        )
        self.dropout_row = DropoutRowwise(self.dropout)
        self.pair_transition = Transition(c_in=self.c_z, n=4)

        # Initialize single representation components if needed
        if self.c_s > 0:
            # Enable single‑rep conditioning and gating
            self.attention_pair_bias = AttentionPairBias(
                has_s=True,  # <-- enable s‑conditioning
                n_heads=self.n_heads,
                c_a=self.c_s,
                c_z=self.c_z,
            )
            self.single_transition = Transition(c_in=self.c_s, n=4)

    def forward(
        self,
        s: Optional[torch.Tensor],
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass of the PairformerBlock.

        Args:
            s (Optional[torch.Tensor]): single feature
                [..., N_token, c_s]
            z (torch.Tensor): pair embedding
                [..., N_token, N_token, c_z]
            pair_mask (torch.Tensor): pair mask
                [..., N_token, N_token]
            use_memory_efficient_kernel (bool): Whether to use memory-efficient kernel. Defaults to False.
            use_deepspeed_evo_attention (bool): Whether to use DeepSpeed evolutionary attention. Defaults to False.
            use_lma (bool): Whether to use low-memory attention. Defaults to False.
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            tuple[Optional[torch.Tensor], torch.Tensor]: the update of s[Optional] and z
                [..., N_token, c_s] | None
                [..., N_token, N_token, c_z]
        """
        # Ensure float pair_mask
        if pair_mask is not None and pair_mask.dtype != torch.float32:
            pair_mask = pair_mask.float()
        if inplace_safe:
            z = self.tri_mul_out(
                z, mask=pair_mask, inplace_safe=inplace_safe, _add_with_inplace=True
            )
            z = self.tri_mul_in(
                z, mask=pair_mask, inplace_safe=inplace_safe, _add_with_inplace=True
            )
            z += self.tri_att_start(
                z,
                mask=pair_mask,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
            z = z.transpose(-2, -3).contiguous()
            z += self.tri_att_end(
                z,
                mask=pair_mask.transpose(-1, -2) if pair_mask is not None else None,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
            z = z.transpose(-2, -3).contiguous()
            z += self.pair_transition(z)
            if self.c_s > 0:
                s += self.attention_pair_bias(
                    a=s,
                    s=s,  # Pass s for conditioning and gating
                    z=z,
                )
                s += self.single_transition(s)
            return s, z
        else:
            # again ensure float after we unify code paths
            if pair_mask is not None and pair_mask.dtype != torch.float32:
                pair_mask = pair_mask.float()

            tmu_update = self.tri_mul_out(
                z, mask=pair_mask, inplace_safe=inplace_safe, _add_with_inplace=False
            )
            z = z + self.dropout_row(tmu_update)
            del tmu_update
            tmu_update = self.tri_mul_in(
                z, mask=pair_mask, inplace_safe=inplace_safe, _add_with_inplace=False
            )
            z = z + self.dropout_row(tmu_update)
            del tmu_update
            z = z + self.dropout_row(
                self.tri_att_start(
                    z,
                    mask=pair_mask,
                    use_memory_efficient_kernel=use_memory_efficient_kernel,
                    use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                    use_lma=use_lma,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                )
            )
            z = z.transpose(-2, -3)
            z = z + self.dropout_row(
                self.tri_att_end(
                    z,
                    mask=pair_mask.transpose(-1, -2) if pair_mask is not None else None,
                    use_memory_efficient_kernel=use_memory_efficient_kernel,
                    use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                    use_lma=use_lma,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                )
            )
            z = z.transpose(-2, -3)

            z = z + self.pair_transition(z)
            if self.c_s > 0:
                s = s + self.attention_pair_bias(
                    a=s,
                    s=s,  # Pass s for conditioning and gating
                    z=z,
                )
                s = s + self.single_transition(s)
            return s, z


class PairformerStack(nn.Module):
    """
    Implements Algorithm 17 [PairformerStack] in AF3
    """

    def __init__(
        self,
        cfg: Union[DictConfig, PairformerStackConfig],
    ) -> None:
        """
        Initialize a PairformerStack with configuration.

        Args:
            cfg: Configuration object containing parameters for the PairformerStack.
                Can be either a DictConfig from Hydra or a PairformerStackConfig.
        """
        super(PairformerStack, self).__init__()

        # Extract parameters from config
        self.n_blocks = cfg.n_blocks
        self.n_heads = cfg.n_heads
        self.c_z = cfg.c_z
        self.c_s = cfg.c_s
        self.dropout = cfg.dropout
        self.blocks_per_ckpt = cfg.blocks_per_ckpt

        # Create block configuration with all fields from PairformerBlockConfig

        # Helper function to sanitize integer values
        def _sanitize(val: Any, default: int, name: str) -> int:
            try:
                v = int(val)
            except (TypeError, ValueError):
                return default
            if v <= 0:
                raise ValueError(f"{name} must be > 0, got {v}")
            return v

        # Get values from config with validation, using defaults from PairformerBlockConfig
        c_hidden_mul = _sanitize(getattr(cfg, "c_hidden_mul", 4), 4, "c_hidden_mul")
        c_hidden_pair_att = _sanitize(getattr(cfg, "c_hidden_pair_att", 32), 32, "c_hidden_pair_att")
        no_heads_pair = _sanitize(getattr(cfg, "no_heads_pair", 8), 8, "no_heads_pair")

        # Create a new PairformerBlockConfig with explicit parameters
        block_cfg = PairformerBlockConfig(
            n_heads=self.n_heads,  # Force n_heads to match stack
            c_z=self.c_z,  # Force c_z to match stack
            c_s=self.c_s,  # Now set in Hydra config for pair stack
            c_hidden_mul=c_hidden_mul,
            c_hidden_pair_att=c_hidden_pair_att,
            no_heads_pair=no_heads_pair,
            dropout=self.dropout  # Force dropout to match stack
        )

        # Initialize blocks
        self.blocks = nn.ModuleList()
        for _ in range(self.n_blocks):
            block = PairformerBlock(cfg=block_cfg)
            self.blocks.append(block)

    def _prep_blocks(
        self,
        pair_mask: Optional[torch.Tensor],
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        clear_cache_between_blocks: bool = False,
    ):
        blocks = [
            partial(
                b,
                pair_mask=pair_mask,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
            for b in self.blocks
        ]

        def clear_cache(b, *args, **kwargs):
            torch.cuda.empty_cache()
            return b(*args, **kwargs)

        if clear_cache_between_blocks:
            blocks = [partial(clear_cache, b) for b in blocks]
        return blocks

    def forward(
        self,
        s: Optional[torch.Tensor],
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Args:
            s (Optional[torch.Tensor]): single feature
                [..., N_token, c_s]
            z (torch.Tensor): pair embedding
                [..., N_token, N_token, c_z]
            pair_mask (torch.Tensor): pair mask
                [..., N_token, N_token]
            use_memory_efficient_kernel (bool): Whether to use memory-efficient kernel. Defaults to False.
            use_deepspeed_evo_attention (bool): Whether to use DeepSpeed evolutionary attention. Defaults to False.
            use_lma (bool): Whether to use low-memory attention. Defaults to False.
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            tuple[Optional[torch.Tensor], torch.Tensor]: the update of s and z
                [..., N_token, c_s] or None if s is None
                [..., N_token, N_token, c_z]
        """
        if z.shape[-2] > 2000 and (not self.training):
            clear_cache_between_blocks = True
        else:
            clear_cache_between_blocks = False

        # Convert pair_mask to float
        if pair_mask is not None and pair_mask.dtype != torch.float32:
            pair_mask = pair_mask.float()
        blocks = self._prep_blocks(
            pair_mask=pair_mask,
            use_memory_efficient_kernel=use_memory_efficient_kernel,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            clear_cache_between_blocks=clear_cache_between_blocks,
        )

        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None
        s, z = checkpoint_blocks(
            blocks,
            args=(s, z),
            blocks_per_ckpt=blocks_per_ckpt,
        )
        return s, z


class MSAPairWeightedAveraging(nn.Module):
    """
    Implements Algorithm 10 [MSAPairWeightedAveraging] in AF3
    """

    def __init__(self, cfg: Union[DictConfig, MSAConfig]) -> None:
        """
        Initialize MSAPairWeightedAveraging with configuration.

        Args:
            cfg: Configuration object containing parameters for MSAPairWeightedAveraging.
                Can be either a DictConfig from Hydra or an MSAConfig.
        """
        super(MSAPairWeightedAveraging, self).__init__()

        # Validate required parameters
        required_params = ["c_m", "c", "c_z", "n_heads"]
        for param in required_params:
            if not hasattr(cfg, param):
                raise ValueError(f"Configuration missing required parameter: {param}")

        # Extract parameters from config
        self.c_m = cfg.c_m
        self.c = cfg.c
        self.c_z = cfg.c_z
        self.n_heads = cfg.n_heads

        # Input projections
        self.layernorm_m = LayerNorm(self.c_m)
        self.linear_no_bias_mv = LinearNoBias(
            in_features=self.c_m, out_features=self.c * self.n_heads
        )
        self.layernorm_z = LayerNorm(self.c_z)
        self.linear_no_bias_z = LinearNoBias(
            in_features=self.c_z, out_features=self.n_heads
        )
        self.linear_no_bias_mg = LinearNoBias(
            in_features=self.c_m, out_features=self.c * self.n_heads
        )
        # Weighted average with gating
        self.softmax_w = nn.Softmax(dim=-2)
        # Output projection
        self.linear_no_bias_out = LinearNoBias(
            in_features=self.c * self.n_heads, out_features=self.c_m
        )

    def forward(self, m: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            m (torch.Tensor): msa embedding
                [...,n_msa_sampled, n_token, c_m]
            z (torch.Tensor): pair embedding
                [...,n_token, n_token, c_z]
        Returns:
            torch.Tensor: updated msa embedding
                [...,n_msa_sampled, n_token, c_m]
        """
        # Input projections
        m = self.layernorm_m(m)  # [...,n_msa_sampled, n_token, c_m]
        v = self.linear_no_bias_mv(m)  # [...,n_msa_sampled, n_token, n_heads * c]
        v = v.reshape(
            *v.shape[:-1], self.n_heads, self.c
        )  # [...,n_msa_sampled, n_token, n_heads, c]
        b = self.linear_no_bias_z(
            self.layernorm_z(z)
        )  # [...,n_token, n_token, n_heads]
        g = torch.sigmoid(
            self.linear_no_bias_mg(m)
        )  # [...,n_msa_sampled, n_token, n_heads * c]
        g = g.reshape(
            *g.shape[:-1], self.n_heads, self.c
        )  # [...,n_msa_sampled, n_token, n_heads, c]
        w = self.softmax_w(b)  # [...,n_token, n_token, n_heads]
        wv = torch.einsum(
            "...ijh,...mjhc->...mihc", w, v
        )  # [...,n_msa_sampled,n_token,n_heads,c]
        o = g * wv
        o = o.reshape(
            *o.shape[:-2], self.n_heads * self.c
        )  # [...,n_msa_sampled, n_token, n_heads * c]
        m = self.linear_no_bias_out(o)  # [...,n_msa_sampled, n_token, c_m]
        return m


class MSAStack(nn.Module):
    """
    Implements MSAStack Line7-Line8 in Algorithm 8
    """

    def __init__(self, cfg: Union[DictConfig, MSAConfig]) -> None:
        """
        Initialize MSAStack with configuration.

        Args:
            cfg: Configuration object containing parameters for MSAStack.
                Can be either a DictConfig from Hydra or an MSAConfig.
        """
        super(MSAStack, self).__init__()

        # Validate required parameters
        required_params = ["c_m", "c", "dropout"]
        for param in required_params:
            if not hasattr(cfg, param):
                raise ValueError(f"Configuration missing required parameter: {param}")

        # Extract parameters from config
        self.c_m = cfg.c_m
        self.c = cfg.c
        self.dropout = cfg.dropout

        # Initialize components with config
        self.msa_pair_weighted_averaging = MSAPairWeightedAveraging(cfg)
        self.dropout_row = DropoutRowwise(self.dropout)
        self.transition_m = Transition(c_in=self.c_m, n=4)

    def forward(self, m: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            m (torch.Tensor): msa embedding
                [...,n_msa_sampled, n_token, c_m]
            z (torch.Tensor): pair embedding
                [...,n_token, n_token, c_z]

        Returns:
            torch.Tensor: updated msa embedding
                [...,n_msa_sampled, n_token, c_m]
        """
        m = m + self.dropout_row(self.msa_pair_weighted_averaging(m, z))
        m = m + self.transition_m(m)
        return m


class MSABlock(nn.Module):
    """
    Base MSA Block, Line6-Line13 in Algorithm 8
    """

    def __init__(
        self,
        cfg: Union[DictConfig, MSAConfig],
        is_last_block: bool = False,
    ) -> None:
        """
        Initialize MSABlock with configuration.

        Args:
            cfg: Configuration object containing parameters for MSABlock.
                Can be either a DictConfig from Hydra or an MSAConfig.
            is_last_block: Whether this is the last block of MSAModule. Defaults to False.
        """
        super(MSABlock, self).__init__()

        # Validate required parameters
        required_params = ["c_m", "c", "c_z", "dropout"]
        for param in required_params:
            if not hasattr(cfg, param):
                raise ValueError(f"Configuration missing required parameter: {param}")

        # Extract parameters from config
        self.c_m = cfg.c_m
        self.c_z = cfg.c_z
        self.c_hidden = cfg.c  # Use c as c_hidden
        self.is_last_block = is_last_block
        self.msa_dropout = cfg.dropout
        self.pair_dropout = getattr(cfg, "pair_dropout", 0.25)  # Default to 0.25 if not provided

        # Communication
        self.outer_product_mean_msa = OuterProductMean(
            c_m=self.c_m, c_z=self.c_z, c_hidden=self.c_hidden
        )

        if not self.is_last_block:
            # MSA stack
            # Create MSA config for MSAStack
            msa_stack_cfg = MSAConfig(
                c_m=self.c_m,
                c=self.c_hidden,
                dropout=self.msa_dropout
            )
            self.msa_stack = MSAStack(cfg=msa_stack_cfg)

        # Pair stack - create configuration from Hydra config with defaults for missing values
        pair_block_cfg = PairformerBlockConfig(
            n_heads=getattr(cfg, "n_heads", 2),  # Default to 2 if not provided
            c_z=cfg.c_z,
            c_s=0,  # Default to 0 since we don't use single representation in MSABlock
            c_hidden_mul=4,  # Default to 4
            c_hidden_pair_att=4,  # Default to 4
            no_heads_pair=2,  # Default to 2
            dropout=self.pair_dropout
        )
        self.pair_stack = PairformerBlock(cfg=pair_block_cfg)

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        pair_mask,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Args:
            m (torch.Tensor): msa embedding
                [...,n_msa_sampled, n_token, c_m]
            z (torch.Tensor): pair embedding
                [...,n_token, n_token, c_z]
            pair_mask (torch.Tensor): pair mask
                [..., N_token, N_token]
            use_memory_efficient_kernel (bool): Whether to use memory-efficient kernel. Defaults to False.
            use_deepspeed_evo_attention (bool): Whether to use DeepSpeed evolutionary attention. Defaults to False.
            use_lma (bool): Whether to use low-memory attention. Defaults to False.
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: updated m z of MSABlock
                [...,n_msa_sampled, n_token, c_m]
                [...,n_token, n_token, c_z]
        """
        # Communication
        z = z + self.outer_product_mean_msa(
            m, inplace_safe=inplace_safe, chunk_size=chunk_size
        )
        if not self.is_last_block:
            # MSA stack
            m = self.msa_stack(m, z)
        # Pair stack
        _, z = self.pair_stack(
            s=None,
            z=z,
            pair_mask=pair_mask,
            use_memory_efficient_kernel=use_memory_efficient_kernel,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )

        if not self.is_last_block:
            return m, z
        else:
            return None, z  # to ensure that `m` will not be used.


class MSAModule(nn.Module):
    """
    Implements Algorithm 8 [MSAModule] in AF3
    """

    def __init__(
        self,
        cfg: Union[DictConfig, MSAConfig],
    ) -> None:
        """Main Entry of MSAModule

        Args:
            cfg: Configuration object containing parameters for MSAModule.
                Can be either a DictConfig from Hydra or an MSAConfig.
        """
        super(MSAModule, self).__init__()

        # Validate required parameters
        required_params = ["c_m", "c", "c_z", "dropout", "n_blocks"]
        for param in required_params:
            if not hasattr(cfg, param):
                raise ValueError(f"Configuration missing required parameter: {param}")

        # Extract parameters from config with defaults for missing values
        self.n_blocks = cfg.n_blocks
        self.c_m = cfg.c_m
        self.c = cfg.c
        self.dropout = cfg.dropout
        self.c_s_inputs = getattr(cfg, "c_s_inputs", 8)  # Default to 8 if not provided
        self.blocks_per_ckpt = getattr(cfg, "blocks_per_ckpt", 1)  # Default to 1 if not provided
        self.c_z = cfg.c_z

        # Input feature dimensions from config with default if not provided
        default_input_feature_dims = {"msa": 32, "has_deletion": 1, "deletion_value": 1}
        self.input_feature = getattr(cfg, "input_feature_dims", default_input_feature_dims)

        # Set up msa_configs from the structured config with defaults
        self.msa_configs = {
            "enable": getattr(cfg, "enable", False),  # Default to False if not provided
            "strategy": getattr(cfg, "strategy", "random"),  # Default to "random" if not provided
            "train_cutoff": getattr(cfg, "train_cutoff", 512),  # Default to 512 if not provided
            "test_cutoff": getattr(cfg, "test_cutoff", 16384),  # Default to 16384 if not provided
            "train_lowerb": getattr(cfg, "train_lowerb", 1),  # Default to 1 if not provided
            "test_lowerb": getattr(cfg, "test_lowerb", 1),  # Default to 1 if not provided
        }

        # Initialize linear layers
        self.linear_no_bias_m = LinearNoBias(
            in_features=32 + 1 + 1, out_features=self.c_m
        )

        self.linear_no_bias_s = LinearNoBias(
            in_features=self.c_s_inputs, out_features=self.c_m
        )

        # Initialize MSA blocks
        self.blocks = nn.ModuleList()

        # Create a base MSA config for blocks with all required parameters
        msa_block_cfg = MSAConfig(
            c_m=self.c_m,
            c=self.c,
            c_z=self.c_z,  # Add c_z to MSAConfig
            dropout=self.dropout,
            n_blocks=1,  # Not used in MSABlock
            enable=self.msa_configs["enable"],
            strategy=self.msa_configs["strategy"],
            train_cutoff=self.msa_configs["train_cutoff"],
            test_cutoff=self.msa_configs["test_cutoff"],
            train_lowerb=self.msa_configs["train_lowerb"],
            test_lowerb=self.msa_configs["test_lowerb"],
            # Add additional parameters needed by MSABlock
            pair_dropout=getattr(cfg, "pair_dropout", 0.25),  # Default to 0.25 if not provided
            n_heads=getattr(cfg, "n_heads", 2),  # Default to 2 if not provided
            c_s_inputs=self.c_s_inputs,
            blocks_per_ckpt=self.blocks_per_ckpt,
            input_feature_dims=self.input_feature
        )

        for i in range(self.n_blocks):
            block = MSABlock(
                cfg=msa_block_cfg,
                is_last_block=(i + 1 == self.n_blocks),
            )
            self.blocks.append(block)

    def _prep_blocks(
        self,
        pair_mask: Optional[torch.Tensor],
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        clear_cache_between_blocks: bool = False,
    ):
        blocks = [
            partial(
                b,
                pair_mask=pair_mask,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
            for b in self.blocks
        ]

        def clear_cache(b, *args, **kwargs):
            torch.cuda.empty_cache()
            return b(*args, **kwargs)

        if clear_cache_between_blocks:
            blocks = [partial(clear_cache, b) for b in blocks]
        return blocks

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        z: torch.Tensor,
        s_inputs: torch.Tensor,
        pair_mask: torch.Tensor,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_feature_dict (dict[str, Any]):
                input meta feature dict
            z (torch.Tensor): pair embedding
                [..., N_token, N_token, c_z]
            s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
                [..., N_token, c_s_inputs]
            pair_mask (torch.Tensor): pair mask
                [..., N_token, N_token]
            use_memory_efficient_kernel (bool): Whether to use memory-efficient kernel. Defaults to False.
            use_deepspeed_evo_attention (bool): Whether to use DeepSpeed evolutionary attention. Defaults to False.
            use_lma (bool): Whether to use low-memory attention. Defaults to False.
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            torch.Tensor: the updated z
                [..., N_token, N_token, c_z]
        """
        # If n_blocks < 1, return z
        if self.n_blocks < 1:
            return z

        if "msa" not in input_feature_dict:
            return z

        # Convert pair_mask to float if needed
        if pair_mask is not None and pair_mask.dtype != torch.float32:
            pair_mask = pair_mask.float()

        # Sample MSA features with explicit type conversion
        sample_size = int(self.msa_configs["train_cutoff"] if self.training else self.msa_configs["test_cutoff"])
        msa_feat = sample_msa_feature_dict_random_without_replacement(
            feat_dict=input_feature_dict,
            # dim_dict removed as it's not an accepted argument
            sample_size=sample_size
        )
        # pylint: disable=E1102
        msa_feat["msa"] = torch.nn.functional.one_hot(
            msa_feat["msa"],
            num_classes=self.input_feature["msa"],
        )

        # E.g. if "msa" => shape [n_msa, n_token, 32]
        target_shape = msa_feat["msa"].shape[:2]  # (n_msa, n_token), ignoring the 32

        # Build combined features
        # Each named feature is reshaped to [n_msa, n_token, dimension], then concatenated
        # "msa": 32, "has_deletion":1, "deletion_value":1
        feat_list = []
        for name, dim in self.input_feature.items():
            x = msa_feat[name]
            # Reshape to [*target_shape, dim]
            x = x.reshape(*target_shape, dim)
            feat_list.append(x)
        msa_sample = torch.cat(
            feat_list, dim=-1
        )  # [n_msa, n_token, 32+1+1] = [n_msa, n_token, 34]
        print(f"[DEBUG][MSAModule] msa_sample.shape before linear: {msa_sample.shape}")
        # Line2
        msa_sample = self.linear_no_bias_m(msa_sample)

        # Auto broadcast [...,n_msa_sampled, n_token, c_m]
        msa_sample = msa_sample + self.linear_no_bias_s(s_inputs)
        if z.shape[-2] > 2000 and (not self.training):
            clear_cache_between_blocks = True
        else:
            clear_cache_between_blocks = False
        blocks = self._prep_blocks(
            pair_mask=pair_mask,
            use_memory_efficient_kernel=use_memory_efficient_kernel,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            clear_cache_between_blocks=clear_cache_between_blocks,
        )
        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None
        msa_sample, z = checkpoint_blocks(
            blocks,
            args=(msa_sample, z),
            blocks_per_ckpt=blocks_per_ckpt,
        )
        if z.shape[-2] > 2000:
            torch.cuda.empty_cache()
        return z


class TemplateEmbedder(nn.Module):
    """
    Implements Algorithm 16 in AF3
    """

    def __init__(self, cfg: Union[DictConfig, TemplateEmbedderConfig]) -> None:
        """
        Initialize TemplateEmbedder with configuration.

        Args:
            cfg: Configuration object containing parameters for TemplateEmbedder.
                Can be either a DictConfig from Hydra or a TemplateEmbedderConfig.
        """
        super(TemplateEmbedder, self).__init__()

        # Validate required parameters
        required_params = ["n_blocks", "c", "c_z", "dropout", "blocks_per_ckpt", "input_feature_dims", "distogram"]
        for param in required_params:
            if not hasattr(cfg, param):
                raise ValueError(f"Configuration missing required parameter: {param}")

        # Extract parameters from config
        self.n_blocks = cfg.n_blocks
        self.c = cfg.c
        self.c_z = cfg.c_z
        self.dropout = cfg.dropout
        self.blocks_per_ckpt = cfg.blocks_per_ckpt

        # Get input feature dimensions from config
        self.input_feature1 = cfg.input_feature_dims["feature1"]
        self.input_feature2 = cfg.input_feature_dims["feature2"]

        # Get distogram parameters from config
        self.distogram = cfg.distogram
        self.inf = 100000.0  # This could also be moved to config

        self.linear_no_bias_z = LinearNoBias(in_features=self.c_z, out_features=self.c)
        self.layernorm_z = LayerNorm(self.c_z)
        self.linear_no_bias_a = LinearNoBias(
            in_features=sum(self.input_feature1.values())
            + sum(self.input_feature2.values()),
            out_features=self.c,
        )
        # Create PairformerStack configuration
        stack_cfg = PairformerStackConfig(
            c_s=0,
            c_z=self.c,
            n_blocks=self.n_blocks,
            dropout=self.dropout,
            blocks_per_ckpt=self.blocks_per_ckpt,
        )
        self.pairformer_stack = PairformerStack(cfg=stack_cfg)
        self.layernorm_v = LayerNorm(self.c)
        self.linear_no_bias_u = LinearNoBias(in_features=self.c, out_features=self.c_z)

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        z: torch.Tensor,  # pylint: disable=W0613
        pair_mask: Optional[torch.Tensor] = None,  # pylint: disable=W0613
        use_memory_efficient_kernel: bool = False,  # pylint: disable=W0613
        use_deepspeed_evo_attention: bool = False,  # pylint: disable=W0613
        use_lma: bool = False,  # pylint: disable=W0613
        inplace_safe: bool = False,  # pylint: disable=W0613
        chunk_size: Optional[int] = None,  # pylint: disable=W0613
    ) -> torch.Tensor:
        """
        Args:
            input_feature_dict (dict[str, Any]): input feature dict
            z (torch.Tensor): pair embedding
                [..., N_token, N_token, c_z]
            pair_mask (torch.Tensor, optional): pair masking. Default to None.
                [..., N_token, N_token]

        Returns:
            torch.Tensor: the template feature
                [..., N_token, N_token, c_z]
        """
        # Per Algorithm 16 and typical usage, return 0 if templates aren't used or n_blocks is 0.
        if "template_restype" not in input_feature_dict or self.n_blocks < 1:
            return torch.zeros_like(z)  # Return zero tensor with same shape as z

        # If templates are present but logic isn't fully implemented,
        # also return 0 for now, consistent with the original behavior's intent.
        # TODO: Implement the actual template embedding logic here when ready.
        # For now, maintain the behavior of returning 0 if the main condition isn't met.
        return torch.zeros_like(z)  # Return zero tensor with same shape as z
