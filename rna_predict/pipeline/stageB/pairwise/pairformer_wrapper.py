"""
Pairformer Wrapper Module for RNA_PREDICT Pipeline Stage B.

This module provides a wrapper around the PairformerStack model, integrating it
with the Hydra configuration system and ensuring proper tensor dimension handling.

Configuration Requirements:
    The module expects a Hydra configuration with the following structure:
    - stageB_pairformer:
        - n_blocks: Number of transformer blocks
        - n_heads: Number of attention heads
        - c_z: Dimension of pair (Z) embeddings
        - c_s: Dimension of single (S) embeddings
        - dropout: Dropout rate
        - device: Device to run on (cpu, cuda, mps)
        - use_memory_efficient_kernel: Whether to use memory efficient attention
        - use_deepspeed_evo_attention: Whether to use DeepSpeed evolution attention
        - use_lma: Whether to use linear multi-head attention
        - inplace_safe: Whether to use inplace operations safely
        - chunk_size: Chunk size for attention computation
        - lora: LoRA configuration (optional)
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig

# Import structured configs
from rna_predict.conf.config_schema import PairformerStackConfig

from rna_predict.pipeline.stageB.pairwise.pairformer import PairformerStack

logger = logging.getLogger(__name__)

class PairformerWrapper(nn.Module):
    """
    Integrates Protenix's PairformerStack into our pipeline for global pairwise encoding.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize PairformerWrapper with configuration.

        Args:
            cfg: Hydra configuration object containing either:
                - stageB_pairformer section (root level), or
                - model.stageB.pairformer section (nested)

        Raises:
            ValueError: If required configuration sections are missing
        """
        super().__init__()

        # Try both configuration structures
        pairformer_cfg = None
        try:
            # Try root level first
            pairformer_cfg = cfg.stageB_pairformer
            logger.info("Using root level stageB_pairformer configuration")
        except AttributeError:
            if hasattr(cfg, "model") and hasattr(cfg.model, "stageB") and hasattr(cfg.model.stageB, "pairformer"):
                pairformer_cfg = cfg.model.stageB.pairformer
                logger.info("Using nested model.stageB.pairformer configuration")
            else:
                raise ValueError("Pairformer config not found in Hydra config")

        # Emit a debug/info log for test detection if debug_logging is enabled
        debug_logging = False
        # Try to get debug_logging from either config structure
        if hasattr(cfg, "debug_logging"):
            debug_logging = cfg.debug_logging
        elif hasattr(cfg, "model") and hasattr(cfg.model, "stageB") and hasattr(cfg.model.stageB, "pairformer") and hasattr(cfg.model.stageB.pairformer, "debug_logging"):
            debug_logging = cfg.model.stageB.pairformer.debug_logging
        elif hasattr(cfg, "stageB_pairformer") and hasattr(cfg.stageB_pairformer, "debug_logging"):
            debug_logging = cfg.stageB_pairformer.debug_logging
        if debug_logging:
            logger.debug("[UNIQUE-DEBUG-STAGEB-PAIRFORMER-TEST] PairformerWrapper initialized with debug_logging=True")

        # Extract configuration
        self.device = getattr(cfg, "device", None)  # Get global device first
        if self.device is None:  # Fallback to pairformer_cfg device if global not set
            self.device = getattr(
                pairformer_cfg, "device", "cuda" if torch.cuda.is_available() else "cpu"
            )
        # Ensure device is torch.device object
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

        # Get other configuration values with defaults
        self.init_z_from_adjacency = getattr(pairformer_cfg, "init_z_from_adjacency", True)
        self.model_name = getattr(pairformer_cfg, "model_name", "default")
        self.checkpoint_path = getattr(pairformer_cfg, "checkpoint_path", None)

        logger.info(f"Initializing Pairformer wrapper with device: {self.device}")
        logger.info(f"Model name: {self.model_name}")
        logger.info(f"Checkpoint path: {self.checkpoint_path}")
        logger.info(f"Init z from adjacency: {self.init_z_from_adjacency}")

        # Validate required parameters
        required_params = ["n_blocks", "c_z", "c_s", "dropout", "use_memory_efficient_kernel",
                          "use_deepspeed_evo_attention", "use_lma", "inplace_safe", "chunk_size"]
        for param in required_params:
            if not hasattr(pairformer_cfg, param):
                logger.warning(f"Configuration missing parameter: {param}, using default value")

        # Store config parameters with defaults
        self.n_blocks = getattr(pairformer_cfg, "n_blocks", 48)
        self.c_z = getattr(pairformer_cfg, "c_z", 128)  # Original c_z from config
        self.c_s = getattr(pairformer_cfg, "c_s", 384)
        self.dropout = getattr(pairformer_cfg, "dropout", 0.25)

        # Store other config flags needed for the forward pass or elsewhere
        self.use_memory_efficient_kernel = getattr(pairformer_cfg, "use_memory_efficient_kernel", False)
        self.use_deepspeed_evo_attention = getattr(pairformer_cfg, "use_deepspeed_evo_attention", False)
        self.use_lma = getattr(pairformer_cfg, "use_lma", False)
        self.inplace_safe = getattr(pairformer_cfg, "inplace_safe", False)
        self.chunk_size = getattr(pairformer_cfg, "chunk_size", None)  # Will be passed in forward

        # Check if use_checkpoint exists in the config schema
        # If not, we'll default to False
        if hasattr(pairformer_cfg, "use_checkpoint"):
            self.use_checkpoint = pairformer_cfg.use_checkpoint
        else:
            self.use_checkpoint = False
            logger.warning("'use_checkpoint' not found in config, defaulting to False")

        # Ensure c_z is a multiple of 16 for AttentionPairBias compatibility
        self.c_z_adjusted = max(16, ((self.c_z + 15) // 16) * 16)

        # Create a PairformerStackConfig for the stack
        stack_cfg = PairformerStackConfig(
            n_blocks=self.n_blocks,
            n_heads=pairformer_cfg.n_heads if hasattr(pairformer_cfg, "n_heads") else 16,  # Default to 16 heads if not specified
            c_z=self.c_z_adjusted,
            c_s=self.c_s,
            dropout=self.dropout,
            blocks_per_ckpt=1 if self.use_checkpoint else None
        )

        # Instantiate the underlying PairformerStack with parameters from config
        self.stack = PairformerStack(stack_cfg)

        # Optional: Apply LoRA if enabled in config
        if hasattr(pairformer_cfg, "lora") and hasattr(pairformer_cfg.lora, "enabled") and pairformer_cfg.lora.enabled:
            lora_cfg = pairformer_cfg.lora
            logger.info(f"LoRA enabled for Pairformer (r={lora_cfg.r}). Applying LoRA layers...")
            # Placeholder: Add your apply_lora function call here for Pairformer
            pass

    def forward(self, s: torch.Tensor, z: torch.Tensor, pair_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the Pairformer stack.

        Args:
            s: Single representation tensor [batch, N, c_s]
            z: Pair representation tensor [batch, N, N, c_z]
            pair_mask: Mask for pair attention [batch, N, N]

        Returns:
            tuple: (s_updated, z_updated) - Updated single and pair representations
        """
        # If c_z_adjusted != c_z, need to adapt the input z tensor
        if self.c_z_adjusted != self.c_z:
            # Pad or project z to match c_z_adjusted
            if self.c_z_adjusted > self.c_z:
                # Pad with zeros
                padding = torch.zeros(
                    *z.shape[:-1],
                    self.c_z_adjusted - self.c_z,
                    device=z.device,
                    dtype=z.dtype,
                )
                z_adjusted = torch.cat([z, padding], dim=-1)
            else:
                # This case shouldn't happen with our adjustment logic, but for completeness
                z_adjusted = z[..., : self.c_z_adjusted]
        else:
            z_adjusted = z

        # Pass relevant flags from config to the forward call of the stack
        # All these parameters come directly from the configuration
        s_updated, z_updated = self.stack(
            s,
            z_adjusted,
            pair_mask,
            use_memory_efficient_kernel=self.use_memory_efficient_kernel,
            use_deepspeed_evo_attention=self.use_deepspeed_evo_attention,
            use_lma=self.use_lma,
            inplace_safe=self.inplace_safe,
            chunk_size=self.chunk_size
        )

        # If we adjusted c_z, adjust the output accordingly
        if self.c_z_adjusted != self.c_z:
            z_updated = z_updated[..., : self.c_z]

        return s_updated, z_updated

    def adjust_z_dimensions(self, z: torch.Tensor) -> torch.Tensor:
        """
        Adjust the dimensions of the pair representation tensor to match c_z_adjusted.

        This is a utility method that can be called externally to ensure tensor dimensions
        are compatible with the Pairformer's requirements.

        Args:
            z: Pair representation tensor [batch, N, N, c_z]

        Returns:
            torch.Tensor: Adjusted pair representation tensor [batch, N, N, c_z_adjusted]
        """
        if self.c_z_adjusted > self.c_z:
            # Pad with zeros
            padding = torch.zeros(
                *z.shape[:-1],
                self.c_z_adjusted - self.c_z,
                device=z.device,  # Add device to ensure tensor is on the same device
                dtype=z.dtype,
            )
            z_adjusted = torch.cat([z, padding], dim=-1)
        else:
            # This case shouldn't happen with our adjustment logic, but for completeness
            z_adjusted = z[..., :self.c_z_adjusted]
        return z_adjusted

    def _initialize_model(self):
        """Initialize the Pairformer model."""
        # Placeholder for model initialization
        # This would be replaced with actual model loading code
        pass

    def predict(
        self, sequence: str, adjacency: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict RNA structure using the Pairformer model.

        Args:
            sequence: RNA sequence string
            adjacency: Optional adjacency matrix tensor

        Returns:
            Tuple of (single_embeddings, pair_embeddings)
            - single_embeddings: [L, dim_s] tensor of single-residue embeddings
            - pair_embeddings: [L, L, dim_z] tensor of pair embeddings
        """
        # For now, return dummy tensors
        L = len(sequence)
        s_emb = torch.randn(L, 384, device=self.device)  # Example dimension
        z_emb = torch.randn(L, L, 128, device=self.device)  # Example dimension
        return s_emb, z_emb
