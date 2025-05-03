"""
Atom attention encoder module.
"""

import torch
import torch.nn as nn
import logging

from rna_predict.pipeline.stageA.input_embedding.current.primitives import LinearNoBias
from rna_predict.pipeline.stageA.input_embedding.current.transformer.atom_attention.components import (
    AttentionComponents,
    CoordinateProcessor,
    FeatureProcessor,
)
from rna_predict.pipeline.stageA.input_embedding.current.transformer.atom_attention.config import (
    AtomAttentionConfig,
    EncoderForwardParams,
)

logger = logging.getLogger("rna_predict.pipeline.stageA.input_embedding.current.transformer.atom_attention.encoder")


class AtomAttentionEncoder(nn.Module):
    """
    Encoder that processes atom-level features and produces token-level embeddings.
    Implements Algorithm 5 in AlphaFold3.
    """

    def __init__(self, config: AtomAttentionConfig) -> None:
        """
        Initialize the AtomAttentionEncoder with a configuration object.

        Args:
            config: Configuration parameters for the encoder
        """
        super().__init__()
        self.has_coords = config.has_coords
        self.c_atom = config.c_atom
        self.c_atompair = config.c_atompair
        self.c_token = config.c_token
        self.c_s = config.c_s
        self.c_z = config.c_z
        self.n_queries = config.n_queries
        self.n_keys = config.n_keys

        logger.debug("TEST: AtomAttentionEncoder constructed")
        logger.debug(f"[AtomAttentionEncoder] __init__ debug_logging={getattr(config, 'debug_logging', None)}")

        # Initialize components
        c_ref_element = getattr(config, 'c_ref_element', 128)
        logger.debug(f"[DEBUG][AtomAttentionEncoder] Using c_ref_element={c_ref_element}")
        logger.debug(f"[DEBUG][AtomAttentionEncoder] debug_logging in config: {getattr(config, 'debug_logging', None)}")
        debug_logging = getattr(config, 'debug_logging', None)
        if debug_logging is None:
            debug_logging = False
        self.feature_processor = FeatureProcessor(
            c_atom=self.c_atom,
            c_atompair=self.c_atompair,
            c_s=self.c_s,
            c_z=self.c_z,
            c_ref_element=c_ref_element,
            debug_logging=bool(debug_logging),
        )

        if self.has_coords:
            self.coordinate_processor = CoordinateProcessor(
                c_atom=self.c_atom,
                c_atompair=self.c_atompair,
                c_s=self.c_s,
                c_z=self.c_z,
            )

        self.attention_components = AttentionComponents(
            c_atom=self.c_atom,
            c_atompair=self.c_atompair,
            n_blocks=config.n_blocks,
            n_heads=config.n_heads,
            n_queries=self.n_queries,
            n_keys=self.n_keys,
            blocks_per_ckpt=config.blocks_per_ckpt or 0,  # Default to 0 if None
        )

        # Output projection to token dimension
        self.linear_no_bias_q = LinearNoBias(
            in_features=self.c_atom, out_features=self.c_token
        )

    def linear_init(
        self,
        zero_init_atom_encoder_residual_linear: bool = False,
        he_normal_init_atom_encoder_small_mlp: bool = False,
        he_normal_init_atom_encoder_output: bool = False,
    ) -> None:
        """
        Initialize the parameters of the module.

        Args:
            zero_init_atom_encoder_residual_linear: Whether to zero-initialize residual linear layers
            he_normal_init_atom_encoder_small_mlp: Whether to initialize MLP with He normal initialization
            he_normal_init_atom_encoder_output: Whether to initialize output with He normal initialization
        """
        if zero_init_atom_encoder_residual_linear:
            nn.init.zeros_(self.attention_components.linear_no_bias_cl.weight)
            nn.init.zeros_(self.attention_components.linear_no_bias_cm.weight)

        if he_normal_init_atom_encoder_small_mlp:
            for layer in self.attention_components.small_mlp:
                if isinstance(
                    layer, nn.Linear
                ):  # Changed from LinearNoBias to nn.Linear
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

        if he_normal_init_atom_encoder_output:
            nn.init.kaiming_normal_(self.linear_no_bias_q.weight, nonlinearity="relu")

    def forward(
        self,
        params: EncoderForwardParams,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Forward pass of the encoder.

        Args:
            params: Forward pass parameters

        Returns:
            Tuple of (token embeddings, pair embeddings, style embeddings, coordinate embeddings)
        """
        # Extract features
        c_l = self.feature_processor.extract_atom_features(params.input_feature_dict)

        # Create pair features
        p_l = self.feature_processor.create_pair_embedding(params.input_feature_dict)

        # Create attention mask if needed
        if "ref_mask" in params.input_feature_dict:
            mask = params.input_feature_dict["ref_mask"]
        else:
            mask = torch.ones_like(c_l[..., 0], dtype=torch.bool)

        # Ensure mask has correct shape for attention
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)
        if mask.shape[-1] == 1:
            mask = mask.expand(-1, -1, self.c_atompair)

        # Apply transformer
        a_atom = self.attention_components.apply_transformer(
            c_l, p_l, mask=mask, chunk_size=params.chunk_size or 0
        )

        # Get number of tokens, ensuring it's an integer
        num_tokens_obj = params.input_feature_dict.get("num_tokens", None)
        if num_tokens_obj is not None:
            # Check type before casting
            if isinstance(num_tokens_obj, (int, float)):
                num_tokens = int(num_tokens_obj)
            else:
                # Handle unexpected type - raise error or log warning
                raise TypeError(f"Expected 'num_tokens' to be int or float, but got {type(num_tokens_obj)}")
        else:
            # Infer from other sources
            restype = params.input_feature_dict.get("restype", None)
            if restype is not None:
                # Ensure restype.shape[1] is convertible
                try:
                    num_tokens = int(restype.shape[1])
                except (TypeError, IndexError) as e:
                     raise TypeError(f"Could not determine num_tokens from restype shape: {e}")

            else:
                # Default to maximum token index + 1
                atom_to_token_idx = params.input_feature_dict.get("atom_to_token_idx", None)
                if atom_to_token_idx is None:
                    raise ValueError("Cannot determine num_tokens: 'num_tokens', 'restype', and 'atom_to_token_idx' are all missing.")
                try:
                    num_tokens = int(atom_to_token_idx.max().item() + 1)
                except Exception as e:
                    raise TypeError(f"Could not determine num_tokens from atom_to_token_idx: {e}")

        # num_tokens is now guaranteed to be an int

        # Aggregate to token level
        a_token = self.feature_processor.aggregate_to_token_level(
            a_atom,
            params.input_feature_dict["atom_to_token_idx"],
            num_tokens,  # Pass the guaranteed int
        )

        # Project to token dimension
        a_token = self.linear_no_bias_q(a_token)

        # Process coordinate-dependent features if available
        if self.has_coords and params.r_l is not None:
            # Process coordinate encoding
            a_atom = self.coordinate_processor.process_coordinate_encoding(
                a_atom, params.r_l, params.input_feature_dict["ref_pos"]
            )

            # Process style embedding if available
            if params.s is not None:
                a_atom = self.coordinate_processor.process_style_embedding(
                    a_atom,
                    params.s,
                    params.input_feature_dict["atom_to_token_idx"],
                )

            # Process pair embedding if available
            if params.z is not None:
                p_l = self.coordinate_processor.process_pair_embedding(p_l, params.z)

        return a_token, p_l, a_atom, params.r_l if self.has_coords else None

    @classmethod
    def from_args(
        cls,
        has_coords: bool,
        c_token: int,
        c_atom: int = 128,
        c_atompair: int = 16,
        c_s: int = 384,
        c_z: int = 128,
        n_blocks: int = 3,
        n_heads: int = 4,
        n_queries: int = 32,
        n_keys: int = 128,
        blocks_per_ckpt: int | None = None,
        debug_logging: bool = False,
    ) -> "AtomAttentionEncoder":
        """
        Create an AtomAttentionEncoder instance from arguments.
        All arguments should be set via Hydra config for best practices.
        """
        config = AtomAttentionConfig(
            has_coords=has_coords,
            c_token=c_token,
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_s=c_s,
            c_z=c_z,
            n_blocks=n_blocks,
            n_heads=n_heads,
            n_queries=n_queries,
            n_keys=n_keys,
            blocks_per_ckpt=blocks_per_ckpt,
        )
        return cls(config)  # If you want to use debug_logging, pass it separately here
