"""
Atom attention encoder and decoder modules for RNA structure prediction.
"""

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from rna_predict.utils.scatter_utils import scatter_mean

from rna_predict.pipeline.stageA.input_embedding.current.primitives import (
    LayerNorm,
    LinearNoBias,
)
from rna_predict.pipeline.stageA.input_embedding.current.transformer.atom_transformer import (
    AtomTransformer,
)
from rna_predict.pipeline.stageA.input_embedding.current.transformer.common import (
    InputFeatureDict,
    safe_tensor_access,
)
from rna_predict.pipeline.stageA.input_embedding.current.utils import (
    aggregate_atom_to_token,
    broadcast_token_to_atom,
)


@dataclass
class AtomAttentionConfig:
    """Configuration parameters for atom attention modules."""

    has_coords: bool
    c_token: int  # token embedding dim (384 or 768)
    c_atom: int = 128  # atom embedding dim
    c_atompair: int = 16  # atom pair embedding dim
    c_s: int = 384  # single embedding dim
    c_z: int = 128  # pair embedding dim
    n_blocks: int = 3
    n_heads: int = 4
    n_queries: int = 32
    n_keys: int = 128
    blocks_per_ckpt: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate config parameters."""
        if self.c_atom <= 0:
            raise ValueError(f"c_atom must be positive, got {self.c_atom}")
        if self.c_token <= 0:
            raise ValueError(f"c_token must be positive, got {self.c_token}")
        if self.n_blocks <= 0:
            raise ValueError(f"n_blocks must be positive, got {self.n_blocks}")


@dataclass
class EncoderForwardParams:
    """Parameters for AtomAttentionEncoder forward method."""

    input_feature_dict: InputFeatureDict
    r_l: Optional[torch.Tensor] = None
    s: Optional[torch.Tensor] = None
    z: Optional[torch.Tensor] = None
    inplace_safe: bool = False
    chunk_size: Optional[int] = None


@dataclass
class DecoderForwardParams:
    """Parameters for AtomAttentionDecoder forward method."""

    a: torch.Tensor
    r_l: torch.Tensor
    extra_feats: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    atom_mask: Optional[torch.Tensor] = None
    atom_to_token_idx: Optional[torch.Tensor] = None
    chunk_size: Optional[int] = None


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
        super(AtomAttentionEncoder, self).__init__()
        self.has_coords = config.has_coords
        self.c_atom = config.c_atom
        self.c_atompair = config.c_atompair
        self.c_token = config.c_token
        self.c_s = config.c_s
        self.c_z = config.c_z
        self.n_queries = config.n_queries
        self.n_keys = config.n_keys
        self.local_attention_method = "local_cross_attention"

        # Set up component configurations
        self._setup_feature_dimensions()
        self._setup_atom_encoders()
        self._setup_distance_encoders()

        if self.has_coords:
            self._setup_coordinate_components()

        self._setup_pair_projections()
        self._setup_small_mlp()

        # Atom transformer for atom-level processing
        self.atom_transformer = self._create_atom_transformer(
            config.n_blocks, config.n_heads, config.blocks_per_ckpt
        )

        # Output projection to token dimension
        self.linear_no_bias_q = LinearNoBias(
            in_features=self.c_atom, out_features=self.c_token
        )

    def _setup_feature_dimensions(self) -> None:
        """Define expected feature dimensions."""
        self.input_feature = {
            "ref_pos": 3,
            "ref_charge": 1,
            "ref_mask": 1,
            "ref_element": 128,
            "ref_atom_name_chars": 4 * 64,
        }

    def _setup_atom_encoders(self) -> None:
        """Set up encoders for atom features."""
        self.linear_no_bias_f = LinearNoBias(
            in_features=sum(self.input_feature.values()), out_features=self.c_atom
        )

    def _setup_distance_encoders(self) -> None:
        """Set up encoders for distance-related features."""
        self.linear_no_bias_d = LinearNoBias(
            in_features=3, out_features=self.c_atompair
        )
        self.linear_no_bias_invd = LinearNoBias(
            in_features=1, out_features=self.c_atompair
        )
        self.linear_no_bias_v = LinearNoBias(
            in_features=1, out_features=self.c_atompair
        )

    def _setup_coordinate_components(self) -> None:
        """Set up components used when coordinates are available."""
        # Style normalization and projection
        self.layernorm_s = LayerNorm(self.c_s)
        self.linear_no_bias_s = LinearNoBias(
            in_features=self.c_s, out_features=self.c_atom
        )

        # Pair embedding normalization and projection
        self.layernorm_z = LayerNorm(self.c_z)  # memory bottleneck
        self.linear_no_bias_z = LinearNoBias(
            in_features=self.c_z, out_features=self.c_atompair
        )

        # Position encoder
        self.linear_no_bias_r = LinearNoBias(in_features=3, out_features=self.c_atom)

    def _setup_pair_projections(self) -> None:
        """Set up linear projections for atom features to pair dimension."""
        self.linear_no_bias_cl = LinearNoBias(
            in_features=self.c_atom, out_features=self.c_atompair
        )
        self.linear_no_bias_cm = LinearNoBias(
            in_features=self.c_atom, out_features=self.c_atompair
        )

    def _setup_small_mlp(self) -> None:
        """Set up small MLP for pair feature processing."""
        self.small_mlp = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(in_features=self.c_atompair, out_features=self.c_atompair),
            nn.ReLU(),
            LinearNoBias(in_features=self.c_atompair, out_features=self.c_atompair),
            nn.ReLU(),
            LinearNoBias(in_features=self.c_atompair, out_features=self.c_atompair),
        )

    def _create_atom_transformer(
        self, n_blocks: int, n_heads: int, blocks_per_ckpt: Optional[int]
    ) -> AtomTransformer:
        """
        Create the AtomTransformer instance.

        Args:
            n_blocks: Number of blocks in transformer
            n_heads: Number of attention heads
            blocks_per_ckpt: Number of blocks per checkpoint

        Returns:
            Configured AtomTransformer instance
        """
        return AtomTransformer(
            n_blocks=n_blocks,
            n_heads=n_heads,
            c_atom=self.c_atom,
            c_atompair=self.c_atompair,
            n_queries=self.n_queries,
            n_keys=self.n_keys,
            blocks_per_ckpt=blocks_per_ckpt,
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
        self._init_residual_layers(zero_init_atom_encoder_residual_linear)
        self._init_mlp_layers(he_normal_init_atom_encoder_small_mlp)
        self._init_output_layer(he_normal_init_atom_encoder_output)

    def _init_residual_layers(self, zero_init: bool) -> None:
        """
        Initialize residual connection layers.

        Args:
            zero_init: Whether to zero-initialize the weights
        """
        if not zero_init:
            return

        # Always initialize these layers
        nn.init.zeros_(self.linear_no_bias_invd.weight)
        nn.init.zeros_(self.linear_no_bias_v.weight)
        nn.init.zeros_(self.linear_no_bias_cl.weight)
        nn.init.zeros_(self.linear_no_bias_cm.weight)

        # Initialize coordinate-dependent layers if needed
        if self.has_coords:
            nn.init.zeros_(self.linear_no_bias_s.weight)
            nn.init.zeros_(self.linear_no_bias_z.weight)
            nn.init.zeros_(self.linear_no_bias_r.weight)

    def _init_mlp_layers(self, use_he_normal: bool) -> None:
        """
        Initialize MLP layers with He normal initialization.

        Args:
            use_he_normal: Whether to use He normal initialization
        """
        if not use_he_normal:
            return

        for layer in self.small_mlp:
            if not isinstance(layer, torch.nn.modules.activation.ReLU):
                nn.init.kaiming_normal_(
                    layer.weight,
                    a=0,
                    mode="fan_in",
                    nonlinearity="relu",
                )

    def _init_output_layer(self, use_he_normal: bool) -> None:
        """
        Initialize output layer with He normal initialization.

        Args:
            use_he_normal: Whether to use He normal initialization
        """
        if not use_he_normal:
            return

        nn.init.kaiming_normal_(
            self.linear_no_bias_q.weight, a=0, mode="fan_in", nonlinearity="relu"
        )

    def _process_input_features(self, input_feature_dict: InputFeatureDict) -> None:
        """Process and validate input feature dimensions."""
        # Handle ref_space_uid dimension
        if "ref_space_uid" in input_feature_dict:
            ref_space_uid = input_feature_dict["ref_space_uid"]
            if ref_space_uid.dim() == 2:  # [B, N_atom]
                input_feature_dict["ref_space_uid"] = ref_space_uid.unsqueeze(-1)  # [B, N_atom, 1]

        # Handle atom_to_token_idx dimension
        if "atom_to_token_idx" in input_feature_dict:
            atom_to_token_idx = input_feature_dict["atom_to_token_idx"]
            if atom_to_token_idx.dim() == 1:  # [N_atom]
                input_feature_dict["atom_to_token_idx"] = atom_to_token_idx.unsqueeze(0)  # [1, N_atom]

    def _process_simple_embedding(
        self, input_feature_dict: InputFeatureDict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """
        Process simple embedding without coordinates.

        Args:
            input_feature_dict: Dictionary of input features

        Returns:
            Tuple containing:
                - Token-level embedding
                - Atom-level embedding (q_l)
                - Atom-level embedding (c_l)
                - None (no pair embedding in simple case)
        """
        atom_to_token_idx = safe_tensor_access(input_feature_dict, "atom_to_token_idx")
        c_l = self.extract_atom_features(input_feature_dict)
        q_l = c_l  # No position embedding in simple case

        # Map atom embeddings to token dimension
        a_atom = torch.relu(self.linear_no_bias_q(q_l))

        # Determine number of tokens from atom_to_token_idx
        if atom_to_token_idx.dim() == 2:
            # shape e.g. [B, N_atom]
            max_idx = int(atom_to_token_idx.max().item())
            n_token = max_idx + 1
        else:
            # shape e.g. [N_atom], simpler
            max_idx = int(atom_to_token_idx.max().item())
            n_token = max_idx + 1

        # Cross-check with restype if available
        if "restype" in input_feature_dict:
            restype = safe_tensor_access(input_feature_dict, "restype")
            n_restype = restype.shape[-2]

            # Adjust token count if needed to match restype
            if n_restype < n_token:
                warnings.warn(
                    f"[AtomAttentionEncoder] restype tokens={n_restype} < aggregator tokens={n_token}. "
                    f"This mismatch may be unintended."
                )
                # Fix: Adjust n_token to match n_restype to prevent out-of-bounds
                n_token = n_restype

            elif n_restype > n_token:
                warnings.warn(
                    f"[AtomAttentionEncoder] restype tokens={n_restype} > aggregator tokens={n_token}. "
                    f"We'll produce only {n_token} aggregator tokens, ignoring the extras."
                )

            # Ensure atom_to_token_idx doesn't contain values >= n_token
            if atom_to_token_idx.max() >= n_token:
                warnings.warn(
                    f"[AtomAttentionEncoder] atom_to_token_idx contains indices >= {n_token}. "
                    f"Clipping indices to {n_token - 1}."
                )
                atom_to_token_idx = torch.clamp(atom_to_token_idx, max=n_token - 1)

        # Aggregate atom features to token level
        aggregated = aggregate_atom_to_token(
            x_atom=a_atom,
            atom_to_token_idx=atom_to_token_idx,
            n_token=n_token,
            reduce="mean",
        )

        return aggregated, q_l, c_l, None

    def extract_atom_features(
        self, input_feature_dict: InputFeatureDict
    ) -> torch.Tensor:
        """
        Extract and combine atom features from input dictionary.

        Args:
            input_feature_dict: Dictionary of input features

        Returns:
            Combined atom features as tensor
        """
        ref_pos = safe_tensor_access(input_feature_dict, "ref_pos")
        batch_shape = ref_pos.shape[:-2]
        N_atom = ref_pos.shape[-2]

        # Collect all available features to concatenate
        features_to_concat = []
        for name in self.input_feature:
            if name in input_feature_dict:
                feature = safe_tensor_access(input_feature_dict, name)
                # Reshape to expected dimensions
                reshaped_feature = feature.reshape(
                    *batch_shape, N_atom, self.input_feature[name]
                )
                features_to_concat.append(reshaped_feature)

        # Project concatenated features to atom embedding dimension
        c_l = self.linear_no_bias_f(
            torch.cat(features_to_concat, dim=-1)
        )  # => [..., N_atom, c_atom]

        return c_l

    def ensure_space_uid(self, input_feature_dict: InputFeatureDict) -> None:
        """
        Ensure ref_space_uid exists and has correct shape.

        Args:
            input_feature_dict: Dictionary of input features to check/modify
        """
        ref_pos = safe_tensor_access(input_feature_dict, "ref_pos")
        batch_shape = ref_pos.shape[:-2]
        N_atom = ref_pos.shape[-2]

        if "ref_space_uid" in input_feature_dict:
            uid = safe_tensor_access(input_feature_dict, "ref_space_uid")
            if uid.ndim == 2:  # shape was [B, N_atom]
                uid = uid.unsqueeze(-1)  # becomes [B, N_atom, 1]
                input_feature_dict["ref_space_uid"] = uid
        else:
            # Create ref_space_uid if it doesn't exist
            input_feature_dict["ref_space_uid"] = torch.zeros(
                *batch_shape, N_atom, 1, device=ref_pos.device
            )

    def create_pair_embedding(
        self, input_feature_dict: InputFeatureDict
    ) -> torch.Tensor:
        """
        Create pair embedding for atom transformer.

        Args:
            input_feature_dict: Dictionary of input features

        Returns:
            Pair embedding tensor
        """
        ref_pos = safe_tensor_access(input_feature_dict, "ref_pos")
        batch_shape = ref_pos.shape[:-2]

        # Create trunk-based pair embedding
        # Shape: [batch_size, n_blocks (1), n_queries, n_keys, c_atompair]
        p_lm = torch.zeros(
            *batch_shape,
            1,
            self.n_queries,
            self.n_keys,
            self.c_atompair,
            device=ref_pos.device,
        )

        # Apply small MLP to initialize with non-zero values
        p_lm = self.small_mlp(p_lm)

        return p_lm

    def _process_coordinate_encoding(
        self, q_l: torch.Tensor, r_l: Optional[torch.Tensor], ref_pos: torch.Tensor
    ) -> torch.Tensor:
        """
        Process and add coordinate-based positional encoding.

        Args:
            q_l: Input atom features
            r_l: Atom coordinates, shape [..., N_atom, 3]
            ref_pos: Reference atom positions

        Returns:
            Updated atom features with positional encoding
        """
        if r_l is None:
            return q_l

        # Check coordinates shape matches expected
        if r_l.size(-1) == 3 and r_l.size(1) == ref_pos.size(1):
            return q_l + self.linear_no_bias_r(r_l)
        else:
            # Log shape mismatch and skip linear transformation
            warnings.warn(
                f"Warning: r_l shape mismatch. Expected [..., {ref_pos.size(1)}, 3], "
                f"got {r_l.shape}. Skipping linear_no_bias_r."
            )
            return q_l

    def _process_style_embedding(
        self,
        c_l: torch.Tensor,
        s: Optional[torch.Tensor],
        atom_to_token_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process style embedding from token to atom level.

        Args:
            c_l: Input atom features
            s: Token-level style embedding
            atom_to_token_idx: Mapping from atoms to tokens

        Returns:
            Updated atom features with style information
        """
        if s is None:
            return c_l

        # Broadcast token-level s to atom-level
        broadcasted_s = broadcast_token_to_atom(s, atom_to_token_idx)

        # Ensure compatible shape for layernorm
        if broadcasted_s.size(-1) != self.c_s:
            broadcasted_s = self._adapt_tensor_dimensions(broadcasted_s)

        # Apply layer norm and add to atom embedding
        return c_l + self.linear_no_bias_s(self.layernorm_s(broadcasted_s))

    def _adapt_tensor_dimensions(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Adapt tensor dimensions to match expected size for style embedding.

        Args:
            tensor: Input tensor to adapt

        Returns:
            Adapted tensor with compatible dimensions
        """
        if tensor.size(-1) == 1:
            # Expand singleton dimension
            return tensor.expand(*tensor.shape[:-1], self.c_s)
        else:
            # Create compatible tensor with padding
            compatible_tensor = torch.zeros(
                *tensor.shape[:-1], self.c_s, device=tensor.device
            )
            # Copy values where dimensions overlap
            compatible_tensor[..., : min(tensor.size(-1), self.c_s)] = tensor[
                ..., : min(tensor.size(-1), self.c_s)
            ]
            return compatible_tensor

    def _aggregate_to_token_level(
        self, a_atom: torch.Tensor, atom_to_token_idx: torch.Tensor, num_tokens: int
    ) -> torch.Tensor:
        """
        Aggregate atom-level features to token-level.

        Args:
            a_atom: Atom-level features
            atom_to_token_idx: Mapping from atoms to tokens
            num_tokens: Number of tokens

        Returns:
            Token-level aggregated features
        """
        # Ensure atom_to_token_idx doesn't exceed num_tokens to prevent out-of-bounds
        if atom_to_token_idx.max() >= num_tokens:
            warnings.warn(
                f"[AtomAttentionEncoder] atom_to_token_idx contains indices >= {num_tokens}. "
                f"Clipping indices to prevent out-of-bounds error."
            )
            atom_to_token_idx = torch.clamp(atom_to_token_idx, max=num_tokens - 1)

        # Aggregate atom features to token level using scatter_mean
        a_token = torch.zeros(
            (*a_atom.shape[:-2], num_tokens, a_atom.shape[-1]),
            device=a_atom.device,
            dtype=a_atom.dtype
        )
        
        # Handle batched inputs
        if atom_to_token_idx.dim() == 2:  # [B, N_atom]
            batch_size = atom_to_token_idx.size(0)
            for b in range(batch_size):
                a_token[b] = scatter_mean(
                    a_atom[b],
                    atom_to_token_idx[b],
                    dim=0,
                    dim_size=num_tokens
                )
        else:  # [N_atom]
            a_token = scatter_mean(
                a_atom,
                atom_to_token_idx,
                dim=0,
                dim_size=num_tokens
            )

        return a_token

    def process_inputs_with_coords(
        self,
        input_feature_dict: InputFeatureDict,
        r_l: Optional[torch.Tensor],
        s: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
        c_l: torch.Tensor,
        chunk_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process inputs when coordinates are available.

        Args:
            input_feature_dict: Dictionary of input features
            r_l: Atom coordinates
            s: Single embedding
            z: Pair embedding
            c_l: Atom features
            chunk_size: Chunk size for memory-efficient processing

        Returns:
            Tuple containing:
                - Token-level embedding
                - Atom-level embedding (q_l)
                - Atom-level embedding (c_l)
                - Pair embedding (p_lm)
        """
        # Ensure ref_space_uid exists and has correct shape
        self.ensure_space_uid(input_feature_dict)

        # Create pair embedding for atom transformer
        p_lm = self.create_pair_embedding(input_feature_dict)

        # Get required tensors
        atom_to_token_idx = safe_tensor_access(input_feature_dict, "atom_to_token_idx")
        ref_pos = safe_tensor_access(input_feature_dict, "ref_pos")

        # Process coordinates and style embedding
        q_l = self._process_coordinate_encoding(c_l, r_l, ref_pos)
        c_l = self._process_style_embedding(c_l, s, atom_to_token_idx)

        # Process through atom transformer
        q_l = self.atom_transformer(q_l, c_l, p_lm, chunk_size=chunk_size)

        # Project to token dimension with ReLU
        a_atom = F.relu(self.linear_no_bias_q(q_l))

        # Get token count and aggregate to token level
        restype = safe_tensor_access(input_feature_dict, "restype")
        num_tokens = restype.shape[-2]
        a = self._aggregate_to_token_level(a_atom, atom_to_token_idx, num_tokens)

        return a, q_l, c_l, p_lm

    def forward(
        self,
        input_feature_dict: InputFeatureDict,
        chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the encoder.

        Args:
            input_feature_dict: Dictionary of input features
            chunk_size: Optional chunk size for processing

        Returns:
            Tuple of (token embeddings, atom embeddings, initial atom embeddings, pair embeddings)
        """
        # Process input feature dimensions
        self._process_input_features(input_feature_dict)

        # Extract features
        c_l = self.extract_atom_features(input_feature_dict)

        # Create pair features
        p_l = self.create_pair_embedding(input_feature_dict)

        # Create attention mask if needed
        if "ref_mask" in input_feature_dict:
            mask = input_feature_dict["ref_mask"]
        else:
            mask = torch.ones_like(c_l[..., 0], dtype=torch.bool)

        # Ensure mask has correct shape for attention
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)
        if mask.shape[-1] == 1:
            mask = mask.expand(-1, -1, self.c_atompair)

        # Apply transformer
        a_atom = self.atom_transformer(
            c_l, p_l, mask=mask, chunk_size=chunk_size or 0
        )

        # Get number of tokens from restype if available
        if "restype" in input_feature_dict:
            n_tokens = input_feature_dict["restype"].shape[1]  # [B, N_tokens, ...]
        else:
            # Fallback to atom_to_token_idx if restype not available
            n_tokens = input_feature_dict["atom_to_token_idx"].max().item() + 1

        # Aggregate to token level
        a_token = self._aggregate_to_token_level(
            a_atom,
            input_feature_dict["atom_to_token_idx"],
            int(n_tokens),  # Explicitly cast to int
        )

        return a_token, a_atom, c_l, p_l

    # For backward compatibility - uses the old parameter style
    def forward_legacy(
        self,
        input_feature_dict: InputFeatureDict,
        r_l: Optional[torch.Tensor] = None,
        s: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Legacy forward pass with individual parameters.

        Args:
            input_feature_dict: Contains per-atom features and reference to token-level data
            r_l: Noisy position if has_coords=True, shape [..., N_atom, 3]
            s: Single embedding if has_coords=True, shape [..., N_token, c_s]
            z: Pair embedding if has_coords=True, shape [..., N_token, N_token, c_z]
            inplace_safe: Whether to do some ops in-place for memory efficiency
            chunk_size: If set, break computations into smaller chunks to reduce memory usage

        Returns:
            Tuple containing:
                - Final token-level embedding, shape [..., N_token, c_token]
                - Atom-level embedding, shape [..., N_atom, c_atom]
                - Another atom-level embedding, shape [..., N_atom, c_atom]
                - The trunk-based pair embedding or None if trunk is skipped
        """
        params = EncoderForwardParams(
            input_feature_dict=input_feature_dict,
            r_l=r_l,
            s=s,
            z=z,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        return self.forward(params)

    # For backward compatibility - creates a config from args
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
        blocks_per_ckpt: Optional[int] = None,
    ) -> "AtomAttentionEncoder":
        """
        Create AtomAttentionEncoder from individual arguments.

        Args:
            has_coords: Whether the module input will contain coordinates
            c_token: Token embedding dimension
            c_atom: Atom embedding dimension
            c_atompair: Atom pair embedding dimension
            c_s: Single embedding dimension
            c_z: Pair embedding dimension
            n_blocks: Number of blocks in AtomTransformer
            n_heads: Number of heads in AtomTransformer
            n_queries: Number of queries for local attention
            n_keys: Number of keys for local attention
            blocks_per_ckpt: Number of blocks per checkpoint for memory efficiency

        Returns:
            Initialized AtomAttentionEncoder
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
        return cls(config)


class AtomAttentionDecoder(nn.Module):
    """
    Decoder that processes token-level embeddings and produces atom-level coordinates.
    Implements Algorithm 6 in AlphaFold3.
    """

    def __init__(self, config: AtomAttentionConfig) -> None:
        """
        Initialize the AtomAttentionDecoder with a configuration object.

        Args:
            config: Configuration parameters for the decoder
        """
        super(AtomAttentionDecoder, self).__init__()
        self.n_blocks = config.n_blocks
        self.n_heads = config.n_heads
        self.c_token = config.c_token
        self.c_atom = config.c_atom
        self.c_atompair = config.c_atompair
        self.n_queries = config.n_queries
        self.n_keys = config.n_keys

        # Project token features to atom dimension
        self.linear_no_bias_a = LinearNoBias(
            in_features=config.c_token, out_features=config.c_atom
        )

        # Layer normalization and output projection
        self.layernorm_q = LayerNorm(config.c_atom)
        self.linear_no_bias_out = LinearNoBias(
            in_features=config.c_atom, out_features=3
        )

        # Atom transformer for processing
        self.atom_transformer = AtomTransformer(
            n_blocks=config.n_blocks,
            n_heads=config.n_heads,
            c_atom=config.c_atom,
            c_atompair=config.c_atompair,
            n_queries=config.n_queries,
            n_keys=config.n_keys,
            blocks_per_ckpt=config.blocks_per_ckpt,
        )

    def forward(
        self,
        params: DecoderForwardParams,
    ) -> torch.Tensor:
        """
        Forward pass for the AtomAttentionDecoder.

        Args:
            params: Parameters for the forward pass

        Returns:
            Predicted atom coordinates, shape [..., N_atom, 3]
        """
        # Project token features to atom dimension
        q = self.linear_no_bias_a(params.a)

        # Broadcast from token to atom if needed
        if params.atom_to_token_idx is not None:
            # Broadcast token features to atoms
            q = broadcast_token_to_atom(q, params.atom_to_token_idx)

        # Include extra features if provided
        if params.extra_feats is not None:
            q = q + params.extra_feats

        # Apply atom mask if provided
        if params.atom_mask is not None:
            q = q * params.atom_mask.unsqueeze(-1)

        # Process through atom transformer
        q = self.atom_transformer(
            q=q, p=params.r_l, a=None, chunk_size=params.chunk_size
        )

        # Apply token mask if provided
        if params.mask is not None:
            q = q * params.mask.unsqueeze(-1)

        # Project to 3D coordinates
        r = self.linear_no_bias_out(self.layernorm_q(q))
        return cast(torch.Tensor, r)

    # For backward compatibility - uses the old parameter style
    def forward_legacy(
        self,
        a: torch.Tensor,
        r_l: torch.Tensor,
        extra_feats: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        atom_mask: Optional[torch.Tensor] = None,
        atom_to_token_idx: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Legacy forward pass with individual parameters.

        Args:
            a: Token embedding, shape [..., N_token, c_token]
            r_l: Initial atom positions, shape [..., N_atom, 3]
            extra_feats: Additional atom features, shape [..., N_atom, c_atom]
            mask: Token mask, shape [..., N_token]
            atom_mask: Atom mask, shape [..., N_atom]
            atom_to_token_idx: Mapping from atoms to tokens, shape [..., N_atom]
            chunk_size: If set, break computations into smaller chunks to reduce memory usage

        Returns:
            Predicted atom coordinates, shape [..., N_atom, 3]
        """
        params = DecoderForwardParams(
            a=a,
            r_l=r_l,
            extra_feats=extra_feats,
            mask=mask,
            atom_mask=atom_mask,
            atom_to_token_idx=atom_to_token_idx,
            chunk_size=chunk_size,
        )
        return self.forward(params)

    # For backward compatibility - creates a config from args
    @classmethod
    def from_args(
        cls,
        n_blocks: int = 3,
        n_heads: int = 4,
        c_token: int = 384,
        c_atom: int = 128,
        c_atompair: int = 16,
        n_queries: int = 32,
        n_keys: int = 128,
        blocks_per_ckpt: Optional[int] = None,
    ) -> "AtomAttentionDecoder":
        """
        Create AtomAttentionDecoder from individual arguments.

        Args:
            n_blocks: Number of blocks in the atom transformer
            n_heads: Number of attention heads
            c_token: Token embedding dimension
            c_atom: Atom embedding dimension
            c_atompair: Atom pair embedding dimension
            n_queries: Number of queries for local attention
            n_keys: Number of keys for local attention
            blocks_per_ckpt: Number of blocks per checkpoint for memory efficiency

        Returns:
            Initialized AtomAttentionDecoder
        """
        config = AtomAttentionConfig(
            has_coords=True,  # Decoder always deals with coordinates
            c_token=c_token,
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_s=0,  # Not used in decoder
            c_z=0,  # Not used in decoder
            n_blocks=n_blocks,
            n_heads=n_heads,
            n_queries=n_queries,
            n_keys=n_keys,
            blocks_per_ckpt=blocks_per_ckpt,
        )
        return cls(config)
