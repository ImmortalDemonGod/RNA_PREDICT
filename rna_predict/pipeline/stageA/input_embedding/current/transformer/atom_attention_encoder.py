"""
Atom attention encoder module for RNA structure prediction.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """Parameters for AtomAttentionEncoder.forward method."""
    input_feature_dict: InputFeatureDict
    r_l: Optional[torch.Tensor] = None
    s: Optional[torch.Tensor] = None
    z: Optional[torch.Tensor] = None
    inplace_safe: bool = False
    chunk_size: Optional[int] = None


@dataclass
class ProcessInputsParams:
    """Parameters for input processing methods."""
    input_feature_dict: InputFeatureDict
    r_l: Optional[torch.Tensor]
    s: Optional[torch.Tensor]
    z: Optional[torch.Tensor]
    c_l: torch.Tensor
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

    def _process_feature(
        self, input_feature_dict: InputFeatureDict, feature_name: str, expected_dim: int
    ) -> Optional[torch.Tensor]:
        """
        Process a feature from the input dictionary.

        Args:
            input_feature_dict: Dictionary containing input features
            feature_name: Name of the feature to extract
            expected_dim: Expected dimension of the feature

        Returns:
            Processed feature tensor or None if feature is invalid
        """
        if feature_name not in input_feature_dict:
            warnings.warn(f"Feature {feature_name} missing from input dictionary.")
            return None

        # Using string literal for TypedDict key access
        feature = input_feature_dict[feature_name]  # type: ignore

        # Check if shape is already correct
        if feature.dim() > 0 and feature.shape[-1] == expected_dim:
            return feature

        # Try to reshape
        try:
            return feature.view(*feature.shape[:-1], expected_dim)
        except RuntimeError:
            warnings.warn(
                f"Feature {feature_name} has wrong shape {feature.shape}, "
                f"expected last dim to be {expected_dim}. Skipping."
            )
            return None

    def extract_atom_features(
        self, input_feature_dict: InputFeatureDict
    ) -> torch.Tensor:
        """
        Extract atom features from input dictionary.

        Args:
            input_feature_dict: Dictionary containing atom features

        Returns:
            Tensor of atom features
        """
        features = []

        # Process each feature individually
        for feature_name, feature_dim in self.input_feature.items():
            processed_feature = self._process_feature(
                input_feature_dict, feature_name, feature_dim
            )
            if processed_feature is not None:
                features.append(processed_feature)

        # Check if we have any valid features
        if not features:
            raise ValueError("No valid features found in input dictionary.")

        # Concatenate features along last dimension
        cat_features = torch.cat(features, dim=-1)

        # Pass through atom encoder
        return self.linear_no_bias_f(cat_features)

    def ensure_space_uid(self, input_feature_dict: InputFeatureDict) -> None:
        """
        Ensure ref_space_uid exists and has correct shape.

        Args:
            input_feature_dict: Dictionary of input features
        """
        ref_space_uid = safe_tensor_access(input_feature_dict, "ref_space_uid")

        # Check shape
        if ref_space_uid.shape[-1] != 3:
            warnings.warn(
                f"ref_space_uid has wrong shape {ref_space_uid.shape}, expected [..., 3]. "
                f"Setting to zeros."
            )
            input_feature_dict["ref_space_uid"] = torch.zeros(
                (*ref_space_uid.shape[:-1], 3), device=ref_space_uid.device
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
        # Get reference positions and charges
        ref_pos = safe_tensor_access(input_feature_dict, "ref_pos")
        ref_charge = safe_tensor_access(input_feature_dict, "ref_charge")

        # Create all-pairs distance tensor
        n_atoms = ref_pos.shape[-2]

        # Initialize the pair embedding tensor
        p_lm = torch.zeros(
            (*ref_pos.shape[:-2], self.n_queries, self.n_keys, self.c_atompair),
            device=ref_pos.device,
        )

        # Return empty embedding if there are no atoms
        if n_atoms == 0:
            return p_lm

        # Process distance and charge information
        p_lm = self._process_distances(p_lm, ref_pos)
        p_lm = self._process_charges(p_lm, ref_charge)

        return p_lm

    def _process_distances(
        self, pair_embed: torch.Tensor, ref_pos: torch.Tensor
    ) -> torch.Tensor:
        """
        Process distance information for pair embedding.

        Args:
            pair_embed: Initial pair embedding tensor
            ref_pos: Reference positions tensor

        Returns:
            Updated pair embedding tensor with distance features
        """
        # Process distances between atom pairs
        for query_idx in range(self.n_queries):
            for key_idx in range(self.n_keys):
                # Get positions of query and key atoms
                pos_query = ref_pos[..., query_idx, :]
                pos_key = ref_pos[..., key_idx, :]

                # Calculate distance vector
                dist_vector = pos_query - pos_key

                # Apply distance encoding
                dist_features = self.linear_no_bias_d(dist_vector)

                # Add to pair embedding
                pair_embed[..., query_idx, key_idx, :] = dist_features

        return pair_embed

    def _process_charges(
        self, pair_embed: torch.Tensor, ref_charge: torch.Tensor
    ) -> torch.Tensor:
        """
        Process charge information for pair embedding.

        Args:
            pair_embed: Pair embedding tensor with distance features
            ref_charge: Reference charges tensor

        Returns:
            Updated pair embedding tensor with charge features
        """
        # Initialize charge products tensor
        charge_products = torch.zeros(
            (*ref_charge.shape[:-2], self.n_queries, self.n_keys, 1),
            device=ref_charge.device,
        )

        # Process charge products between atom pairs
        for query_idx in range(self.n_queries):
            for key_idx in range(self.n_keys):
                # Get charges of query and key atoms
                charge_query = ref_charge[..., query_idx, 0]
                charge_key = ref_charge[..., key_idx, 0]

                # Calculate charge product
                charge_product = charge_query * charge_key

                # Add to charge products
                charge_products[..., query_idx, key_idx, 0] = charge_product

        # Apply volume encoding to charge products
        volume_features = self.linear_no_bias_v(charge_products)

        # Add charge product features to pair embedding
        return pair_embed + volume_features

    def _process_simple_embedding(
        self, input_feature_dict: InputFeatureDict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """
        Process input features without coordinates.

        Args:
            input_feature_dict: Dictionary of input features

        Returns:
            Tuple containing:
                - Token-level embedding
                - Atom-level embedding (q_l)
                - Atom-level embedding (c_l)
                - None (no pair embedding)
        """
        # Extract features
        c_l = self.extract_atom_features(input_feature_dict)

        # Process through a simple projection
        q_l = c_l

        # Project to token dimension
        a_atom = F.relu(self.linear_no_bias_q(q_l))

        # Get token count from restype
        restype = safe_tensor_access(input_feature_dict, "restype")
        num_tokens = restype.shape[-2]

        # Get atom to token mapping
        atom_to_token_idx = safe_tensor_access(input_feature_dict, "atom_to_token_idx")

        # Ensure atom_to_token_idx doesn't exceed num_tokens
        if atom_to_token_idx.max() >= num_tokens:
            warnings.warn(
                f"[AtomAttentionEncoder] atom_to_token_idx contains indices >= {num_tokens}. "
                f"Clipping indices to prevent out-of-bounds error."
            )
            atom_to_token_idx = torch.clamp(atom_to_token_idx, max=num_tokens - 1)

        # Aggregate atom features to token level
        a = aggregate_atom_to_token(
            x_atom=a_atom,
            atom_to_token_idx=atom_to_token_idx,
            n_token=num_tokens,
            reduce="mean",
        )

        return a, q_l, c_l, None

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

        # Aggregate atom features to token level
        return aggregate_atom_to_token(
            x_atom=a_atom,
            atom_to_token_idx=atom_to_token_idx,
            n_token=num_tokens,
            reduce="mean",
        )

    def process_inputs_with_coords(
        self,
        params: ProcessInputsParams,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process inputs when coordinates are available.

        Args:
            params: Parameters for processing inputs with coordinates

        Returns:
            Tuple containing:
                - Token-level embedding
                - Atom-level embedding (q_l)
                - Atom-level embedding (c_l)
                - Pair embedding (p_lm)
        """
        # Ensure ref_space_uid exists and has correct shape
        self.ensure_space_uid(params.input_feature_dict)

        # Create pair embedding for atom transformer
        p_lm = self.create_pair_embedding(params.input_feature_dict)

        # Get required tensors
        atom_to_token_idx = safe_tensor_access(params.input_feature_dict, "atom_to_token_idx")
        ref_pos = safe_tensor_access(params.input_feature_dict, "ref_pos")

        # Process coordinates and style embedding
        q_l = self._process_coordinate_encoding(params.c_l, params.r_l, ref_pos)
        c_l = self._process_style_embedding(params.c_l, params.s, atom_to_token_idx)

        # Process through atom transformer
        q_l = self.atom_transformer(q_l, c_l, p_lm, chunk_size=params.chunk_size)

        # Project to token dimension with ReLU
        a_atom = F.relu(self.linear_no_bias_q(q_l))

        # Get token count and aggregate to token level
        restype = safe_tensor_access(params.input_feature_dict, "restype")
        num_tokens = restype.shape[-2]
        a = self._aggregate_to_token_level(a_atom, atom_to_token_idx, num_tokens)

        return a, q_l, c_l, p_lm

    def forward(
        self,
        params: EncoderForwardParams,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for the AtomAttentionEncoder.

        Args:
            params: Parameters for the forward pass

        Returns:
            Tuple containing:
                - Final token-level embedding, shape [..., N_token, c_token]
                - Atom-level embedding, shape [..., N_atom, c_atom]
                - Another atom-level embedding, shape [..., N_atom, c_atom]
                - The trunk-based pair embedding or None if trunk is skipped
        """
        # Simple path for no coordinates case
        if not self.has_coords:
            return self._process_simple_embedding(params.input_feature_dict)

        # Extract atom features from input dictionary
        c_l = self.extract_atom_features(params.input_feature_dict)

        # Create process inputs parameters
        process_params = ProcessInputsParams(
            input_feature_dict=params.input_feature_dict,
            r_l=params.r_l,
            s=params.s,
            z=params.z,
            c_l=c_l,
            chunk_size=params.chunk_size,
        )

        # Full processing path for coordinated case
        return self.process_inputs_with_coords(process_params)

    # For backward compatibility - uses the old parameter style
    def forward_legacy(
        self,
        input_feature_dict: InputFeatureDict,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Legacy forward pass with individual parameters.

        Args:
            input_feature_dict: Contains per-atom features and reference to token-level data
            **kwargs: Additional parameters including:
                - r_l: Noisy position if has_coords=True, shape [..., N_atom, 3]
                - s: Single embedding if has_coords=True, shape [..., N_token, c_s]
                - z: Pair embedding if has_coords=True, shape [..., N_token, N_token, c_z]
                - inplace_safe: Whether to do some ops in-place for memory efficiency
                - chunk_size: If set, break computations into smaller chunks to reduce memory usage

        Returns:
            Tuple containing:
                - Final token-level embedding, shape [..., N_token, c_token]
                - Atom-level embedding, shape [..., N_atom, c_atom]
                - Another atom-level embedding, shape [..., N_atom, c_atom]
                - The trunk-based pair embedding or None if trunk is skipped
        """
        params = EncoderForwardParams(
            input_feature_dict=input_feature_dict,
            r_l=kwargs.get("r_l"),
            s=kwargs.get("s"),
            z=kwargs.get("z"),
            inplace_safe=kwargs.get("inplace_safe", False),
            chunk_size=kwargs.get("chunk_size"),
        )
        return self.forward(params)

    # For backward compatibility - creates a config from args
    @classmethod
    def from_args(
        cls,
        has_coords: bool,
        c_token: int,
        **kwargs: Any,
    ) -> "AtomAttentionEncoder":
        """
        Create AtomAttentionEncoder from arguments for backward compatibility.

        Args:
            has_coords: Whether to use coordinates
            c_token: Token embedding dimension
            **kwargs: Additional configuration parameters

        Returns:
            AtomAttentionEncoder instance
        """
        # Create configuration with defaults
        config = AtomAttentionConfig(has_coords=has_coords, c_token=c_token, **kwargs)

        # Validate and create encoder
        return cls(config)
