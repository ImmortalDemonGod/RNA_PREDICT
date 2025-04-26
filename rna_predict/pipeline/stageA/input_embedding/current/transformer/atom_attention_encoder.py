"""
Atom attention encoder module for RNA structure prediction.
Refactored version using components from the 'encoder_components' directory.
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from rna_predict.pipeline.stageA.input_embedding.current.primitives import LinearNoBias
from rna_predict.pipeline.stageA.input_embedding.current.transformer.common import (
    InputFeatureDict,
)

from .encoder_components.config import (
    AtomAttentionConfig,
    EncoderForwardParams,
    ProcessInputsParams,
)
from .encoder_components.feature_processing import (
    ensure_space_uid,
    extract_atom_features as canonical_extract_atom_features,
)
from .encoder_components.forward_logic import (
    _process_simple_embedding,
    process_inputs_with_coords,
)
from .encoder_components.initialization import (
    create_atom_transformer,
    linear_init,
    setup_atom_encoders,
    setup_coordinate_components,
    setup_distance_encoders,
    setup_feature_dimensions,
    setup_pair_projections,
    setup_small_mlp,
)
from rna_predict.pipeline.stageA.input_embedding.current.transformer.atom_attention.components.atom_attention_feature_processing import FeatureProcessor

class AtomAttentionEncoder(nn.Module):
    """
    Encoder that processes atom-level features and produces token-level embeddings.
    Implements Algorithm 5 in AlphaFold3. Uses refactored components.
    """

    def __init__(self, config: AtomAttentionConfig) -> None:
        """
        Initialize the AtomAttentionEncoder with a configuration object.

        Args:
            config: Configuration parameters for the encoder
        """
        super(AtomAttentionEncoder, self).__init__()
        self.config = config  # Store config
        self.has_coords = config.has_coords
        self.c_atom = config.c_atom
        self.c_atompair = config.c_atompair
        self.c_token = config.c_token
        self.c_s = config.c_s
        self.c_z = config.c_z
        self.n_queries = config.n_queries
        self.n_keys = config.n_keys
        self.local_attention_method = (
            "local_cross_attention"  # Keep as attribute if needed elsewhere
        )

        # Setup components using functions from initialization module
        setup_feature_dimensions(self)
        setup_atom_encoders(self, config)
        setup_distance_encoders(self, config)

        if self.has_coords:
            setup_coordinate_components(self, config)

        setup_pair_projections(self, config)
        setup_small_mlp(self, config)

        # Atom transformer for atom-level processing
        self.atom_transformer = create_atom_transformer(config)

        # Output projection to token dimension
        self.linear_no_bias_q = LinearNoBias(
            in_features=self.c_atom, out_features=self.c_token
        )

    # Expose linear_init as a method of the class
    def linear_init(
        self,
        zero_init_atom_encoder_residual_linear: bool = False,
        he_normal_init_atom_encoder_small_mlp: bool = False,
        he_normal_init_atom_encoder_output: bool = False,
    ) -> None:
        """
        Initialize the parameters of the module using the refactored function.
        """
        linear_init(
            self,
            zero_init_atom_encoder_residual_linear,
            he_normal_init_atom_encoder_small_mlp,
            he_normal_init_atom_encoder_output,
        )

    ##@snoop
    def forward(
        self, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for the AtomAttentionEncoder.
        Supports legacy calling with input_feature_dict as the first argument,
        or a single keyword argument 'params' of type EncoderForwardParams.
        Uses refactored forward logic.
        """
        if "params" in kwargs:
            params = kwargs.pop("params")
            input_feature_dict = params.input_feature_dict
            r_l = params.r_l
            s = params.s
            z = params.z
            chunk_size = getattr(params, 'chunk_size', None)
        else:
            if len(args) > 0:
                input_feature_dict = args[0]
            elif "input_feature_dict" in kwargs:
                input_feature_dict = kwargs.pop("input_feature_dict")
            else:
                raise ValueError(
                    "AtomAttentionEncoder.forward requires either a positional argument (input_feature_dict), a keyword argument 'input_feature_dict', or a keyword argument 'params'."
                )
            r_l = kwargs.get("r_l")
            s = kwargs.get("s")
            z = kwargs.get("z")
            chunk_size = kwargs.get("chunk_size")

        print("[DEBUG][AtomAttentionEncoder.forward] input_feature_dict keys:", list(input_feature_dict.keys()))

        # Simple path for no coordinates case
        if not self.has_coords:
            return _process_simple_embedding(self, input_feature_dict)

        # Ensure ref_space_uid exists and has correct shape (moved from process_inputs_with_coords)
        ensure_space_uid(input_feature_dict)

        # Extract atom features from input dictionary
        print("[DEBUG][AtomAttentionEncoder.forward] Extracting c_l...")
        c_l = self.extract_atom_features(input_feature_dict)
        print("[DEBUG][AtomAttentionEncoder.forward] c_l type:", type(c_l), "shape:", getattr(c_l, 'shape', None), "value:", c_l)
        if c_l is None:
            print("[WARNING][AtomAttentionEncoder.forward] c_l is None!")

        # Create ProcessInputsParams object for coordinated case
        process_params = ProcessInputsParams(
            input_feature_dict=input_feature_dict,
            r_l=r_l,
            s=s,
            z=z,  # z is passed but not explicitly used in process_inputs_with_coords currently
            c_l=c_l,
            chunk_size=chunk_size,
        )

        # Full processing path for coordinated case using refactored logic
        return process_inputs_with_coords(self, process_params)

    # For backward compatibility - uses the old parameter style
    def _forward_legacy_disabled(
        self,
        input_feature_dict: InputFeatureDict,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Legacy forward pass with individual parameters. Kept for compatibility.
        """
        params = EncoderForwardParams(
            input_feature_dict=input_feature_dict,
            r_l=kwargs.get("r_l"),
            s=kwargs.get("s"),
            z=kwargs.get("z"),
            inplace_safe=kwargs.get("inplace_safe", False),
            chunk_size=kwargs.get("chunk_size"),
        )
        # Call the main forward method with unpacked parameters
        return self.forward(params=params)  # Use the 'params' argument style

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
        """
        # Create configuration with defaults
        config = AtomAttentionConfig(has_coords=has_coords, c_token=c_token, **kwargs)

        # Validate and create encoder
        return cls(config)

    def forward_debug(
        self, r_l, t_hat_noise_level, input_feature_dict, s, z=None, inplace_safe=False, debug_logging=False, chunk_size=None
    ):
        # SYSTEMATIC DEBUGGING: Log atom_to_token_idx at entry
        print(f"[DEBUG][AtomAttentionEncoder.forward] Entry: input_feature_dict['atom_to_token_idx']: {input_feature_dict.get('atom_to_token_idx', 'MISSING')}")
        if 'atom_to_token_idx' in input_feature_dict:
            atom_idx_val = input_feature_dict['atom_to_token_idx']
            print(f"[DEBUG][AtomAttentionEncoder.forward] type: {type(atom_idx_val)}, shape: {getattr(atom_idx_val, 'shape', None)}, value: {atom_idx_val if isinstance(atom_idx_val, (int, float)) else ''}")
        else:
            print("[DEBUG][AtomAttentionEncoder.forward] atom_to_token_idx MISSING!")

        # Ensure ref_space_uid exists and has correct shape
        ensure_space_uid(input_feature_dict)

        # Extract atom features from input dictionary - CRITICAL FIX
        # This was missing in the original forward_debug method
        c_l = self.extract_atom_features(input_feature_dict)
        print(f"[DEBUG][AtomAttentionEncoder.forward_debug] Extracted c_l shape: {getattr(c_l, 'shape', None)}")

        # Create process_params as in the original forward
        from rna_predict.pipeline.stageA.input_embedding.current.transformer.encoder_components.config import ProcessInputsParams
        process_params = ProcessInputsParams(
            input_feature_dict=input_feature_dict,
            r_l=r_l,
            s=s,
            z=z,
            c_l=c_l,  # Now properly initialized
            chunk_size=chunk_size,
        )
        return process_inputs_with_coords(self, process_params)

    def extract_atom_features(self, input_feature_dict):
        print("[DEBUG][extract_atom_features] input_feature_dict keys:", list(input_feature_dict.keys()))
        for k, v in input_feature_dict.items():
            print(f"    key: {k}, type: {type(v)}, shape: {getattr(v, 'shape', None)}")
        # --- Begin actual extraction logic ---
        # Use the canonical extract_atom_features from encoder_components/feature_processing.py
        c_l = canonical_extract_atom_features(self, input_feature_dict)
        # --- End extraction logic ---
        print("[DEBUG][extract_atom_features] returning c_l type:", type(c_l), "shape:", getattr(c_l, 'shape', None), "value:", c_l)
        return c_l
