# rna_predict/pipeline/stageD/diffusion/components/diffusion_module.py
import warnings
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

# Imports from the original location - adjust paths as needed if these move too
from rna_predict.pipeline.stageA.input_embedding.current.checkpointing import (
    get_checkpoint_fn,
)
from rna_predict.pipeline.stageA.input_embedding.current.primitives import (
    LayerNorm,
    LinearNoBias,
)
from rna_predict.pipeline.stageA.input_embedding.current.transformer import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
    DiffusionTransformer,
)
from rna_predict.pipeline.stageA.input_embedding.current.transformer.atom_attention_decoder import (
    DecoderForwardParams,
)
from rna_predict.pipeline.stageA.input_embedding.current.transformer.atom_attention_encoder import (
    AtomAttentionConfig,
)

# Imports from within the components directory
from .diffusion_conditioning import DiffusionConditioning

# Removed import from deleted diffusion_utils.py


class DiffusionModule(nn.Module):
    """
    Implements Algorithm 20 in AF3 (Moved from diffusion.py).
    Uses imported DiffusionConditioning and utility functions.
    """

    def __init__(
        self,
        sigma_data: float = 16.0,
        c_atom: int = 128,
        c_atompair: int = 16,
        c_token: int = 768,
        c_s: int = 384,
        c_z: int = 128,
        c_s_inputs: int = 449,
        c_noise_embedding: int = 256,  # Added for DiffusionConditioning init
        atom_encoder: dict[str, int] = {"n_blocks": 3, "n_heads": 4},
        transformer: dict[str, int] = {"n_blocks": 24, "n_heads": 16},
        atom_decoder: dict[str, int] = {"n_blocks": 3, "n_heads": 4},
        blocks_per_ckpt: Optional[int] = None,
        use_fine_grained_checkpoint: bool = False,
        initialization: Optional[dict[str, Union[str, float, bool]]] = None,
    ) -> None:
        """
        Args:
            sigma_data (torch.float, optional): the standard deviation of the data. Defaults to 16.0.
            c_atom (int, optional): embedding dim for atom feature. Defaults to 128.
            c_atompair (int, optional): embedding dim for atompair feature. Defaults to 16.
            c_token (int, optional): feature channel of token (single a). Defaults to 768.
            c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
            c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
            c_s_inputs (int, optional): hidden dim [for single input embedding]. Defaults to 449.
            c_noise_embedding (int, optional): noise embedding dim for conditioning. Defaults to 256.
            atom_encoder (dict[str, int], optional): configs in AtomAttentionEncoder. Defaults to {"n_blocks": 3, "n_heads": 4}.
            transformer (dict[str, int], optional): configs in DiffusionTransformer. Defaults to {"n_blocks": 24, "n_heads": 16}.
            atom_decoder (dict[str, int], optional): configs in AtomAttentionDecoder. Defaults to {"n_blocks": 3, "n_heads": 4}.
            blocks_per_ckpt: number of atom_encoder/transformer/atom_decoder blocks in each activation checkpoint
                Size of each chunk. A higher value corresponds to fewer
                checkpoints, and trades memory for speed. If None, no checkpointing is performed.
            use_fine_grained_checkpoint: whether use fine-gained checkpoint for finetuning stage 2
                only effective if blocks_per_ckpt is not None.
            initialization: initialize the diffusion module according to initialization config.
        """

        super(DiffusionModule, self).__init__()
        self.sigma_data = sigma_data
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.c_s_inputs = c_s_inputs
        self.c_s = c_s
        self.c_z = c_z

        # Grad checkpoint setting
        self.blocks_per_ckpt = blocks_per_ckpt
        self.use_fine_grained_checkpoint = use_fine_grained_checkpoint

        # Use imported DiffusionConditioning
        self.diffusion_conditioning = DiffusionConditioning(
            sigma_data=self.sigma_data,
            c_z=c_z,
            c_s=c_s,
            c_s_inputs=c_s_inputs,
            c_noise_embedding=c_noise_embedding,  # Pass necessary arg
        )

        # --- AtomAttentionEncoder ---
        encoder_config_dict = atom_encoder
        encoder_config = AtomAttentionConfig(
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_token=c_token,
            has_coords=True,  # Specific to this context in DiffusionModule
            c_s=c_s,
            c_z=c_z,
            blocks_per_ckpt=blocks_per_ckpt,
            n_blocks=encoder_config_dict.get("n_blocks", 3),
            n_heads=encoder_config_dict.get("n_heads", 4),
            n_queries=encoder_config_dict.get("n_queries", 32),  # Add default
            n_keys=encoder_config_dict.get("n_keys", 128),  # Add default
        )
        self.atom_attention_encoder = AtomAttentionEncoder(config=encoder_config)

        # Alg20: line4
        self.layernorm_s = LayerNorm(c_s)
        self.linear_no_bias_s = LinearNoBias(in_features=c_s, out_features=c_token)

        # --- DiffusionTransformer Instantiation ---
        self.diffusion_transformer = DiffusionTransformer(
            **transformer,
            c_a=c_token,  # Note: c_a used here, not c_token directly
            c_s=c_s,
            c_z=c_z,
            blocks_per_ckpt=blocks_per_ckpt,
        )
        self.layernorm_a = LayerNorm(c_token)

        # --- AtomAttentionDecoder ---
        decoder_config_dict = atom_decoder
        decoder_config = AtomAttentionConfig(
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_token=c_token,
            has_coords=True,  # Specific to this context in DiffusionModule
            c_s=c_s,
            c_z=c_z,
            blocks_per_ckpt=blocks_per_ckpt,
            n_blocks=decoder_config_dict.get("n_blocks", 3),
            n_heads=decoder_config_dict.get("n_heads", 4),
            n_queries=decoder_config_dict.get("n_queries", 32),  # Add default
            n_keys=decoder_config_dict.get("n_keys", 128),  # Add default
        )
        self.atom_attention_decoder = AtomAttentionDecoder(config=decoder_config)

        # Handle initialization safely
        if initialization is None:
            initialization = {}  # Ensure initialization is a dict
        self.init_parameters(initialization)

    def init_parameters(self, initialization: dict):
        """
        Initializes the parameters of the diffusion module according to the provided initialization configuration.
        """
        # Note: Initialization for imported DiffusionConditioning happens within its own class
        # if initialization.get("zero_init_condition_transition", False):
        #     self.diffusion_conditioning.transition_z1.zero_init() # Access might change if imported
        #     self.diffusion_conditioning.transition_z2.zero_init()
        #     self.diffusion_conditioning.transition_s1.zero_init()
        #     self.diffusion_conditioning.transition_s2.zero_init()

        self.atom_attention_encoder.linear_init(
            zero_init_atom_encoder_residual_linear=initialization.get(
                "zero_init_atom_encoder_residual_linear", False
            ),
            he_normal_init_atom_encoder_small_mlp=initialization.get(
                "he_normal_init_atom_encoder_small_mlp", False
            ),
            he_normal_init_atom_encoder_output=initialization.get(
                "he_normal_init_atom_encoder_output", False
            ),
        )

        if initialization.get("glorot_init_self_attention", False):
            if hasattr(self.atom_attention_encoder, "atom_transformer") and hasattr(
                self.atom_attention_encoder.atom_transformer, "diffusion_transformer"
            ):
                for block in self.atom_attention_encoder.atom_transformer.diffusion_transformer.blocks:
                    if hasattr(block, "attention_pair_bias") and hasattr(
                        block.attention_pair_bias, "glorot_init"
                    ):
                        block.attention_pair_bias.glorot_init()
                    else:
                        warnings.warn(
                            "Could not apply glorot_init_self_attention to atom_encoder block."
                        )
            else:
                warnings.warn(
                    "Atom encoder structure changed, cannot apply glorot_init_self_attention."
                )

        for block in self.diffusion_transformer.blocks:
            if initialization.get("zero_init_adaln", False):
                if hasattr(block, "attention_pair_bias") and hasattr(
                    block.attention_pair_bias, "layernorm_a"
                ):
                    block.attention_pair_bias.layernorm_a.zero_init()
                if hasattr(block, "conditioned_transition_block") and hasattr(
                    block.conditioned_transition_block, "adaln"
                ):
                    block.conditioned_transition_block.adaln.zero_init()
                else:
                    warnings.warn(
                        "Could not apply zero_init_adaln to diffusion_transformer block."
                    )

            if initialization.get("zero_init_residual_condition_transition", False):
                if hasattr(block, "conditioned_transition_block") and hasattr(
                    block.conditioned_transition_block, "linear_nobias_b"
                ):
                    nn.init.zeros_(
                        block.conditioned_transition_block.linear_nobias_b.weight
                    )
                else:
                    warnings.warn(
                        "Could not apply zero_init_residual_condition_transition to diffusion_transformer block."
                    )

        if initialization.get("zero_init_atom_decoder_linear", False):
            if hasattr(self.atom_attention_decoder, "linear_no_bias_a"):
                nn.init.zeros_(self.atom_attention_decoder.linear_no_bias_a.weight)
            else:
                warnings.warn("Could not apply zero_init_atom_decoder_linear.")

        if initialization.get("zero_init_dit_output", False):
            if hasattr(self.atom_attention_decoder, "linear_no_bias_out"):
                nn.init.zeros_(self.atom_attention_decoder.linear_no_bias_out.weight)
            else:
                warnings.warn("Could not apply zero_init_dit_output.")

    # Removed _determine_n_sample method

    def _run_with_checkpointing(
        self, module: nn.Module, *args, **kwargs
    ) -> Any:  # Changed return type hint
        """Runs a module with optional gradient checkpointing."""
        use_ckpt = self.blocks_per_ckpt is not None and torch.is_grad_enabled()
        # Fine-grained checkpointing might apply to specific modules (e.g., encoder/decoder)
        # Add specific checks if needed, here we use a general flag

        # Note: Fine-grained checkpointing logic might need to be more specific
        # depending on how it's implemented within the sub-modules themselves.
        # This wrapper assumes the module call itself can be checkpointed.

        if use_ckpt:  # Includes fine-grained if enabled
            checkpoint_fn = get_checkpoint_fn()
            # Filter out kwargs not accepted by the module's forward method if necessary
            # For simplicity, assume all kwargs are valid for now
            return checkpoint_fn(module, *args, **kwargs)
        else:
            return module(*args, **kwargs)

    def _prepare_decoder_params(
        self,
        a_token: torch.Tensor,
        r_noisy: torch.Tensor,
        q_skip: Optional[torch.Tensor],
        p_skip: Optional[torch.Tensor],
        input_feature_dict: dict,
        chunk_size: Optional[int],
    ) -> DecoderForwardParams:
        """Prepares the parameters object for the AtomAttentionDecoder."""
        atom_mask_val = input_feature_dict.get("ref_mask")
        atom_to_token_idx_val = input_feature_dict.get("atom_to_token_idx")

        # Ensure masks and indices are Tensors or None
        atom_mask: Optional[torch.Tensor] = None
        if isinstance(atom_mask_val, torch.Tensor):
            atom_mask = atom_mask_val
        elif atom_mask_val is not None:
            warnings.warn(
                f"Expected 'ref_mask' to be Tensor or None, got {type(atom_mask_val)}. Setting mask to None."
            )

        mask: Optional[torch.Tensor] = atom_mask  # Use same mask for now

        atom_to_token_idx: Optional[torch.Tensor] = None
        if isinstance(atom_to_token_idx_val, torch.Tensor):
            atom_to_token_idx = atom_to_token_idx_val
        elif atom_to_token_idx_val is not None:
            warnings.warn(
                f"Expected 'atom_to_token_idx' to be Tensor or None, got {type(atom_to_token_idx_val)}. Setting index to None."
            )

        return DecoderForwardParams(
            a=a_token,
            r_l=r_noisy,
            extra_feats=q_skip,
            p_lm=p_skip,
            mask=mask,
            atom_mask=atom_mask,
            atom_to_token_idx=atom_to_token_idx,
            chunk_size=chunk_size,
        )

    def f_forward(
        self,
        r_noisy: torch.Tensor,  # Expected shape [B, N_sample, N_atom, 3] or similar
        t_hat_noise_level: torch.Tensor,  # Expected shape [B, N_sample] or broadcastable
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        s_inputs: Optional[
            torch.Tensor
        ],  # Allow None, expected [B, N_sample, N_token, C]
        s_trunk: Optional[torch.Tensor],  # Allow None, expected [B, N_sample, N_token, C]
        z_trunk: Optional[
            torch.Tensor
        ],  # Allow None, expected [B, N_sample, N_token, N_token, C]
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Core network forward pass F_theta(c_in * x, c_noise(sigma)).
        Assumes inputs have consistent batch and N_sample dimensions.
        """
        # [DEBUG] Removed internal shape checks and manipulations for clarity
        # print("[DEBUG] Starting f_forward (Refactored)")
        # print(f"[DEBUG] Input shapes - r_noisy: {r_noisy.shape}, t_hat_noise_level: {t_hat_noise_level.shape}")
        # if s_trunk is not None: print(f"[DEBUG] s_trunk shape: {s_trunk.shape}")
        # if s_inputs is not None: print(f"[DEBUG] s_inputs shape: {s_inputs.shape}")
        # if z_trunk is not None: print(f"[DEBUG] z_trunk shape: {z_trunk.shape}")

        # 1. Apply Diffusion Conditioning
        # Assumes diffusion_conditioning handles broadcasting of t_hat_noise_level internally
        # and returns s_single/z_pair with appropriate [B, N_sample, ...] dimensions.
        s_single: torch.Tensor
        z_pair: torch.Tensor
        s_single, z_pair = self._run_with_checkpointing(
            self.diffusion_conditioning,
            t_hat_noise_level=t_hat_noise_level,  # Pass potentially [B, N_sample]
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,  # Pass potentially [B, N_sample, ...]
            s_trunk=s_trunk,  # Pass potentially [B, N_sample, ...]
            z_trunk=z_trunk,  # Pass potentially [B, N_sample, ...]
            inplace_safe=inplace_safe,
        )
        # print(f"[DEBUG] After conditioning - s_single shape: {s_single.shape}, z_pair shape: {z_pair.shape}")

        # 2. Apply Atom Attention Encoder
        # Assumes encoder handles broadcasting and returns shapes like [B, N_sample, ...]
        a_token: torch.Tensor
        q_skip: Optional[torch.Tensor]
        c_skip: Optional[torch.Tensor]  # Not used later
        p_skip: Optional[torch.Tensor]
        a_token, q_skip, c_skip, p_skip = self._run_with_checkpointing(
            self.atom_attention_encoder,
            input_feature_dict=input_feature_dict,
            r_l=r_noisy,  # Pass potentially [B, N_sample, N_atom, 3]
            s=s_trunk,  # Pass potentially [B, N_sample, N_token, C]
            z=z_pair,  # Pass potentially [B, N_sample, N_token, N_token, C]
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        # print(f"[DEBUG] After encoder - a_token shape: {a_token.shape}")
        # if q_skip is not None: print(f"[DEBUG] q_skip shape: {q_skip.shape}")
        # if p_skip is not None: print(f"[DEBUG] p_skip shape: {p_skip.shape}")

        # 3. Combine with Single Conditioning and Apply Transformer
        s_single_proj = self.linear_no_bias_s(self.layernorm_s(s_single))
        # print(f"[DEBUG] s_single_proj shape: {s_single_proj.shape}")

        # Add using broadcasting if shapes are compatible (e.g., both [B, N_sample, N_token, C])
        if a_token.shape == s_single_proj.shape:
            if inplace_safe:
                a_token += s_single_proj
            else:
                a_token = a_token + s_single_proj
            # print(f"[DEBUG] After combining - a_token shape: {a_token.shape}")
        else:
            # Attempt broadcasting if dimensions allow (e.g., one is missing N_sample)
            try:
                if inplace_safe:
                    a_token += s_single_proj  # Relies on broadcasting
                else:
                    a_token = a_token + s_single_proj  # Relies on broadcasting
                # print(f"[DEBUG] After combining (broadcasted) - a_token shape: {a_token.shape}")
            except RuntimeError as e:
                warnings.warn(
                    f"Shape mismatch & broadcast failed between a_token ({a_token.shape}) and projected s_single ({s_single_proj.shape}). Skipping addition. Error: {e}"
                )

        # Ensure z_pair has compatible dimensions for the transformer's attention bias
        # Typically expects z.ndim == a.ndim + 1, or matching leading dims.
        # Let's try to add leading singleton dims to z if needed.
        if z_pair is not None:
            target_z_ndim = a_token.ndim + 1  # Expected relationship for attention bias
            while z_pair.ndim < target_z_ndim:
                z_pair = z_pair.unsqueeze(0)
            # print(f"[DEBUG] Expanded z_pair shape for transformer: {z_pair.shape}")

        # Apply Transformer
        # Assumes transformer handles broadcasting of s_single/z_pair
        a_token_transformed: torch.Tensor
        a_token_transformed = self._run_with_checkpointing(
            self.diffusion_transformer,
            a=a_token,
            s=s_single,  # Pass potentially [B, N_sample, ...]
            z=z_pair,  # Pass potentially [B, N_sample, ...]
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        # print(f"[DEBUG] a_token_transformed shape: {a_token_transformed.shape}")
        a_token = self.layernorm_a(a_token_transformed)
        # print(f"[DEBUG] a_token shape after layernorm: {a_token.shape}")

        # 4. Prepare Decoder Inputs
        decoder_params = self._prepare_decoder_params(
            a_token=a_token,
            r_noisy=r_noisy,
            q_skip=q_skip,
            p_skip=p_skip,
            input_feature_dict=input_feature_dict,
            chunk_size=chunk_size,
        )

        # 5. Apply Atom Attention Decoder
        # Assumes decoder handles broadcasting
        r_update: torch.Tensor
        r_update = self._run_with_checkpointing(
            self.atom_attention_decoder,
            params=decoder_params,
        )
        # print(f"[DEBUG] r_update shape: {r_update.shape}")

        return r_update

    def forward(
        self,
        x_noisy: torch.Tensor,
        t_hat_noise_level: torch.Tensor,
        input_feature_dict: Dict[str, Any],
        s_inputs: Optional[torch.Tensor] = None,
        s_trunk: Optional[torch.Tensor] = None,
        z_trunk: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        inplace_safe: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the diffusion module.
        Handles N_sample detection and prepares inputs for f_forward.
        """
        # print("[DEBUG] Starting DiffusionModule forward (Refactored)")
        # print(f"[DEBUG] Initial x_noisy shape: {x_noisy.shape}")
        # print(f"[DEBUG] Initial t_hat_noise_level shape: {t_hat_noise_level.shape}")

        # --- Input Shape Handling ---
        # Assume x_noisy arrives as [B, N_atom, 3] or [B, N_sample, N_atom, 3]
        if x_noisy.ndim == 3:
            N_sample = 1
            # Add sample dimension for internal consistency
            x_noisy = x_noisy.unsqueeze(1)  # [B, 1, N_atom, 3]
            # Ensure t_hat is at least [B] or [B, 1]
            if t_hat_noise_level.ndim == 0:
                t_hat_noise_level = t_hat_noise_level.unsqueeze(0)  # [1]
            if (
                t_hat_noise_level.ndim == 1
                and t_hat_noise_level.shape[0] == x_noisy.shape[0]
            ):
                t_hat_noise_level = t_hat_noise_level.unsqueeze(1)  # [B, 1]
            elif t_hat_noise_level.shape != (x_noisy.shape[0], 1):
                warnings.warn(
                    f"Broadcasting t_hat ({t_hat_noise_level.shape}) to x_noisy ({x_noisy.shape}) might be ambiguous."
                )
        elif x_noisy.ndim == 4:
            N_sample = x_noisy.shape[1]
            # Ensure t_hat is [B, N_sample]
            if (
                t_hat_noise_level.ndim == 1
                and t_hat_noise_level.shape[0] == x_noisy.shape[0]
            ):
                # Assume t_hat was [B], needs expansion to [B, N_sample]
                t_hat_noise_level = t_hat_noise_level.unsqueeze(1).expand(-1, N_sample)
            elif t_hat_noise_level.shape != (x_noisy.shape[0], N_sample):
                warnings.warn(
                    f"Broadcasting t_hat ({t_hat_noise_level.shape}) to x_noisy ({x_noisy.shape}) might be ambiguous."
                )
        else:
            raise ValueError(
                f"Unexpected x_noisy dimensions: {x_noisy.ndim}. Expected 3 or 4."
            )

        # Ensure conditioning tensors also have the sample dimension if N_sample > 1
        if N_sample > 1:
            if s_inputs is not None and s_inputs.ndim == 3:
                s_inputs = s_inputs.unsqueeze(1).expand(-1, N_sample, -1, -1)
            if s_trunk is not None and s_trunk.ndim == 3:
                s_trunk = s_trunk.unsqueeze(1).expand(-1, N_sample, -1, -1)
            if z_trunk is not None and z_trunk.ndim == 4:
                z_trunk = z_trunk.unsqueeze(1).expand(-1, N_sample, -1, -1, -1)
            # Expand relevant tensors within input_feature_dict as well
            if "atom_to_token_idx" in input_feature_dict:
                atom_idx = input_feature_dict["atom_to_token_idx"]
                if isinstance(atom_idx, torch.Tensor) and atom_idx.ndim == 2: # Shape [B, N_token]
                    input_feature_dict["atom_to_token_idx"] = atom_idx.unsqueeze(1).expand(-1, N_sample, -1)

            if "restype" in input_feature_dict:
                res_type = input_feature_dict["restype"]
                if isinstance(res_type, torch.Tensor) and res_type.ndim == 2: # Shape [B, N_token]
                    input_feature_dict["restype"] = res_type.unsqueeze(1).expand(-1, N_sample, -1)

            # Note: Other features like ref_mask, ref_charge, ref_element might also need expansion
            # depending on how they are used downstream. Add expansion here if necessary.
            # Example for ref_mask (assuming it's [B, N_token, 1]):
            # if "ref_mask" in input_feature_dict:
            #     mask = input_feature_dict["ref_mask"]

            if "ref_mask" in input_feature_dict:
                mask = input_feature_dict["ref_mask"]
                # Assuming ref_mask is [B, N_token, 1] or similar 3D shape
                if isinstance(mask, torch.Tensor) and mask.ndim == 3:
                    input_feature_dict["ref_mask"] = mask.unsqueeze(1).expand(-1, N_sample, -1, -1)

            # Note: Other features like ref_charge, ref_element might also need expansion
            # depending on how they are used downstream. Add expansion here if necessary.
            #     if isinstance(mask, torch.Tensor) and mask.ndim == 3:
            #         input_feature_dict["ref_mask"] = mask.unsqueeze(1).expand(-1, N_sample, -1, -1)
        # If N_sample is 1, assume conditioning tensors might be [B, ...] or [B, 1, ...]
        # Submodules should handle broadcasting from [B, ...] or explicit [B, 1, ...]

        # print(f"[DEBUG] Processed x_noisy shape: {x_noisy.shape}")
        # print(f"[DEBUG] Processed t_hat_noise_level shape: {t_hat_noise_level.shape}")
        # if s_trunk is not None: print(f"[DEBUG] Processed s_trunk shape: {s_trunk.shape}")
        # if s_inputs is not None: print(f"[DEBUG] Processed s_inputs shape: {s_inputs.shape}")
        # if z_trunk is not None: print(f"[DEBUG] Processed z_trunk shape: {z_trunk.shape}")

        # --- Core Logic ---
        # Calculate EDM scaling factors, ensuring they broadcast correctly
        # t_hat_noise_level should be [B, N_sample] or [B, 1] at this point
        c_in, c_skip, c_out = self._calculate_edm_scaling_factors(
            t_hat_noise_level
        )  # sigma is t_hat
        # Reshape factors to broadcast with x_noisy [B, N_sample, N_atom, 3]
        c_in = c_in.view(*c_in.shape, 1, 1)
        c_skip = c_skip.view(*c_skip.shape, 1, 1)
        c_out = c_out.view(*c_out.shape, 1, 1)
        # print(f"[DEBUG] EDM factors shapes - c_in: {c_in.shape}, c_skip: {c_skip.shape}, c_out: {c_out.shape}")

        # Scale noisy input
        r_noisy = (
            x_noisy * c_in
        )  # Broadcasting: [B, N_sample, N_atom, 3] * [B, N_sample, 1, 1]
        # print(f"[DEBUG] r_noisy shape after scaling: {r_noisy.shape}")

        # Forward pass through f_theta
        r_update = self.f_forward(
            r_noisy=r_noisy,
            t_hat_noise_level=t_hat_noise_level,  # Pass potentially [B, N_sample]
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,  # Pass potentially [B, N_sample, ...]
            s_trunk=s_trunk,  # Pass potentially [B, N_sample, ...]
            z_trunk=z_trunk,  # Pass potentially [B, N_sample, ...]
            chunk_size=chunk_size,
            inplace_safe=inplace_safe,
        )
        # print(f"[DEBUG] r_update shape: {r_update.shape}")

        # Apply denoising formula
        # Broadcasting: [B, N_sample, N_atom, 3] * [B, N_sample, 1, 1] + [B, N_sample, N_atom, 3] * [B, N_sample, 1, 1]
        x_denoised = r_noisy * c_skip + r_update * c_out
        # print(f"[DEBUG] x_denoised shape: {x_denoised.shape}")

        # Compute loss if target is available
        loss = torch.tensor(0.0, device=x_denoised.device)
        if "ref_pos" in input_feature_dict:
            x_target = input_feature_dict["ref_pos"]
            # Ensure target also has sample dimension if needed
            if x_target.ndim == 3 and x_denoised.ndim == 4:
                x_target = x_target.unsqueeze(1).expand_as(x_denoised)
            mask = input_feature_dict.get("ref_mask", None)
            # Ensure mask also has sample dimension if needed
            if mask is not None and mask.ndim == 3 and x_denoised.ndim == 4:
                mask = mask.unsqueeze(1).expand(
                    *x_denoised.shape[:-1], 1
                )  # Expand to [B, N_sample, N_atom, 1]

            # Pass t_hat_noise_level which is already [B, N_sample] or [B, 1]
            loss = self._compute_loss(x_denoised, x_target, t_hat_noise_level, mask)

        # Return shape [B, N_sample, N_atom, 3]
        print(
            f"[DEBUG][DiffusionModule.forward] Returning x_denoised shape: {x_denoised.shape}"
        )  # DEBUG PRINT
        return x_denoised, loss.squeeze()

    def _compute_loss(
        self,
        x_denoised: torch.Tensor,
        x_target: torch.Tensor,
        t_hat_noise_level: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the EDM loss between denoised and target coordinates.

        Args:
            x_denoised: Denoised coordinates
            x_target: Target coordinates
            t_hat_noise_level: Noise level at time t_hat
            mask: Optional mask for valid coordinates

        Returns:
            Scalar loss tensor
        """
        # Compute squared error
        squared_error = (x_denoised - x_target).pow(2).sum(dim=-1)  # [..., N_atom]

        # Apply mask if provided
        if mask is not None:
            # Ensure mask is broadcastable to squared_error
            while mask.ndim < squared_error.ndim:
                mask = mask.unsqueeze(-1)
            if mask.shape[-1] == 1:  # Handle potential extra dim from expansion
                mask = mask.squeeze(-1)
            squared_error = squared_error * mask

        # Weight by noise level
        # Ensure t_hat is broadcastable to squared_error [B, N_sample, N_atom]
        t_hat_expanded = t_hat_noise_level
        while t_hat_expanded.ndim < squared_error.ndim:
            t_hat_expanded = t_hat_expanded.unsqueeze(-1)
        weighted_error = squared_error / (t_hat_expanded**2 + 1e-8)  # Add epsilon

        # Average over atoms and batch dimensions
        if mask is not None:
            # Average only over valid atoms
            num_valid = mask.sum(dim=-1, keepdim=True)  # [B, N_sample, 1]
            loss = (
                weighted_error.sum(dim=-1, keepdim=True) / (num_valid + 1e-8)
            ).mean()
        else:
            # Simple mean over all dimensions except batch/sample
            loss = weighted_error.mean(
                dim=list(range(2, weighted_error.ndim))
            ).mean()  # Mean over atom and any extra dims, then mean over batch/sample

        return loss.reshape(())  # Ensure scalar

    def _calculate_edm_scaling_factors(
        self,
        sigma: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate EDM scaling factors for denoising.

        Args:
            sigma: Noise level tensor [B, N_sample] or [B, 1] or [B] or scalar

        Returns:
            Tuple of (c_in, c_skip, c_out) scaling factors, broadcastable to [B, N_sample]
        """
        # Ensure sigma is at least 1D
        if sigma.ndim == 0:
            sigma = sigma.unsqueeze(0)

        # Calculate EDM scaling factors
        sigma_sq = sigma**2
        c_skip = 1 / (sigma_sq + 1)
        c_out = sigma * c_skip
        c_in = 1 / (sigma * (sigma_sq + 1) ** 0.5 + 1e-8)  # Add epsilon for stability

        return c_in, c_skip, c_out
