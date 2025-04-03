# rna_predict/pipeline/stageD/diffusion/components/diffusion_module.py
import warnings
from typing import Optional, Union, Tuple, Any

import torch
import torch.nn as nn

# Imports from the original location - adjust paths as needed if these move too
from rna_predict.pipeline.stageA.input_embedding.current.checkpointing import (
    get_checkpoint_fn,
)
from rna_predict.pipeline.stageA.input_embedding.current.primitives import (
    LayerNorm,
    LinearNoBias,
    Transition,
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
from .diffusion_utils import _ensure_tensor_shape, _calculate_edm_scaling_factors


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
        c_noise_embedding: int = 256, # Added for DiffusionConditioning init
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
            c_noise_embedding=c_noise_embedding # Pass necessary arg
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
             if hasattr(self.atom_attention_encoder, 'atom_transformer') and \
                hasattr(self.atom_attention_encoder.atom_transformer, 'diffusion_transformer'):
                 for block in self.atom_attention_encoder.atom_transformer.diffusion_transformer.blocks:
                     if hasattr(block, 'attention_pair_bias') and hasattr(block.attention_pair_bias, 'glorot_init'):
                         block.attention_pair_bias.glorot_init()
                     else:
                          warnings.warn("Could not apply glorot_init_self_attention to atom_encoder block.")
             else:
                 warnings.warn("Atom encoder structure changed, cannot apply glorot_init_self_attention.")


        for block in self.diffusion_transformer.blocks:
            if initialization.get("zero_init_adaln", False):
                if hasattr(block, 'attention_pair_bias') and hasattr(block.attention_pair_bias, 'layernorm_a'):
                    block.attention_pair_bias.layernorm_a.zero_init()
                if hasattr(block, 'conditioned_transition_block') and hasattr(block.conditioned_transition_block, 'adaln'):
                    block.conditioned_transition_block.adaln.zero_init()
                else:
                    warnings.warn("Could not apply zero_init_adaln to diffusion_transformer block.")

            if initialization.get("zero_init_residual_condition_transition", False):
                 if hasattr(block, 'conditioned_transition_block') and hasattr(block.conditioned_transition_block, 'linear_nobias_b'):
                     nn.init.zeros_(block.conditioned_transition_block.linear_nobias_b.weight)
                 else:
                     warnings.warn("Could not apply zero_init_residual_condition_transition to diffusion_transformer block.")

        if initialization.get("zero_init_atom_decoder_linear", False):
            if hasattr(self.atom_attention_decoder, 'linear_no_bias_a'):
                 nn.init.zeros_(self.atom_attention_decoder.linear_no_bias_a.weight)
            else:
                 warnings.warn("Could not apply zero_init_atom_decoder_linear.")


        if initialization.get("zero_init_dit_output", False):
            if hasattr(self.atom_attention_decoder, 'linear_no_bias_out'):
                nn.init.zeros_(self.atom_attention_decoder.linear_no_bias_out.weight)
            else:
                 warnings.warn("Could not apply zero_init_dit_output.")

    def _determine_n_sample(self, r_noisy: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """Determines N_sample and ensures r_noisy has the sample dimension."""
        print(f"[DEBUG] _determine_n_sample input shape: {r_noisy.shape}")
        n_sample_dim_index = -3  # Expected: [..., N_sample, N_atom, 3]
        original_ndim = r_noisy.ndim
        original_shape = r_noisy.shape

        # Handle 5D tensors by squeezing unnecessary dimensions
        if original_ndim == 5:
            # Check if any dimension before N_sample is 1 and can be squeezed
            for dim in range(original_ndim - 3):  # Only check dimensions before N_sample
                if r_noisy.shape[dim] == 1:
                    r_noisy = r_noisy.squeeze(dim)
                    print(f"[DEBUG] Squeezed dimension {dim}, new shape: {r_noisy.shape}")
                    break

        # Handle 4D tensors
        if r_noisy.ndim == 4:
            if r_noisy.shape[n_sample_dim_index] > 0:
                N_sample = r_noisy.size(n_sample_dim_index)
                print(f"[DEBUG] Found N_sample={N_sample} in 4D tensor")
            else:
                N_sample = 1
                r_noisy = r_noisy.unsqueeze(n_sample_dim_index)
                print(f"[DEBUG] Added sample dimension to 4D tensor, new shape: {r_noisy.shape}")
        # Handle 3D tensors
        elif r_noisy.ndim == 3:
            N_sample = 1
            r_noisy = r_noisy.unsqueeze(n_sample_dim_index)
            print(f"[DEBUG] Added sample dimension to 3D tensor, new shape: {r_noisy.shape}")
        else:
            N_sample = 1
            if r_noisy.ndim >= 3:
                r_noisy = r_noisy.unsqueeze(n_sample_dim_index)
                print(f"[DEBUG] Added sample dimension to tensor, new shape: {r_noisy.shape}")
            else:
                raise ValueError(f"Cannot handle r_noisy shape: {original_shape}")

        print(f"[DEBUG] _determine_n_sample output - N_sample: {N_sample}, shape: {r_noisy.shape}")
        return N_sample, r_noisy

    def _run_with_checkpointing(self, module: nn.Module, *args, **kwargs) -> Any: # Changed return type hint
        """Runs a module with optional gradient checkpointing."""
        use_ckpt = self.blocks_per_ckpt is not None and torch.is_grad_enabled()
        # Fine-grained checkpointing might apply to specific modules (e.g., encoder/decoder)
        # Add specific checks if needed, here we use a general flag
        use_fine_grained = use_ckpt and self.use_fine_grained_checkpoint

        # Note: Fine-grained checkpointing logic might need to be more specific
        # depending on how it's implemented within the sub-modules themselves.
        # This wrapper assumes the module call itself can be checkpointed.

        if use_ckpt: # Includes fine-grained if enabled
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
             warnings.warn(f"Expected 'ref_mask' to be Tensor or None, got {type(atom_mask_val)}. Setting mask to None.")

        mask: Optional[torch.Tensor] = atom_mask # Use same mask for now

        atom_to_token_idx: Optional[torch.Tensor] = None
        if isinstance(atom_to_token_idx_val, torch.Tensor):
            atom_to_token_idx = atom_to_token_idx_val
        elif atom_to_token_idx_val is not None:
             warnings.warn(f"Expected 'atom_to_token_idx' to be Tensor or None, got {type(atom_to_token_idx_val)}. Setting index to None.")


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
        r_noisy: torch.Tensor,
        t_hat_noise_level: torch.Tensor,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        s_inputs: Optional[torch.Tensor], # Allow None
        s_trunk: torch.Tensor,
        z_trunk: Optional[torch.Tensor], # Allow None
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Core network forward pass F_theta(c_in * x, c_noise(sigma)).
        """
        print("[DEBUG] Starting f_forward")
        print(f"[DEBUG] Input shapes - r_noisy: {r_noisy.shape}, t_hat_noise_level: {t_hat_noise_level.shape}")
        print(f"[DEBUG] s_trunk shape: {s_trunk.shape}")
        if s_inputs is not None:
            print(f"[DEBUG] s_inputs shape: {s_inputs.shape}")
        if z_trunk is not None:
            print(f"[DEBUG] z_trunk shape: {z_trunk.shape}")

        # 1. Determine N_sample and ensure r_noisy has sample dimension
        print("[DEBUG] Determining N_sample")
        N_sample, r_noisy = self._determine_n_sample(r_noisy)
        print(f"[DEBUG] N_sample: {N_sample}, r_noisy shape after: {r_noisy.shape}")

        # 2. Ensure t_hat_noise_level has compatible shape
        print("[DEBUG] Ensuring t_hat_noise_level shape compatibility")
        t_hat_target_ndim = r_noisy.ndim - 2
        if t_hat_target_ndim < 1: t_hat_target_ndim = 1
        t_hat_noise_level = _ensure_tensor_shape(
            t_hat_noise_level,
            target_ndim=t_hat_target_ndim,
            target_shape=(r_noisy.shape[0], N_sample) if t_hat_target_ndim==2 else (r_noisy.shape[0],),
            warn_prefix="[f_forward t_hat]"
        )
        print(f"[DEBUG] t_hat_noise_level shape after reshape: {t_hat_noise_level.shape}")

        # 3. Apply Diffusion Conditioning (using imported module)
        print("[DEBUG] Applying Diffusion Conditioning")
        # Explicitly type the unpacked variables
        s_single: torch.Tensor
        z_pair: torch.Tensor
        s_single, z_pair = self._run_with_checkpointing(
            self.diffusion_conditioning,
            t_hat_noise_level=t_hat_noise_level,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            inplace_safe=inplace_safe,
        )
        print(f"[DEBUG] After conditioning - s_single shape: {s_single.shape}, z_pair shape: {z_pair.shape}")

        # 4. Expand s_trunk to match s_single's dimensions for the encoder
        print("[DEBUG] Expanding s_trunk")
        s_trunk_expanded = _ensure_tensor_shape(s_trunk, s_single.ndim, ref_tensor=s_single, warn_prefix="[f_forward s_trunk]")
        print(f"[DEBUG] s_trunk_expanded shape: {s_trunk_expanded.shape}")

        # 5. Apply Atom Attention Encoder
        print("[DEBUG] Applying Atom Attention Encoder")
        # Explicitly type the unpacked variables (assuming Optional for skips)
        a_token: torch.Tensor
        q_skip: Optional[torch.Tensor]
        c_skip: Optional[torch.Tensor] # c_skip is not used later, but typing for completeness
        p_skip: Optional[torch.Tensor]
        a_token, q_skip, c_skip, p_skip = self._run_with_checkpointing(
            self.atom_attention_encoder,
            input_feature_dict=input_feature_dict,
            r_l=r_noisy,
            s=s_trunk_expanded,
            z=z_pair, # Pass original z_pair to encoder
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        print(f"[DEBUG] After encoder - a_token shape: {a_token.shape}")
        if q_skip is not None:
            print(f"[DEBUG] q_skip shape: {q_skip.shape}")
        if p_skip is not None:
            print(f"[DEBUG] p_skip shape: {p_skip.shape}")

        # 6. Combine with Single Conditioning and Apply Transformer
        print("[DEBUG] Combining with Single Conditioning")
        s_single_proj = self.linear_no_bias_s(self.layernorm_s(s_single))
        print(f"[DEBUG] s_single_proj shape: {s_single_proj.shape}")

        # Ensure a_token and s_single_proj have compatible shapes
        if a_token.shape != s_single_proj.shape:
            print(f"[DEBUG] Shape mismatch - a_token: {a_token.shape}, s_single_proj: {s_single_proj.shape}")
            # Try to reshape s_single_proj to match a_token's shape
            if len(a_token.shape) > len(s_single_proj.shape):
                # Add missing dimensions to s_single_proj
                for _ in range(len(a_token.shape) - len(s_single_proj.shape)):
                    s_single_proj = s_single_proj.unsqueeze(0)
                print(f"[DEBUG] Reshaped s_single_proj to: {s_single_proj.shape}")
            elif len(s_single_proj.shape) > len(a_token.shape):
                # Add missing dimensions to a_token
                for _ in range(len(s_single_proj.shape) - len(a_token.shape)):
                    a_token = a_token.unsqueeze(0)
                print(f"[DEBUG] Reshaped a_token to: {a_token.shape}")

        if a_token.shape == s_single_proj.shape:
            if inplace_safe:
                a_token += s_single_proj
            else:
                a_token = a_token + s_single_proj
            print(f"[DEBUG] After combining - a_token shape: {a_token.shape}")
        else:
            warnings.warn(f"Shape mismatch between a_token ({a_token.shape}) and projected s_single ({s_single_proj.shape}). Skipping addition.")

        # Expand z_pair based on a_token's dimensions AFTER a_token is generated
        print("[DEBUG] Expanding z_pair")
        # Target ndim should be a_token.ndim + 1 (e.g., if a=4D, z should be 5D)
        target_z_ndim = a_token.ndim + 1
        print(f"[DEBUG] Target z_pair ndim: {target_z_ndim}, current z_pair shape: {z_pair.shape}")
        
        # First ensure z_pair has the right number of dimensions
        while z_pair.ndim < target_z_ndim:
            z_pair = z_pair.unsqueeze(0)
            print(f"[DEBUG] Added dimension to z_pair, new shape: {z_pair.shape}")
        
        # Then try to match the feature dimension
        try:
            z_pair_expanded = _ensure_tensor_shape(
                z_pair,
                target_ndim=target_z_ndim,
                ref_tensor=a_token.unsqueeze(-1),  # Use a_token shape as reference (adding feature dim)
                warn_prefix="[f_forward z_pair post-a_token]"
            )
            print(f"[DEBUG] z_pair_expanded shape: {z_pair_expanded.shape}")
        except RuntimeError as e:
            print(f"[DEBUG] Failed to expand z_pair: {e}")
            # If expansion fails, try to reshape z_pair to match a_token's dimensions
            z_pair_expanded = z_pair.view(*a_token.shape[:-1], -1)
            print(f"[DEBUG] Reshaped z_pair to: {z_pair_expanded.shape}")

        # Explicitly type the output of the transformer
        print("[DEBUG] Applying Diffusion Transformer")
        a_token_transformed: torch.Tensor
        a_token_transformed = self._run_with_checkpointing(
            self.diffusion_transformer,
            a=a_token,
            s=s_single,
            z=z_pair_expanded, # Use correctly expanded z_pair
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        print(f"[DEBUG] a_token_transformed shape: {a_token_transformed.shape}")
        a_token = self.layernorm_a(a_token_transformed) # Assign back to a_token after layernorm
        print(f"[DEBUG] a_token shape after layernorm: {a_token.shape}")

        # 7. Prepare Decoder Inputs
        print("[DEBUG] Preparing Decoder Inputs")
        decoder_params = self._prepare_decoder_params(
            a_token=a_token,
            r_noisy=r_noisy,
            q_skip=q_skip,
            p_skip=p_skip,
            input_feature_dict=input_feature_dict,
            chunk_size=chunk_size,
        )

        # 8. Apply Atom Attention Decoder
        print("[DEBUG] Applying Atom Attention Decoder")
        # Explicitly type the output
        r_update: torch.Tensor
        r_update = self._run_with_checkpointing(
            self.atom_attention_decoder,
            params=decoder_params,
        )
        print(f"[DEBUG] r_update shape: {r_update.shape}")

        return r_update


    def forward(
        self,
        x_noisy: torch.Tensor,
        t_hat_noise_level: torch.Tensor,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        s_inputs: Optional[torch.Tensor], # Allow None
        s_trunk: torch.Tensor,
        z_trunk: Optional[torch.Tensor], # Allow None
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Performs one step of denoising using the EDM formulation.
        Uses imported _calculate_edm_scaling_factors.
        """
        print("[DEBUG] Starting DiffusionModule forward")
        print(f"[DEBUG] x_noisy shape: {x_noisy.shape}")
        print(f"[DEBUG] t_hat_noise_level shape: {t_hat_noise_level.shape}")
        print(f"[DEBUG] s_trunk shape: {s_trunk.shape}")
        if s_inputs is not None:
            print(f"[DEBUG] s_inputs shape: {s_inputs.shape}")
        if z_trunk is not None:
            print(f"[DEBUG] z_trunk shape: {z_trunk.shape}")

        # 1. Calculate EDM scaling factors (using imported utility)
        print("[DEBUG] Calculating EDM scaling factors")
        c_in, c_skip, c_out = _calculate_edm_scaling_factors(
            sigma=t_hat_noise_level,
            sigma_data=self.sigma_data,
            ref_tensor=x_noisy
        )
        print(f"[DEBUG] EDM factors shapes - c_in: {c_in.shape}, c_skip: {c_skip.shape}, c_out: {c_out.shape}")

        # 2. Scale noisy input: r_noisy = c_in * x_noisy
        # Ensure c_in is broadcastable
        try:
            print("[DEBUG] Scaling noisy input")
            r_noisy = c_in * x_noisy
            print(f"[DEBUG] r_noisy shape after scaling: {r_noisy.shape}")
        except RuntimeError as e:
             warnings.warn(f"Could not multiply c_in {c_in.shape} with x_noisy {x_noisy.shape}: {e}. Returning x_noisy.")
             return x_noisy

        # 3. Get the network prediction (coordinate update)
        print("[DEBUG] Calling f_forward")
        r_update = self.f_forward(
            r_noisy=r_noisy,
            t_hat_noise_level=t_hat_noise_level,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        print(f"[DEBUG] r_update shape: {r_update.shape}")

        # 4. Apply denoising formula: x_denoised = c_skip * x_noisy + c_out * r_update
        # Ensure factors are broadcastable and dtypes match
        try:
            print("[DEBUG] Applying denoising formula")
            dtype = r_update.dtype
            x_denoised = (c_skip.to(dtype) * x_noisy.to(dtype) + c_out.to(dtype) * r_update).to(dtype)
            print(f"[DEBUG] x_denoised shape: {x_denoised.shape}")
        except RuntimeError as e:
             warnings.warn(f"Could not compute denoised output. Shapes: c_skip={c_skip.shape}, x_noisy={x_noisy.shape}, c_out={c_out.shape}, r_update={r_update.shape}. Error: {e}. Returning x_noisy.")
             return x_noisy

        return x_denoised
