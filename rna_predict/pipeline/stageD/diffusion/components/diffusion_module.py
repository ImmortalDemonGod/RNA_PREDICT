# rna_predict/pipeline/stageD/diffusion/components/diffusion_module.py
import logging
from omegaconf import DictConfig, OmegaConf
from typing import Any, Dict, Optional, Tuple, Union, Callable
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

class DiffusionModule(nn.Module):
    """
    Implements Algorithm 20 in AF3 (Moved from diffusion.py).
    Uses imported DiffusionConditioning and utility functions.
    """

    def __init__(
        self,
        cfg: DictConfig,
        *args,
        **kwargs
    ):
        """
        Initialize DiffusionModule from a Hydra config (DictConfig or structured config).
        Args:
            cfg: Hydra config for Stage D Diffusion (should be DictConfig or structured)
        """
        
        super().__init__()
        # Validate config structure
        required_fields = [
            "model_architecture", "transformer", "atom_encoder", "atom_decoder", "debug_logging"
        ]
        for field in required_fields:
            if field not in cfg:
                raise ValueError(f"Missing required config field: {field}")

        # Extract model architecture parameters
        arch = cfg.model_architecture
        sigma_data = arch.sigma_data
        c_atom = arch.c_atom
        c_atompair = arch.c_atompair
        c_token = arch.c_token
        c_s = arch.c_s
        c_z = arch.c_z
        c_s_inputs = arch.c_s_inputs
        c_noise_embedding = arch.c_noise_embedding

        # Store c_s as an attribute for later use (needed by _prepare_decoder_params)
        self.c_s = c_s

        # Extract transformer/encoder/decoder configs
        atom_encoder = OmegaConf.to_container(cfg.atom_encoder, resolve=True)
        transformer = OmegaConf.to_container(cfg.transformer, resolve=True)
        atom_decoder = OmegaConf.to_container(cfg.atom_decoder, resolve=True)
        if isinstance(transformer, dict):
            blocks_per_ckpt = transformer.get("blocks_per_ckpt", None)
        else:
            blocks_per_ckpt = None
        self.blocks_per_ckpt = blocks_per_ckpt
        initialization = cfg.get("initialization", {})
        debug_logging = cfg.debug_logging

        # Set up logger
        self.logger = logging.getLogger("rna_predict.pipeline.stageD.diffusion.components.diffusion_module")
        self.debug_logging = debug_logging
        if self.debug_logging:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        # --- DiffusionConditioning ---
        self.logger.debug(f"Instantiating DiffusionConditioning with: sigma_data={sigma_data}, c_z={c_z}, c_s={c_s}, c_s_inputs={c_s_inputs}, c_noise_embedding={c_noise_embedding}, debug_logging={debug_logging}")
        self.diffusion_conditioning = DiffusionConditioning(
            sigma_data=sigma_data,
            c_z=c_z,
            c_s=c_s,
            c_s_inputs=c_s_inputs,
            c_noise_embedding=c_noise_embedding,
            debug_logging=debug_logging,
        )

        # --- AtomAttentionEncoder ---
        self.logger.debug(f"encoder_config_dict: {atom_encoder}")
        if isinstance(atom_encoder, dict):
            n_heads = int(atom_encoder.get("n_heads", 8))
            n_queries = int(atom_encoder.get("n_queries", 64))
            n_keys = int(atom_encoder.get("n_keys", 64))
            n_blocks = int(atom_encoder.get("n_blocks", 1))
        else:
            n_heads = 8
            n_queries = 64
            n_keys = 64
            n_blocks = 1
        encoder_config = AtomAttentionConfig(
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_token=c_token,
            has_coords=True,
            c_s=c_s,
            c_z=c_z,
            blocks_per_ckpt=blocks_per_ckpt,
            n_blocks=n_blocks,
            n_heads=n_heads,
            n_queries=n_queries,
            n_keys=n_keys,
            debug_logging=debug_logging,
        )
        self.atom_attention_encoder = AtomAttentionEncoder(config=encoder_config)

        self.layernorm_s = LayerNorm(c_s)
        self.linear_no_bias_s = LinearNoBias(in_features=c_s, out_features=c_token)

        # --- DiffusionTransformer ---
        if isinstance(transformer, dict):
            n_blocks = int(transformer.get("n_blocks", 1))
            n_heads = int(transformer.get("n_heads", 8))
        else:
            n_blocks = 1
            n_heads = 8
        transformer_params = {
            "n_blocks": n_blocks,
            "n_heads": n_heads,
            "c_a": c_token,
            "c_s": c_s,
            "c_z": c_z,
            "blocks_per_ckpt": blocks_per_ckpt
        }
        self.diffusion_transformer = DiffusionTransformer(**transformer_params)
        self.layernorm_a = LayerNorm(c_token)

        # --- AtomAttentionDecoder ---
        if isinstance(atom_decoder, dict):
            n_heads = int(atom_decoder.get("n_heads", 8))
            n_queries = int(atom_decoder.get("n_queries", 64))
            n_keys = int(atom_decoder.get("n_keys", 64))
            n_blocks = int(atom_decoder.get("n_blocks", 1))
        else:
            n_heads = 8
            n_queries = 64
            n_keys = 64
            n_blocks = 1
        decoder_config = AtomAttentionConfig(
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_token=c_token,
            has_coords=True,
            c_s=c_s,
            c_z=c_z,
            blocks_per_ckpt=blocks_per_ckpt,
            n_blocks=n_blocks,
            n_heads=n_heads,
            n_queries=n_queries,
            n_keys=n_keys,
            debug_logging=debug_logging,
        )
        self.atom_attention_decoder = AtomAttentionDecoder(config=decoder_config)

        # Handle initialization safely
        if initialization is None:
            initialization = {}
        self.init_parameters(initialization)

        # Validate initialization dict
        if not isinstance(initialization, dict):
            raise ValueError("initialization must be a dict")

        # Log config
        self.logger.debug(f"DiffusionModule config: {OmegaConf.to_yaml(cfg)}")

        # Accept and ignore extra kwargs for now (for config robustness)
        # Optionally, log or warn about ignored keys
        if kwargs:
            import warnings
            warnings.warn(f"DiffusionModule received unexpected config keys: {list(kwargs.keys())}")

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
                        self.logger.warning(
                            "Could not apply glorot_init_self_attention to atom_encoder block."
                        )
            else:
                self.logger.warning(
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
                    self.logger.warning(
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
                    self.logger.warning(
                        "Could not apply zero_init_residual_condition_transition to diffusion_transformer block."
                    )

        if initialization.get("zero_init_atom_decoder_linear", False):
            if hasattr(self.atom_attention_decoder, "linear_no_bias_a"):
                nn.init.zeros_(self.atom_attention_decoder.linear_no_bias_a.weight)
            else:
                self.logger.warning("Could not apply zero_init_atom_decoder_linear.")

        if initialization.get("zero_init_dit_output", False):
            if hasattr(self.atom_attention_decoder, "linear_no_bias_out"):
                nn.init.zeros_(self.atom_attention_decoder.linear_no_bias_out.weight)
            else:
                self.logger.warning("Could not apply zero_init_dit_output.")

    ###@snoop
    def _run_with_checkpointing(
        self, module: Union[nn.Module, Callable], *args, **kwargs
    ) -> Any:  # Changed return type hint to support both Module and Callable
        """Runs a module or function with optional gradient checkpointing."""
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
        chunk_size: Optional[int] = None,
    ) -> DecoderForwardParams:
        """Prepares the parameters object for the AtomAttentionDecoder."""
        atom_mask_val = input_feature_dict.get("ref_mask")
        atom_to_token_idx_val = input_feature_dict.get("atom_to_token_idx")

        # Ensure masks and indices are Tensors or None
        atom_mask: Optional[torch.Tensor] = None
        if isinstance(atom_mask_val, torch.Tensor):
            atom_mask = atom_mask_val
        elif atom_mask_val is not None:
            self.logger.warning(
                f"Expected 'ref_mask' to be Tensor or None, got {type(atom_mask_val)}. Setting mask to None."
            )

        mask: Optional[torch.Tensor] = atom_mask  # Use same mask for now

        atom_to_token_idx: Optional[torch.Tensor] = None
        if isinstance(atom_to_token_idx_val, torch.Tensor):
            atom_to_token_idx = atom_to_token_idx_val
        elif atom_to_token_idx_val is not None:
            self.logger.warning(
                f"Expected 'atom_to_token_idx' to be Tensor or None, got {type(atom_to_token_idx_val)}. Setting index to None."
            )

        # CRITICAL FIX: Ensure q_skip has the correct feature dimension for the decoder
        # The decoder expects c_s=384 but q_skip might have c_atom=128
        if q_skip is not None:
            # Get the expected feature dimension from the decoder (c_s=384)
            expected_feature_dim = self.c_s  # This is the default c_s in AtomAttentionDecoder

            # Check if q_skip has the wrong feature dimension
            if q_skip.shape[-1] != expected_feature_dim:
                current_dim = q_skip.shape[-1]
                if self.debug_logging:
                    self.logger.debug(f"[DEBUG AGG] Adapting q_skip feature dimension from {current_dim} to {expected_feature_dim}")
                # Adapt the feature dimension
                if current_dim < expected_feature_dim:
                    # Pad with zeros
                    padding = torch.zeros(
                        *q_skip.shape[:-1],
                        expected_feature_dim - current_dim,
                        device=q_skip.device,
                        dtype=q_skip.dtype,
                    )
                    q_skip = torch.cat([q_skip, padding], dim=-1)
                else:
                    # Truncate
                    q_skip = q_skip[..., :expected_feature_dim]

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

    ###@snoop
    def f_forward(
        self,
        r_noisy: torch.Tensor,  # Expected shape [B, N_sample, N_atom, 3] or similar
        t_hat_noise_level: torch.Tensor,  # Expected shape [B, N_sample] or broadcastable
        input_feature_dict: Dict[str, Any],  # Use Dict[str, Any] for compatibility
        s_inputs: Optional[
            torch.Tensor
        ],  # Allow None, expected [B, N_sample, N_token, C]
        s_trunk: Optional[torch.Tensor],  # Allow None, expected [B, N_sample, N_token, C]
        z_trunk: Optional[
            torch.Tensor
        ],  # Allow None, expected [B, N_sample, N_token, N_token, C]
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ):
        # SYSTEMATIC DEBUGGING: Log atom_to_token_idx at entry
        if self.debug_logging:
            self.logger.debug(f"[DEBUG][DiffusionModule.f_forward] Entry: input_feature_dict['atom_to_token_idx']: {input_feature_dict.get('atom_to_token_idx', 'MISSING')}")
        if 'atom_to_token_idx' in input_feature_dict:
            atom_idx_val = input_feature_dict['atom_to_token_idx']
            if self.debug_logging:
                self.logger.debug(f"[DEBUG][DiffusionModule.f_forward] type: {type(atom_idx_val)}, shape: {getattr(atom_idx_val, 'shape', None)}, value: {atom_idx_val if isinstance(atom_idx_val, (int, float)) else ''}")
        else:
            if self.debug_logging:
                self.logger.debug("[DEBUG][DiffusionModule.f_forward] atom_to_token_idx MISSING!")
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
        if self.debug_logging:
            self.logger.debug(f"[DEBUG] After conditioning - s_single shape: {s_single.shape}, z_pair shape: {z_pair.shape}")

        # 2. Apply Atom Attention Encoder
        # Assumes encoder handles broadcasting and returns shapes like [B, N_sample, ...]
        a_token: torch.Tensor
        q_skip: Optional[torch.Tensor]
        c_skip: Optional[torch.Tensor]  # Not used later
        p_skip: Optional[torch.Tensor]
        # SYSTEMATIC DEBUGGING: Log atom_to_token_idx before _run_with_checkpointing
        if self.debug_logging:
            self.logger.debug(f"[DEBUG][DiffusionModule.f_forward] Before _run_with_checkpointing: input_feature_dict['atom_to_token_idx']: {input_feature_dict.get('atom_to_token_idx', 'MISSING')}")
        if 'atom_to_token_idx' in input_feature_dict:
            atom_idx_val = input_feature_dict['atom_to_token_idx']
            if self.debug_logging:
                self.logger.debug(f"[DEBUG][DiffusionModule.f_forward] type: {type(atom_idx_val)}, shape: {getattr(atom_idx_val, 'shape', None)}, value: {atom_idx_val if isinstance(atom_idx_val, (int, float)) else ''}")
        else:
            if self.debug_logging:
                self.logger.debug("[DEBUG][DiffusionModule.f_forward] atom_to_token_idx MISSING!")
        # SYSTEMATIC DEBUGGING: Use forward_debug for AtomAttentionEncoder
        # Replace self.atom_attention_encoder(...) with .forward_debug(...) for deeper logging
        encoder_fn = getattr(self.atom_attention_encoder, "forward_debug", self.atom_attention_encoder)
        a_token, q_skip, c_skip, p_skip = self._run_with_checkpointing(
            encoder_fn,
            r_noisy,
            t_hat_noise_level,
            input_feature_dict,
            s_inputs,
            s_trunk,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        if self.debug_logging:
            self.logger.debug(f"[DEBUG] After encoder - a_token shape: {a_token.shape}")
        if q_skip is not None:
            if self.debug_logging:
                self.logger.debug(f"[DEBUG] q_skip shape: {q_skip.shape}")
        if p_skip is not None:
            if self.debug_logging:
                self.logger.debug(f"[DEBUG] p_skip shape: {p_skip.shape}")

        # 3. Combine with Single Conditioning and Apply Transformer
        s_single_proj = self.linear_no_bias_s(self.layernorm_s(s_single))
        # Safely access shape attributes with getattr to avoid mypy errors
        if self.debug_logging:
            self.logger.debug(f"[DEBUG AGG] a_token shape: {getattr(a_token, 'shape', None)}")

        # Safely access atom_to_token_idx and its shape
        atom_to_token_idx = input_feature_dict.get('atom_to_token_idx')
        if isinstance(atom_to_token_idx, torch.Tensor):
            if self.debug_logging:
                self.logger.debug(f"[DEBUG AGG] atom_to_token_idx shape: {atom_to_token_idx.shape}")
        else:
            if self.debug_logging:
                self.logger.debug(f"[DEBUG AGG] atom_to_token_idx is not a tensor: {type(atom_to_token_idx)}")

        if self.debug_logging:
            self.logger.debug(f"[DEBUG AGG] num_tokens: {a_token.shape[-2] if hasattr(a_token, 'shape') else None}")

        # CRITICAL FIX: Handle token dimension mismatch between a_token and s_single_proj
        # This is the key issue in the RNA pipeline where a_token is token-level (8 residues)
        # but s_single_proj might be atom-level (168 atoms)
        if a_token.shape[-2] != s_single_proj.shape[-2]:
            # If a_token has fewer tokens (residue-level) and s_single_proj has more (atom-level)
            if a_token.shape[-2] < s_single_proj.shape[-2]:
                # We need to aggregate s_single_proj from atom-level to residue-level
                # We need the atom-to-residue mapping from input_feature_dict
                if "atom_to_token_idx" in input_feature_dict:
                    from rna_predict.utils.scatter_utils import scatter_mean

                    # Get atom-to-token mapping
                    atom_to_token_idx = input_feature_dict["atom_to_token_idx"]

                    # Reshape s_single_proj to match the expected input for scatter_mean
                    # We need to handle the batch and sample dimensions
                    s_single_proj_reshaped = s_single_proj.reshape(-1, s_single_proj.shape[-2], s_single_proj.shape[-1])
                    atom_to_token_idx_reshaped = atom_to_token_idx.reshape(-1, atom_to_token_idx.shape[-1])

                    # Aggregate atom-level s_single_proj to token-level
                    s_single_proj_aggregated = []

                    # Handle the case when atom_to_token_idx_reshaped has fewer samples than s_single_proj_reshaped
                    if atom_to_token_idx_reshaped.shape[0] < s_single_proj_reshaped.shape[0]:
                        # Create a new tensor with the correct shape
                        new_atom_to_token_idx = atom_to_token_idx_reshaped[0:1].expand(s_single_proj_reshaped.shape[0], -1)
                        atom_to_token_idx_reshaped = new_atom_to_token_idx

                    for i in range(s_single_proj_reshaped.shape[0]):
                        # Make sure i is within bounds of atom_to_token_idx_reshaped
                        idx = min(i, atom_to_token_idx_reshaped.shape[0] - 1)

                        # Use scatter_mean to aggregate atoms to tokens
                        aggregated = scatter_mean(
                            s_single_proj_reshaped[i],
                            atom_to_token_idx_reshaped[idx],
                            dim_size=a_token.shape[-2],
                            dim=0
                        )
                        s_single_proj_aggregated.append(aggregated)

                    # Stack the aggregated tensors and reshape back to original batch/sample dimensions
                    s_single_proj_aggregated_tensor = torch.stack(s_single_proj_aggregated)
                    s_single_proj = s_single_proj_aggregated_tensor.reshape(*s_single_proj.shape[:-2], a_token.shape[-2], s_single_proj.shape[-1])
                else:
                    # Fallback: use the first a_token.shape[-2] tokens from s_single_proj
                    self.logger.warning(
                        f"atom_to_token_idx not found in input_feature_dict. Using first {a_token.shape[-2]} tokens from s_single_proj."
                    )
                    s_single_proj = s_single_proj[..., :a_token.shape[-2], :]
            else:
                # If a_token has more tokens (atom-level) and s_single_proj has fewer (residue-level)
                # We need to expand s_single_proj from residue-level to atom-level
                if "atom_to_token_idx" in input_feature_dict:
                    # Get atom-to-token mapping
                    atom_to_token_idx = input_feature_dict["atom_to_token_idx"]

                    # Create a new tensor with a_token's shape
                    s_single_proj_expanded = torch.zeros_like(a_token)

                    # Use atom_to_token_idx to map residue embeddings to atoms
                    for i in range(s_single_proj_expanded.shape[0]):  # Batch dimension
                        for j in range(s_single_proj_expanded.shape[1] if s_single_proj_expanded.dim() > 3 else 1):  # Sample dimension
                            # Get the correct indices for this batch and sample
                            batch_idx = i
                            sample_idx = j if s_single_proj_expanded.dim() > 3 else 0

                            # Get atom_to_token_idx for this batch/sample
                            if isinstance(atom_to_token_idx, torch.Tensor):
                                if atom_to_token_idx.dim() > 2:  # Has batch and sample dims
                                    idx = atom_to_token_idx[batch_idx, sample_idx]
                                elif atom_to_token_idx.dim() > 1:  # Has batch dim only
                                    idx = atom_to_token_idx[batch_idx]
                                else:  # No batch dim
                                    idx = atom_to_token_idx
                            else:
                                # Handle non-tensor case
                                self.logger.warning(f"atom_to_token_idx is not a tensor: {type(atom_to_token_idx)}")
                                continue

                            # For each atom, copy the embedding from its corresponding residue
                            if not isinstance(idx, torch.Tensor):
                                continue
                            for k in range(idx.shape[0]):  # For each atom
                                residue_idx = idx[k].item()
                                if residue_idx < s_single_proj.shape[-2]:
                                    if s_single_proj_expanded.dim() > 3:
                                        s_single_proj_expanded[i, j, k] = s_single_proj[i, j, residue_idx]
                                    else:
                                        s_single_proj_expanded[i, k] = s_single_proj[i, residue_idx]

                    # Use the expanded tensor
                    s_single_proj = s_single_proj_expanded
                else:
                    # Fallback: repeat s_single_proj to match a_token's shape
                    self.logger.warning(
                        "atom_to_token_idx not found in input_feature_dict. Repeating s_single_proj to match a_token's shape."
                    )
                    # Repeat each token's embedding to create the required number of tokens
                    repeat_factor = a_token.shape[-2] // s_single_proj.shape[-2]
                    remainder = a_token.shape[-2] % s_single_proj.shape[-2]

                    # Repeat and then add any remainder
                    s_single_proj_repeated = s_single_proj.repeat_interleave(repeat_factor, dim=-2)
                    if remainder > 0:
                        s_single_proj_remainder = s_single_proj[..., :remainder, :]
                        s_single_proj = torch.cat([s_single_proj_repeated, s_single_proj_remainder], dim=-2)

        # Now that shapes match, add the tensors
        if a_token.shape == s_single_proj.shape:
            if inplace_safe:
                a_token += s_single_proj
            else:
                a_token = a_token + s_single_proj
        else:
            # Attempt broadcasting if dimensions allow (e.g., one is missing N_sample)
            try:
                if inplace_safe:
                    a_token += s_single_proj  # Relies on broadcasting
                else:
                    a_token = a_token + s_single_proj  # Relies on broadcasting
            except RuntimeError as e:
                self.logger.warning(
                    f"Shape mismatch & broadcast failed between a_token ({a_token.shape}) and projected s_single ({s_single_proj.shape}). Skipping addition. Error: {e}"
                )

        # Ensure z_pair has compatible dimensions for the transformer's attention bias
        # Typically expects z.ndim == a.ndim + 1, or matching leading dims.
        # Let's try to add leading singleton dims to z if needed.
        if z_pair is not None:
            target_z_ndim = a_token.ndim + 1  # Expected relationship for attention bias
            while z_pair.ndim < target_z_ndim:
                z_pair = z_pair.unsqueeze(0)
            if self.debug_logging:
                self.logger.debug(f"[DEBUG] Expanded z_pair shape for transformer: {z_pair.shape}")

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
        if self.debug_logging:
            self.logger.debug(f"[DEBUG] a_token_transformed shape: {a_token_transformed.shape}")
        a_token = self.layernorm_a(a_token_transformed)
        if self.debug_logging:
            self.logger.debug(f"[DEBUG] a_token shape after layernorm: {a_token.shape}")

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
        if self.debug_logging:
            self.logger.debug(f"[DEBUG] r_update shape: {r_update.shape}")

        return r_update

    ###@snoop
    def forward(
        self,
        x_noisy: torch.Tensor,
        t_hat_noise_level: torch.Tensor,
        input_feature_dict: Dict[str, Any],  # Use Dict[str, Any] for compatibility
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
        # --- DEVICE DEBUG ---
        device = x_noisy.device
        if self.debug_logging:
            self.logger.debug(f"[DEVICE-DEBUG][StageD] DiffusionModule.forward: x_noisy device={device}")
            self.logger.debug(f"[DEVICE-DEBUG][StageD] t_hat_noise_level device={t_hat_noise_level.device}")
        if s_trunk is not None:
            if self.debug_logging:
                self.logger.debug(f"[DEVICE-DEBUG][StageD] s_trunk device={s_trunk.device}")
        if s_inputs is not None:
            if self.debug_logging:
                self.logger.debug(f"[DEVICE-DEBUG][StageD] s_inputs device={s_inputs.device}")
        if z_trunk is not None:
            if self.debug_logging:
                self.logger.debug(f"[DEVICE-DEBUG][StageD] z_trunk device={z_trunk.device}")
        # --- END DEVICE DEBUG ---

        # Ensure input_feature_dict tensors are on the correct device
        for k, v in input_feature_dict.items():
            if isinstance(v, torch.Tensor):
                if v.device != device:
                    if self.debug_logging:
                        self.logger.debug(f"[DEVICE-DEBUG][StageD] input_feature_dict[{k}] moved from {v.device} to {device}")
                    input_feature_dict[k] = v.to(device)

        # Make sure all allocations in _ensure_input_feature_dict use the correct device
        processed_input_dict = self.diffusion_conditioning._ensure_input_feature_dict(
            input_feature_dict, t_hat_noise_level, device=device
        )
        for k, v in processed_input_dict.items():
            if isinstance(v, torch.Tensor):
                if self.debug_logging:
                    self.logger.debug(f"[DEVICE-DEBUG][StageD] processed_input_dict[{k}] device={v.device}")

        # PATCH: Prevent cubic blowup by enforcing strict shape checks
        # x_noisy should be [B, N_sample, N_atom, 3] or [B, 1, N_atom, 3]
        # t_hat_noise_level should be [B, N_sample] or [B, 1]
        # Never allow expansion along atom dimensions
        if self.debug_logging:
            self.logger.debug(f"[SHAPE-DEBUG][StageD] x_noisy.shape: {x_noisy.shape}")
            self.logger.debug(f"[SHAPE-DEBUG][StageD] t_hat_noise_level.shape: {t_hat_noise_level.shape}")
            if s_trunk is not None:
                self.logger.debug(f"[SHAPE-DEBUG][StageD] s_trunk.shape: {getattr(s_trunk, 'shape', None)}")
            if s_inputs is not None:
                self.logger.debug(f"[SHAPE-DEBUG][StageD] s_inputs.shape: {getattr(s_inputs, 'shape', None)}")
            if z_trunk is not None:
                self.logger.debug(f"[SHAPE-DEBUG][StageD] z_trunk.shape: {getattr(z_trunk, 'shape', None)}")
            for k, v in processed_input_dict.items():
                if isinstance(v, torch.Tensor):
                    self.logger.debug(f"[SHAPE-DEBUG][StageD] processed_input_dict['{k}'].shape: {v.shape}")
        if x_noisy.ndim not in (4,):
            raise ValueError(f"[BUG] x_noisy must be 4D [B, N_sample, N_atom, 3], got {x_noisy.shape}")
        B, N_sample, N_atom, C = x_noisy.shape
        if C != 3:
            raise ValueError(f"[BUG] Last dim of x_noisy must be 3 (xyz), got {C}")
        if t_hat_noise_level.shape not in [(B, N_sample), (B, 1)]:
            raise ValueError(
                f"[BUG] t_hat_noise_level shape {t_hat_noise_level.shape} is not broadcastable to [B, N_sample]. "
                f"This would cause cubic blowup. Aborting."
            )
        # Only expand t_hat_noise_level from [B, 1] to [B, N_sample] if needed
        if t_hat_noise_level.shape == (B, 1):
            t_hat_noise_level = t_hat_noise_level.expand(B, N_sample)
        # If already correct shape, do nothing
        # Remove all ambiguous expansion logic below

        # SYSTEMATIC DEBUGGING: Print input_feature_dict keys and atom_to_token_idx state
        if self.debug_logging:
            self.logger.debug(f"[DEBUG][DiffusionModule.forward] input_feature_dict keys: {list(input_feature_dict.keys())}")
            for k, v in input_feature_dict.items():
                self.logger.debug(f"[DEBUG][DiffusionModule.forward] input_feature_dict[{k}]: type={type(v)}, is_tensor={isinstance(v, torch.Tensor)}, shape={getattr(v, 'shape', None)}")
            self.logger.debug(f"[DEBUG][DiffusionModule.forward] atom_to_token_idx: {input_feature_dict.get('atom_to_token_idx', None)}")
        # Defensive: Ensure atom_to_token_idx is present and a Tensor
        if not ("atom_to_token_idx" in input_feature_dict and isinstance(input_feature_dict["atom_to_token_idx"], torch.Tensor)):
            raise ValueError("[DIFFUSION MODULE PATCH] input_feature_dict missing 'atom_to_token_idx' or it is not a Tensor in forward")

        # SYSTEMATIC DEBUGGING: Instrument input_feature_dict for atom_to_token_idx presence/type/shape
        if "atom_to_token_idx" not in input_feature_dict:
            if self.debug_logging:
                self.logger.debug("[DEBUG][DiffusionModule.forward] atom_to_token_idx MISSING in input_feature_dict!")
        else:
            atom_idx_val = input_feature_dict["atom_to_token_idx"]
            if self.debug_logging:
                self.logger.debug(f"[DEBUG][DiffusionModule.forward] atom_to_token_idx type: {type(atom_idx_val)}, shape: {getattr(atom_idx_val, 'shape', None)}")
            if atom_idx_val is None:
                if self.debug_logging:
                    self.logger.debug("[DEBUG][DiffusionModule.forward] atom_to_token_idx is None!")
            elif isinstance(atom_idx_val, torch.Tensor):
                if self.debug_logging:
                    self.logger.debug(f"[DEBUG][DiffusionModule.forward] atom_to_token_idx tensor shape: {atom_idx_val.shape}")
            else:
                if self.debug_logging:
                    self.logger.debug(f"[DEBUG][DiffusionModule.forward] atom_to_token_idx value: {atom_idx_val}")

        # Also instrument f_forward call
        if self.debug_logging:
            self.logger.debug(f"[DEBUG][DiffusionModule.forward] Calling f_forward with input_feature_dict['atom_to_token_idx']: {input_feature_dict.get('atom_to_token_idx', 'MISSING')}")

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
                self.logger.warning(
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
                self.logger.warning(
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

        if self.debug_logging:
            self.logger.debug(f"[DEBUG] Processed x_noisy shape: {x_noisy.shape}")
            self.logger.debug(f"[DEBUG] Processed t_hat_noise_level shape: {t_hat_noise_level.shape}")
        if s_trunk is not None:
            if self.debug_logging:
                self.logger.debug(f"[DEBUG] Processed s_trunk shape: {s_trunk.shape}")
        if s_inputs is not None:
            if self.debug_logging:
                self.logger.debug(f"[DEBUG] Processed s_inputs shape: {s_inputs.shape}")
        if z_trunk is not None:
            if self.debug_logging:
                self.logger.debug(f"[DEBUG] Processed z_trunk shape: {z_trunk.shape}")

        # --- Core Logic ---
        # Calculate EDM scaling factors, ensuring they broadcast correctly
        # t_hat_noise_level should be [B, N_sample] or [B, 1] or [B] or scalar
        c_in, c_skip, c_out = self._calculate_edm_scaling_factors(
            t_hat_noise_level
        )  # sigma is t_hat
        # Reshape factors to broadcast with x_noisy [B, N_sample, N_atom, 3]
        c_in = c_in.view(*c_in.shape, 1, 1)
        c_skip = c_skip.view(*c_skip.shape, 1, 1)
        c_out = c_out.view(*c_out.shape, 1, 1)
        if self.debug_logging:
            self.logger.debug(f"[DEBUG] EDM factors shapes - c_in: {c_in.shape}, c_skip: {c_skip.shape}, c_out: {c_out.shape}")

        # Scale noisy input
        r_noisy = (
            x_noisy * c_in
        )  # Broadcasting: [B, N_sample, N_atom, 3] * [B, N_sample, 1, 1]
        if self.debug_logging:
            self.logger.debug(f"[DEBUG] r_noisy shape after scaling: {r_noisy.shape}")

        # Forward pass through f_theta
        # Cast processed_input_dict to Dict[str, Any] to satisfy type checker
        from typing import Dict, Any, cast
        r_update = self.f_forward(
            r_noisy=r_noisy,
            t_hat_noise_level=t_hat_noise_level,  # Pass potentially [B, N_sample]
            input_feature_dict=cast(Dict[str, Any], processed_input_dict),  # Cast to Dict[str, Any]
            s_inputs=s_inputs,  # Pass potentially [B, N_sample, ...]
            s_trunk=s_trunk,  # Pass potentially [B, N_sample, ...]
            z_trunk=z_trunk,  # Pass potentially [B, N_sample, ...]
            chunk_size=chunk_size,
            inplace_safe=inplace_safe,
        )
        if self.debug_logging:
            self.logger.debug(f"[DEBUG] r_update shape: {r_update.shape}")

        # Apply denoising formula
        # Broadcasting: [B, N_sample, N_atom, 3] * [B, N_sample, 1, 1] + [B, N_sample, N_atom, 3] * [B, N_sample, 1, 1]
        x_denoised = r_noisy * c_skip + r_update * c_out
        if self.debug_logging:
            self.logger.debug(f"[DEBUG] x_denoised shape: {x_denoised.shape}")

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
        if self.debug_logging:
            self.logger.debug(
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
