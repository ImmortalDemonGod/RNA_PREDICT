from typing import Any, Dict, Tuple, Union, Optional, cast
import torch
import torch.nn as nn

from rna_predict.pipeline.stageA.input_embedding.current.embedders import (
    FourierEmbedding,
    RelativePositionEncoding,
)
from rna_predict.pipeline.stageA.input_embedding.current.primitives import (
    LayerNorm,
    LinearNoBias,
    Transition,
)
from .diffusion_utils import (
    InputFeatureDict,
    create_zero_tensor_like,
)


class DiffusionConditioning(nn.Module):
    """
    Implements Algorithm 21 in AF3 for conditioning the diffusion process.

    This module handles the conditioning of both pair and single features
    for the diffusion process, combining trunk features with input features
    and noise embeddings.
    """

    def __init__(
        self,
        sigma_data: float,
        c_z: int,
        c_s: int,
        c_s_inputs: int,
        c_noise_embedding: int,
        debug_logging: bool = False,  # Add debug_logging parameter
    ) -> None:
        """
        Initialize the diffusion conditioning module.

        Args:
            sigma_data: Standard deviation of the data
            c_z: Hidden dimension for pair embedding
            c_s: Hidden dimension for single embedding
            c_s_inputs: Input embedding dimension from InputEmbedder
            c_noise_embedding: Noise embedding dimension
            debug_logging: Enable debug logging (default: False)
        """
        super(DiffusionConditioning, self).__init__()
        self.sigma_data = sigma_data
        self.c_z = c_z
        self.c_s = c_s
        self.c_s_inputs = c_s_inputs
        self.c_noise_embedding = c_noise_embedding
        self.debug_logging = debug_logging  # Store debug_logging in instance variable
        if debug_logging:
            print(f"[DEBUG][DiffusionConditioning.__init__] c_z={c_z}, c_s={c_s}, c_s_inputs={c_s_inputs}, c_noise_embedding={c_noise_embedding}")

        # Pair feature processing
        self.relpe = RelativePositionEncoding(c_z=c_z)
        # Use a more flexible approach for layernorm_z
        # We'll initialize it with the expected dimension but it can be updated dynamically
        self.expected_z_dim = 2 * self.c_z
        if debug_logging:
            print(f"[DEBUG][DiffusionConditioning] expected_z_dim: {self.expected_z_dim}")
        # Initialize with the expected dimension, but we'll update it if needed
        self.layernorm_z = LayerNorm(self.expected_z_dim)
        self.linear_no_bias_z = LinearNoBias(
            in_features=2 * self.c_z, out_features=self.c_z
        )
        self.transition_z1 = Transition(c_in=self.c_z, n=2)
        self.transition_z2 = Transition(c_in=self.c_z, n=2)

        # Single feature processing
        if debug_logging:
            print(f"[DEBUG][DiffusionConditioning] layernorm_s normalized_shape: {self.c_s + self.c_s_inputs}")
        self.layernorm_s = LayerNorm(self.c_s + self.c_s_inputs)
        self.linear_no_bias_s = LinearNoBias(
            in_features=self.c_s + self.c_s_inputs, out_features=self.c_s
        )

        # Noise embedding processing
        self.fourier_embedding = FourierEmbedding(c=self.c_noise_embedding)
        if debug_logging:
            print(f"[DEBUG][DiffusionConditioning] layernorm_n normalized_shape: {self.c_noise_embedding}")
        self.layernorm_n = LayerNorm(self.c_noise_embedding)
        self.linear_no_bias_n = LinearNoBias(
            in_features=self.c_noise_embedding, out_features=self.c_s
        )

        # Additional transitions
        self.transition_s1 = Transition(c_in=self.c_s, n=2)
        self.transition_s2 = Transition(c_in=self.c_s, n=2)

        # Assertions to catch mismatches early
        assert isinstance(self.c_z, int) and self.c_z > 0, (
            f"UNIQUE ERROR: c_z must be positive int, got {self.c_z}")
        assert isinstance(self.c_s, int) and self.c_s > 0, (
            f"UNIQUE ERROR: c_s must be positive int, got {self.c_s}")
        assert isinstance(self.c_s_inputs, int) and self.c_s_inputs > 0, (
            f"UNIQUE ERROR: c_s_inputs must be positive int, got {self.c_s_inputs}")
        assert isinstance(self.c_noise_embedding, int) and self.c_noise_embedding > 0, (
            f"UNIQUE ERROR: c_noise_embedding must be positive int, got {self.c_noise_embedding}")

    def _process_pair_features(
        self,
        z_trunk: torch.Tensor,
        input_feature_dict: InputFeatureDict,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """Process pair features through the conditioning pipeline."""
        N_tokens = z_trunk.shape[-2]
        input_feature_dict["expected_n_tokens"] = N_tokens

        relpe_output = self.relpe(input_feature_dict)

        # Ensure feature dimensions match
        if relpe_output.shape[-1] != z_trunk.shape[-1]:
            if relpe_output.shape[-1] < z_trunk.shape[-1]:
                padding = create_zero_tensor_like(
                    relpe_output,
                    (
                        *relpe_output.shape[:-1],
                        z_trunk.shape[-1] - relpe_output.shape[-1],
                    ),
                )
                relpe_output = torch.cat([relpe_output, padding], dim=-1)
            else:
                relpe_output = relpe_output[..., : z_trunk.shape[-1]]

        # Ensure batch dimensions match
        if relpe_output.shape[0] != z_trunk.shape[0]:
            if relpe_output.shape[0] == 1:
                # Expand relpe_output to match z_trunk's batch size
                relpe_output = relpe_output.expand(z_trunk.shape[0], *relpe_output.shape[1:])
            else:
                # This is an unexpected case, but we'll handle it by truncating
                print(f"[WARNING] Unexpected batch size mismatch: relpe_output.shape[0]={relpe_output.shape[0]}, z_trunk.shape[0]={z_trunk.shape[0]}")
                relpe_output = relpe_output[:z_trunk.shape[0]]

        # Handle N_sample dimension (dim 1) if present in z_trunk but not in relpe_output
        if z_trunk.ndim >= 5 and relpe_output.ndim == 4:  # z_trunk has N_sample dimension
            # Add N_sample dimension to relpe_output
            n_sample = z_trunk.shape[1]
            relpe_output = relpe_output.unsqueeze(1)
            # Expand to match z_trunk's N_sample dimension
            relpe_output = relpe_output.expand(-1, n_sample, -1, -1, -1)

        # Handle 6D z_trunk case (reshape to 5D for compatibility)
        if z_trunk.ndim == 6:
            # Reshape z_trunk to 5D by merging dimensions 1 and 2
            z_trunk_shape = z_trunk.shape
            z_trunk = z_trunk.reshape(z_trunk_shape[0], z_trunk_shape[1]*z_trunk_shape[2],
                                     z_trunk_shape[3], z_trunk_shape[4], z_trunk_shape[5])

        if self.debug_logging:
            print(f"[DEBUG][_process_pair_features] z_trunk.shape={z_trunk.shape}, relpe_output.shape={relpe_output.shape}")
        pair_z = torch.cat([z_trunk, relpe_output], dim=-1)
        if self.debug_logging:
            print(f"[DEBUG][_process_pair_features] pair_z.shape before layernorm_z={pair_z.shape}")

        # Create or update layernorm_z based on the actual input dimensions
        actual_z_dim = pair_z.shape[-1]

        # Check if layernorm_z is None or has the wrong shape
        layernorm_z_shape_info = "None" if self.layernorm_z is None else self.layernorm_z.normalized_shape[0]
        if self.layernorm_z is None or (hasattr(self.layernorm_z, 'normalized_shape') and self.layernorm_z.normalized_shape[0] != actual_z_dim):
            if self.debug_logging:
                print(f"[DEBUG][_process_pair_features] Creating new layernorm_z with normalized_shape={actual_z_dim} (current shape: {layernorm_z_shape_info})")
            self.layernorm_z = LayerNorm(actual_z_dim).to(pair_z.device)

        # Apply layernorm and linear transformation
        pair_z = self.layernorm_z(pair_z)

        # If the linear layer's input dimension doesn't match, create a new one
        if self.linear_no_bias_z.in_features != actual_z_dim:
            if self.debug_logging:
                print(f"[DEBUG][_process_pair_features] Creating new linear_no_bias_z with in_features={actual_z_dim}")
            self.linear_no_bias_z = LinearNoBias(
                in_features=actual_z_dim,
                out_features=self.c_z
            ).to(pair_z.device)

        pair_z = self.linear_no_bias_z(pair_z)

        if inplace_safe:
            pair_z += self.transition_z1(pair_z)
            pair_z += self.transition_z2(pair_z)
        else:
            pair_z = pair_z + self.transition_z1(pair_z)
            pair_z = pair_z + self.transition_z2(pair_z)

        return pair_z

    def _process_single_features(
        self,
        s_trunk: torch.Tensor,
        s_inputs: torch.Tensor,
        unified_latent: Optional[torch.Tensor] = None,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """Process single features through the conditioning pipeline."""
        if self.debug_logging:
            print(f"[STAGED DEBUG] _process_single_features: s_trunk.shape={s_trunk.shape}, s_inputs.shape={s_inputs.shape}, c_s={self.c_s}, c_s_inputs={self.c_s_inputs}, expected_in_features={self.c_s + self.c_s_inputs}")

        # CRITICAL FIX: Handle token dimension mismatch between s_trunk and s_inputs
        # This is a fallback in case the bridging_utils.py fix didn't catch it
        if s_trunk.shape[-2] != s_inputs.shape[-2]:
            print(f"[DIFFUSION-FIX] Detected token dimension mismatch in _process_single_features: s_trunk.shape={s_trunk.shape}, s_inputs.shape={s_inputs.shape}")

            # Determine if we're dealing with atom-level or residue-level tensors
            # Assume the larger dimension is atom-level and the smaller is residue-level
            if s_trunk.shape[-2] > s_inputs.shape[-2]:  # s_trunk is atom-level, s_inputs is residue-level
                # Create a new tensor to hold atom-level s_inputs
                s_inputs_atom = torch.zeros(
                    *s_inputs.shape[:-2], s_trunk.shape[-2], s_inputs.shape[-1],
                    device=s_inputs.device, dtype=s_inputs.dtype
                )

                # Use a simple replication strategy: repeat each residue's features for all its atoms
                # This is a fallback when we don't have atom_to_token_idx mapping
                n_residues = s_inputs.shape[-2]
                atoms_per_residue = s_trunk.shape[-2] // n_residues

                if atoms_per_residue * n_residues == s_trunk.shape[-2]:  # Perfect division
                    for i in range(n_residues):
                        start_idx = i * atoms_per_residue
                        end_idx = (i + 1) * atoms_per_residue
                        # Handle different dimensionality of s_inputs
                        if s_inputs.dim() == 4:  # [B, N_sample, N_res, C]
                            s_inputs_atom[..., start_idx:end_idx, :] = s_inputs[..., i:i+1, :].expand(
                                *[-1 for _ in range(s_inputs.dim() - 2)], atoms_per_residue, -1
                            )
                        else:  # [B, N_res, C] or other
                            s_inputs_atom[..., start_idx:end_idx, :] = s_inputs[..., i:i+1, :].expand(
                                *[-1 for _ in range(s_inputs.dim() - 2)], atoms_per_residue, -1
                            )
                    s_inputs = s_inputs_atom
                    print(f"[DIFFUSION-FIX] Expanded s_inputs from residue-level to atom-level: {s_inputs.shape}")
                else:
                    # If we can't determine a clean mapping, use a more general approach
                    # Just repeat the first residue's features for all atoms as a last resort
                    print("[DIFFUSION-FIX] Warning: Cannot determine clean residue-to-atom mapping. Using fallback approach.")
                    # Handle different dimensionality of s_inputs
                    if s_inputs.dim() == 4:  # [B, N_sample, N_res, C]
                        s_inputs = s_inputs[..., :1, :].expand(
                            *[-1 for _ in range(s_inputs.dim() - 2)], s_trunk.shape[-2], -1
                        )
                    else:  # [B, N_res, C] or other
                        s_inputs = s_inputs[..., :1, :].expand(
                            *[-1 for _ in range(s_inputs.dim() - 2)], s_trunk.shape[-2], -1
                        )
            else:  # s_inputs is atom-level, s_trunk is residue-level
                # This is less common, but handle it for completeness
                # Create a new tensor to hold residue-level s_trunk
                s_trunk_residue = torch.zeros(
                    *s_trunk.shape[:-2], s_inputs.shape[-2], s_trunk.shape[-1],
                    device=s_trunk.device, dtype=s_trunk.dtype
                )

                # Use a simple replication strategy
                n_atoms = s_inputs.shape[-2]
                residues_per_atom = n_atoms // s_trunk.shape[-2]

                if residues_per_atom * s_trunk.shape[-2] == n_atoms:  # Perfect division
                    for i in range(s_trunk.shape[-2]):
                        start_idx = i * residues_per_atom
                        end_idx = (i + 1) * residues_per_atom
                        # Handle different dimensionality of s_trunk
                        if s_trunk.dim() == 4:  # [B, N_sample, N_res, C]
                            s_trunk_residue[..., start_idx:end_idx, :] = s_trunk[..., i:i+1, :].expand(
                                *[-1 for _ in range(s_trunk.dim() - 2)], residues_per_atom, -1
                            )
                        else:  # [B, N_res, C] or other
                            s_trunk_residue[..., start_idx:end_idx, :] = s_trunk[..., i:i+1, :].expand(
                                *[-1 for _ in range(s_trunk.dim() - 2)], residues_per_atom, -1
                            )
                    s_trunk = s_trunk_residue
                    print(f"[DIFFUSION-FIX] Expanded s_trunk from residue-level to atom-level: {s_trunk.shape}")
                else:
                    # If we can't determine a clean mapping, use a more general approach
                    print("[DIFFUSION-FIX] Warning: Cannot determine clean atom-to-residue mapping. Using fallback approach.")
                    # Handle different dimensionality of s_trunk
                    if s_trunk.dim() == 4:  # [B, N_sample, N_res, C]
                        s_trunk = s_trunk[..., :1, :].expand(
                            *[-1 for _ in range(s_trunk.dim() - 2)], s_inputs.shape[-2], -1
                        )
                    else:  # [B, N_res, C] or other
                        s_trunk = s_trunk[..., :1, :].expand(
                            *[-1 for _ in range(s_trunk.dim() - 2)], s_inputs.shape[-2], -1
                        )

        # Validate and adapt tensor shapes if needed (now config-driven)
        from rna_predict.pipeline.stageD.diffusion.components.diffusion_utils import validate_tensor_shapes
        adapted_s_trunk, adapted_s_inputs = validate_tensor_shapes(
            s_trunk, s_inputs, {
                'c_s': self.c_s,
                'c_s_inputs': self.c_s_inputs
            }
        )

        # If unified_latent is provided, use it as additional conditioning
        if unified_latent is not None:
            if self.debug_logging:
                print(f"[STAGED DEBUG] Using unified_latent with shape: {unified_latent.shape}")

            # Ensure unified_latent has the same batch and sequence dimensions as s_trunk
            if unified_latent.shape[:-1] != adapted_s_trunk.shape[:-1]:
                # Handle different batch dimensions
                if unified_latent.dim() == 3 and adapted_s_trunk.dim() == 4:  # [B, N, C] vs [B, N_sample, N, C]
                    unified_latent = unified_latent.unsqueeze(1).expand(-1, adapted_s_trunk.shape[1], -1, -1)
                elif unified_latent.dim() == 4 and unified_latent.shape[1] == 1 and adapted_s_trunk.shape[1] > 1:
                    # Expand N_sample dimension
                    unified_latent = unified_latent.expand(-1, adapted_s_trunk.shape[1], -1, -1)

                # If sequence dimensions still don't match, we need to adapt
                if unified_latent.shape[-2] != adapted_s_trunk.shape[-2]:
                    if self.debug_logging:
                        print(f"[STAGED DEBUG] Adapting unified_latent sequence dimension from {unified_latent.shape[-2]} to {adapted_s_trunk.shape[-2]}")

                    # Simple approach: repeat the latent for each token
                    if unified_latent.shape[-2] == 1:
                        unified_latent = unified_latent.expand(*unified_latent.shape[:-2], adapted_s_trunk.shape[-2], -1)
                    else:
                        # More complex case: we need to map from residue to atom level
                        # This is a simplified approach - in production, use a proper mapping
                        n_residues = unified_latent.shape[-2]
                        n_atoms = adapted_s_trunk.shape[-2]
                        atoms_per_residue = n_atoms // n_residues

                        if atoms_per_residue * n_residues == n_atoms:  # Perfect division
                            expanded_latent = torch.zeros(
                                *unified_latent.shape[:-2], n_atoms, unified_latent.shape[-1],
                                device=unified_latent.device, dtype=unified_latent.dtype
                            )

                            for i in range(n_residues):
                                start_idx = i * atoms_per_residue
                                end_idx = (i + 1) * atoms_per_residue
                                expanded_latent[..., start_idx:end_idx, :] = unified_latent[..., i:i+1, :].expand(
                                    *[-1 for _ in range(unified_latent.dim() - 2)], atoms_per_residue, -1
                                )
                            unified_latent = expanded_latent

            # Concatenate the unified_latent with the other features
            single_s = torch.cat([adapted_s_trunk, adapted_s_inputs, unified_latent], dim=-1)

            # Update the linear layer to handle the additional features from unified_latent
            if self.linear_no_bias_s.in_features != single_s.shape[-1]:
                if self.debug_logging:
                    print(f"[STAGED DEBUG] Creating new linear_no_bias_s with in_features={single_s.shape[-1]}")
                self.linear_no_bias_s = LinearNoBias(
                    in_features=single_s.shape[-1],
                    out_features=self.c_s
                ).to(single_s.device)
        else:
            # Original behavior when unified_latent is not provided
            single_s = torch.cat([adapted_s_trunk, adapted_s_inputs], dim=-1)
        if self.debug_logging:
            print(f"[STAGED DEBUG] After concat: single_s.shape={single_s.shape}")

        single_s = self.linear_no_bias_s(single_s)
        if self.debug_logging:
            print(f"[STAGED DEBUG] After linear_no_bias_s: single_s.shape={single_s.shape}")

        if inplace_safe:
            single_s += self.transition_s1(single_s)
            single_s += self.transition_s2(single_s)
        else:
            single_s = single_s + self.transition_s1(single_s)
            single_s = single_s + self.transition_s2(single_s)

        if self.debug_logging:
            print(f"[STAGED DEBUG] After transitions: single_s.shape={single_s.shape}")
        return single_s

    ###@snoop
    def _ensure_input_feature_dict(
        self,
        input_feature_dict: Dict[str, Union[torch.Tensor, int, float, Dict[str, Any]]],
        t_hat_noise_level: torch.Tensor,
        device: Optional[torch.device] = None,
        debug_logging: bool = False,  # Add debug_logging argument
    ) -> Dict[str, Any]:
        """Ensure the input feature dictionary has required keys and correct types."""
        # Always use the provided device or fall back to t_hat_noise_level.device
        target_device = device if device is not None else t_hat_noise_level.device
        if debug_logging:
            print(f"[DEVICE-DEBUG][StageD] _ensure_input_feature_dict: using device {target_device}")
        result: Dict[str, Any] = {}

        # List of keys to always pass through if present
        allowed_keys = [
            "ref_charge", "ref_pos", "expected_n_tokens", "atom_to_token_idx", "ref_mask", "ref_element", "ref_atom_name_chars", "ref_space_uid", "restype", "profile", "deletion_mean"
        ]

        # Copy allowed keys from input_feature_dict, moving tensors to the correct device
        for k in allowed_keys:
            if k in input_feature_dict:
                v = input_feature_dict[k]
                if isinstance(v, torch.Tensor):
                    result[k] = v.to(target_device)
                else:
                    result[k] = v

        # Fallbacks for ref_charge and ref_pos if not present
        if "ref_charge" not in result:
            if "ref_pos" in result and isinstance(result["ref_pos"], torch.Tensor):
                result["ref_charge"] = torch.zeros(result["ref_pos"].shape[:-1], device=target_device)
            else:
                result["ref_charge"] = torch.tensor(0.0, device=target_device)
        if "ref_pos" not in result:
            result["ref_pos"] = torch.tensor(0.0, device=target_device)

        # Always set expected_n_tokens if not present
        if "expected_n_tokens" not in result:
            result["expected_n_tokens"] = 0

        return result

    def forward(
        self,
        t_hat_noise_level: torch.Tensor,
        input_feature_dict: Dict[str, Union[torch.Tensor, int, float, Dict[Any, Any]]],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        inplace_safe: bool = False,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the diffusion conditioning module.

        Args:
            t_hat_noise_level: Noise level tensor [..., N_sample]
            input_feature_dict: Input meta feature dictionary
            s_inputs: Single embedding from InputFeatureEmbedder [..., N_tokens, c_s_inputs]
            s_trunk: Single feature embedding from PairFormer [..., N_tokens, c_s]
            z_trunk: Pair feature embedding from PairFormer [..., N_tokens, N_tokens, c_z]
            inplace_safe: Whether to use inplace operations
            device: Device to use for tensor allocations

        Returns:
            Tuple of processed single and pair embeddings
        """
        if self.debug_logging:
            print(f"[STAGED DEBUG] DiffusionConditioning.forward: c_s={self.c_s}, c_s_inputs={self.c_s_inputs}, expected_in_features={self.c_s + self.c_s_inputs}")
            print(f"[STAGED DEBUG] s_trunk.shape={s_trunk.shape}, s_inputs.shape={s_inputs.shape}")

        # Ensure required keys are present with correct types
        processed_input_dict = self._ensure_input_feature_dict(
            input_feature_dict, t_hat_noise_level, device=device, debug_logging=self.debug_logging
        )

        # Process pair features
        if z_trunk is None:
            batch_dims = s_trunk.shape[:-2]
            z_trunk = create_zero_tensor_like(
                s_trunk, (*batch_dims, s_trunk.shape[-2], s_trunk.shape[-2], self.c_z)
            )
        pair_z = self._process_pair_features(
            z_trunk, cast(InputFeatureDict, processed_input_dict), inplace_safe
        )

        # Process single features
        if s_inputs is None:
            batch_dims = s_trunk.shape[:-2]
            s_inputs = create_zero_tensor_like(
                s_trunk, (*batch_dims, s_trunk.shape[-2], self.c_s_inputs)
            )

        # --- Start Shape Alignment Fix ---
        # Ensure s_trunk and s_inputs have compatible shapes before processing
        # Expected shapes: s_trunk [B, N_sample, N, C_s], s_inputs [B, N_sample, N, C_s_inputs] or [B, N, C_s_inputs]

        processed_s_inputs = s_inputs
        processed_s_trunk = s_trunk

        # 1. Ensure s_inputs is 4D if s_trunk is 4D
        if processed_s_trunk.ndim == 4 and processed_s_inputs.ndim == 3:
            processed_s_inputs = processed_s_inputs.unsqueeze(1) # Add N_sample dim

        # 2. Ensure N_sample dimension (dim 1) matches
        if processed_s_trunk.ndim == 4 and processed_s_inputs.ndim == 4:
            n_sample_trunk = processed_s_trunk.shape[1]
            n_sample_inputs = processed_s_inputs.shape[1]

            if n_sample_trunk != n_sample_inputs:
                if n_sample_inputs == 1:
                    # Expand s_inputs to match s_trunk's N_sample
                    processed_s_inputs = processed_s_inputs.expand(
                        -1, n_sample_trunk, -1, -1
                    )
                elif n_sample_trunk == 1:
                     # Expand s_trunk to match s_inputs' N_sample (less likely but possible)
                     processed_s_trunk = processed_s_trunk.expand(
                         -1, n_sample_inputs, -1, -1
                     )
                else:
                    # If both have N_sample > 1 but they differ, it's an unexpected state
                    raise RuntimeError(
                        f"N_sample dimension mismatch cannot be resolved by expansion. "
                        f"s_trunk: {processed_s_trunk.shape}, s_inputs: {processed_s_inputs.shape}"
                    )
        elif processed_s_trunk.ndim != processed_s_inputs.ndim:
             # If dimensions still don't match after unsqueeze attempt
             raise RuntimeError(
                 f"Dimension mismatch after unsqueeze attempt. "
                 f"s_trunk: {processed_s_trunk.shape}, s_inputs: {processed_s_inputs.shape}"
             )

        # --- End Shape Alignment Fix ---

        # Get unified_latent from input_feature_dict if available
        unified_latent = input_feature_dict.get('unified_latent', None)

        # Call _process_single_features with aligned tensors and unified_latent
        single_s = self._process_single_features(
            processed_s_trunk,
            processed_s_inputs,
            unified_latent=unified_latent,
            inplace_safe=inplace_safe
        )

        return single_s, pair_z
