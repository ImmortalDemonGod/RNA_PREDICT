"""ProtenixIntegration Module for RNA_PREDICT Pipeline Stage B.

This module provides integration with Protenix input embedding components for Stage B/C synergy,
building single-token and pair embeddings from raw features using Hydra configuration.

Configuration Requirements:
    The module expects a Hydra configuration with the following structure:
    - model.stageB.pairformer.protenix_integration:
        - device: Device to run on (cpu, cuda, mps)
        - c_token: Token dimension for embeddings
        - restype_dim: Dimension for residue type embeddings
        - profile_dim: Dimension for profile embeddings
        - c_atom: Atom dimension for embeddings
        - c_pair: Pair dimension for embeddings
        - r_max: Maximum relative position
        - s_max: Maximum sequence separation
        - use_optimized: Whether to use optimized implementation
"""

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
import logging
# Import structured configs
from rna_predict.conf.config_schema import ProtenixIntegrationConfig

# Import Protenix components
from rna_predict.pipeline.stageA.input_embedding.current.embedders import (
    InputFeatureEmbedder as ProtenixInputEmbedder,
)
from rna_predict.pipeline.stageA.input_embedding.current.embedders import (
    RelativePositionEncoding,
)


# Initialize logger for Stage B ProtenixIntegration
logger = logging.getLogger("rna_predict.pipeline.stageB.pairwise.protenix_integration")


class ProtenixIntegration:
    """
    Integrates Protenix input embedding components for Stage B/C synergy.
    Builds single-token (s_inputs) and pair (z_init) embeddings from raw features.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize the ProtenixIntegration class with Hydra configuration.

        Args:
            cfg: Hydra configuration object containing model.stageB.pairformer.protenix_integration section

        Raises:
            ValueError: If required configuration sections are missing
        """
        # Validate that the required configuration sections exist
        if not (hasattr(cfg, "model") and
                hasattr(cfg.model, "stageB") and
                hasattr(cfg.model.stageB, "pairformer") and
                hasattr(cfg.model.stageB.pairformer, "protenix_integration")):
            raise ValueError("Configuration must contain model.stageB.pairformer.protenix_integration section")

        # Extract the protenix_integration config for cleaner access
        protenix_cfg: ProtenixIntegrationConfig = cfg.model.stageB.pairformer.protenix_integration

        # Validate required parameters
        required_params = ["device", "c_token", "c_atom", "c_pair", "r_max", "s_max"]
        for param in required_params:
            if not hasattr(protenix_cfg, param):
                raise ValueError(f"Configuration missing required parameter: {param}")

        # Get device from config
        device_str = protenix_cfg.device
        self.device = torch.device(device_str)
        # Set debug_logging as an instance attribute
        self.debug_logging = False
        if hasattr(cfg, 'model') and hasattr(cfg.model, 'stageB') and hasattr(cfg.model.stageB, 'debug_logging'):
            self.debug_logging = cfg.model.stageB.debug_logging
        if self.debug_logging:
            logger.info(f"[ProtenixIntegration] Using device: {self.device}")

        # Get embedding dimensions from config
        c_token = protenix_cfg.c_token
        c_atom = protenix_cfg.c_atom
        c_pair = protenix_cfg.c_pair
        r_max = protenix_cfg.r_max
        s_max = protenix_cfg.s_max
        if self.debug_logging:
            logger.info(f"[ProtenixIntegration] Using c_token: {c_token}, c_atom: {c_atom}, c_pair: {c_pair}")

        # Store as instance attributes for later use
        self.c_token = c_token
        self.c_atom = c_atom
        self.c_pair = c_pair
        self.r_max = r_max
        self.s_max = s_max

        # Initialize the input embedder using Protenix's InputFeatureEmbedder
        self.input_embedder = ProtenixInputEmbedder(
            c_atom=c_atom,
            c_atompair=c_pair,
            c_token=c_token
        ).to(self.device)

        # Initialize the relative position encoding module to create pair embeddings.
        self.rel_pos_encoding = RelativePositionEncoding(
            r_max=r_max,
            s_max=s_max,
            c_z=c_token,  # using c_token as the dimension for pair embeddings
        ).to(self.device)

    ##@snoop
    def build_embeddings(self, input_features: dict) -> dict:
        """
        Given a dictionary of raw features, produce the following embeddings:
          - s_inputs: Single-token embeddings of shape [N_token, c_token]
          - z_init: Pair embeddings of shape [N_token, N_token, c_token]

        The input_features dictionary must include keys:
         'ref_pos', 'ref_charge', 'ref_element', 'ref_atom_name_chars', 'atom_to_token',
         'restype', 'profile', 'deletion_mean'
         Optionally, 'residue_index' can be provided.
        """
        # Ensure the key "atom_to_token_idx" exists; if missing, set it equal to "atom_to_token"
        if (
            "atom_to_token_idx" not in input_features
            and "atom_to_token" in input_features
        ):
            input_features["atom_to_token_idx"] = input_features["atom_to_token"]

        # Step 1: Generate single-token embeddings using Protenix's InputFeatureEmbedder.

        # Ensure 'ref_mask' exists; if not, create a default mask with ones.
        if "ref_mask" not in input_features:
            n_atom = input_features["ref_pos"].shape[0]
            input_features["ref_mask"] = torch.ones(
                n_atom, dtype=torch.bool, device=input_features["ref_pos"].device
            )

        # Ensure 'ref_space_uid' exists; if missing, create a tensor of zeros matching the shape of ref_pos (excluding the last dimension).
        if "ref_space_uid" not in input_features:
            shape_uid = input_features["ref_pos"].shape[
                :-1
            ]  # Exclude the coordinate dimension
            input_features["ref_space_uid"] = torch.zeros(
                shape_uid, dtype=torch.long, device=input_features["ref_pos"].device
            )

        # Instrumentation: Log feature shapes and sample values for debugging shape mismatches
        if self.debug_logging:
            logger.info("[DEBUG][ProtenixIntegration] Input features summary before embedding:")
            for k, v in input_features.items():
                if isinstance(v, torch.Tensor):
                    logger.info(f"  Feature '{k}': shape={v.shape}, dtype={v.dtype}, sample={v.flatten()[:5]}")
                else:
                    logger.info(f"  Feature '{k}': type={type(v)}")
            # Log configured embedding dimensions
            logger.info(f"[DEBUG][ProtenixIntegration] Configured c_token: {self.c_token}, c_atom: {self.c_atom}, c_pair: {self.c_pair}")

        # Compute total concatenated feature dimension for input to embedder
        concat_dim = 0
        for k, v in input_features.items():
            if isinstance(v, torch.Tensor) and v.dim() == 2 and k != "atom_to_token_idx":
                concat_dim += v.shape[1]
        if self.debug_logging:
            logger.info(f"[DEBUG][ProtenixIntegration] Total concatenated input feature dimension: {concat_dim}")

        # Iterate through each key in input_features to ensure proper dimensions for each feature.
        keys_to_process = list(input_features.keys()) # Create a list to iterate over as dict size might change
        for key in keys_to_process:
            # Skip processing if key was removed (e.g., if atom_to_token was copied to atom_to_token_idx and then removed)
            if key not in input_features:
                continue
            val = input_features[key]

            # If the feature is 1D, unsqueeze to add a second dimension,
            # UNLESS it's the atom_to_token_idx which should remain 1D [N_atom].
            if val.dim() == 1 and key != "atom_to_token_idx":
                val = val.unsqueeze(-1)
                input_features[key] = val
            # If it IS atom_to_token_idx and 1D, leave it as is.

            # Special handling for 'ref_atom_name_chars': ensure it has a fixed length of 256.
            # This needs to happen *after* potential unsqueezing if it was 1D (unlikely but possible).
            if key == "ref_atom_name_chars":
                if val.size(1) < 256:
                    pad_len = 256 - val.size(1)
                    val = F.pad(val, (0, pad_len), "constant", 0)
                elif val.size(1) > 256:
                    raise ValueError(
                        f"ref_atom_name_chars feature has dimension {val.size(1)}, expected <= 256"
                    )
                input_features[key] = val

            # Special handling for 'ref_element': ensure it has a fixed length of 128.
            # This is critical to match the expected input dimension in the atom attention encoder.
            if key == "ref_element":
                if val.size(1) < 128:
                    pad_len = 128 - val.size(1)
                    val = F.pad(val, (0, pad_len), "constant", 0)
                    if self.debug_logging:
                        logger.info(f"[DEBUG][ProtenixIntegration] Padded ref_element from {val.size(1)-pad_len} to 128 dimensions")
                elif val.size(1) > 128:
                    # Truncate to 128 dimensions
                    val = val[:, :128]
                    if self.debug_logging:
                        logger.info("[DEBUG][ProtenixIntegration] Truncated ref_element to 128 dimensions")
                input_features[key] = val

            # Verify that each feature has exactly 2 dimensions,
            # UNLESS it's atom_to_token_idx which should remain 1D.
            if key != "atom_to_token_idx" and val.dim() != 2:
                raise ValueError(
                    f"Expected feature '{key}' to have 2D shape [batch, feat_dim], "
                    f"but got {val.shape}."
                )

        # Add additional debug logging for feature dimensions before embedding
        if self.debug_logging:
            logger.info("[DEBUG][ProtenixIntegration] Feature dimensions before embedding:")
            for key in ["ref_pos", "ref_charge", "ref_mask", "ref_element", "ref_atom_name_chars"]:
                if key in input_features:
                    logger.info(f"  {key}: {input_features[key].shape}")

            # Calculate expected total dimension
            expected_dim = 3 + 1 + 1 + 128 + 256  # ref_pos + ref_charge + ref_mask + ref_element + ref_atom_name_chars
            logger.info(f"[DEBUG][ProtenixIntegration] Expected total feature dimension: {expected_dim}")

        # Generate the single-token embedding (s_inputs) from the processed input features.
        s_inputs = self.input_embedder(input_feature_dict=input_features)
        # --- FIX: Ensure s_inputs is residue-level [N_token, c_token] ---
        # If s_inputs is [batch, N_token, c_token], squeeze batch if batch==1
        if s_inputs.dim() == 3 and s_inputs.size(0) == 1:
            s_inputs = s_inputs.squeeze(0)
        # If s_inputs is [N_token, N_token, c_token], take diagonal (per-residue embedding)
        if s_inputs.dim() == 3 and s_inputs.size(0) == s_inputs.size(1):
            # Take diagonal along first two dims, shape [N_token, c_token]
            s_inputs = s_inputs.diagonal(dim1=0, dim2=1).transpose(0, 1)
        assert s_inputs.dim() == 2, f"Expected s_inputs to be [N_token, c_token], got {s_inputs.shape}"

        # Extract the number of tokens from restype or profile
        if "restype" in input_features:
            restype = input_features["restype"]
            if restype.dim() == 2:
                N_token = restype.size(0)
            else:
                N_token = restype.size(1)
        elif "profile" in input_features:
            profile = input_features["profile"]
            if profile.dim() == 2:
                N_token = profile.size(0)
            else:
                N_token = profile.size(1)
        else:
            # Fallback to atom_to_token if neither restype nor profile is available
            atom_to_token = input_features["atom_to_token"]
            if atom_to_token.dim() == 2:
                N_token = atom_to_token.size(0)
            else:
                N_token = atom_to_token.max().item() + 1

        # Ensure s_inputs has the correct number of tokens
        if s_inputs.dim() == 2:
            if s_inputs.size(0) != N_token:
                # If s_inputs has more tokens than needed, truncate
                if s_inputs.size(0) > N_token:
                    s_inputs = s_inputs[:N_token, :]
                # If s_inputs has fewer tokens than needed, pad with zeros
                else:
                    padding = torch.zeros(
                        (N_token - s_inputs.size(0), s_inputs.size(1)),
                        device=s_inputs.device,
                        dtype=s_inputs.dtype,
                    )
                    s_inputs = torch.cat([s_inputs, padding], dim=0)

        # Step 2: Generate pair embeddings using relative positional encoding.
        # Determine residue indices: use provided 'residue_index' if available, otherwise create a default range.
        if "residue_index" in input_features:
            res_idx = input_features["residue_index"].to(self.device)
            # Only squeeze if shape is [N_token, 1]
            if res_idx.dim() == 2 and res_idx.shape[1] == 1:
                res_idx = res_idx.squeeze(-1)
        else:
            res_idx = torch.arange(N_token, device=self.device)

        # Compute the initial pair embedding (z_init) using the relative position encoding module.
        z_init = self.rel_pos_encoding(
            {
                "asym_id": torch.zeros(N_token, dtype=torch.long, device=self.device),
                "residue_index": res_idx,
                "entity_id": torch.zeros(N_token, dtype=torch.long, device=self.device),
                "sym_id": torch.zeros(N_token, dtype=torch.long, device=self.device),
                "token_index": res_idx,
            }
        )

        # If z_init has 4 dimensions [1, N_token, N_token, c_z], squeeze out the batch dimension
        if z_init.dim() == 4:
            z_init = z_init.squeeze(0)

        # Return the computed single-token and pair embeddings.
        return {"s_inputs": s_inputs, "z_init": z_init}
