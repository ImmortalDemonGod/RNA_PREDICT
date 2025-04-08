import torch
import torch.nn.functional as F

from rna_predict.pipeline.stageA.input_embedding.current.embedders import (
    InputFeatureEmbedder as ProtenixInputEmbedder,
)
from rna_predict.pipeline.stageA.input_embedding.current.embedders import (
    RelativePositionEncoding,
)

# This file defines the ProtenixIntegration class, which integrates Protenix input embedding components
# for Stage B/C synergy by building single-token and pair embeddings from raw features.


class ProtenixIntegration:
    """
    Integrates Protenix input embedding components for Stage B/C synergy.
    Builds single-token (s_inputs) and pair (z_init) embeddings from raw features.
    """

    def __init__(
        self,
        c_token=449,
        restype_dim=32,
        profile_dim=32,
        c_atom=128,
        c_pair=32,
        num_heads=4,
        num_layers=3,
        use_optimized=False,
        device=torch.device("cpu"),
    ):
        """
        Initialize the ProtenixIntegration class with embedding and attention configuration.

        Args:
          c_token: dimension for single-token embedding
          restype_dim, profile_dim: dimensions for per-token features (not directly used here)
          c_atom, c_pair: channels for atom and pair embeddings
          num_heads, num_layers: configuration for attention mechanism (not directly used here)
          use_optimized: flag to determine whether to use optimized logic (not directly used here)
          device: torch device to perform computations
        """
        # Store the computation device
        self.device = device

        # Initialize the input embedder using Protenix's InputFeatureEmbedder
        self.input_embedder = ProtenixInputEmbedder(
            c_atom=c_atom, c_atompair=c_pair, c_token=c_token
        ).to(device)

        # Initialize the relative position encoding module to create pair embeddings.
        self.rel_pos_encoding = RelativePositionEncoding(
            r_max=32,
            s_max=2,
            c_z=c_token,  # using c_token as the dimension for pair embeddings
        ).to(device)

    # @snoop
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

            # Verify that each feature has exactly 2 dimensions,
            # UNLESS it's atom_to_token_idx which should remain 1D.
            if key != "atom_to_token_idx" and val.dim() != 2:
                raise ValueError(
                    f"Expected feature '{key}' to have 2D shape [batch, feat_dim], "
                    f"but got {val.shape}."
                )

        # Generate the single-token embedding (s_inputs) from the processed input features.
        s_inputs = self.input_embedder(input_feature_dict=input_features)
        # If s_inputs has a batch dimension of 1, remove it to get shape [N_token, c_token].
        if s_inputs.dim() == 3 and s_inputs.size(0) == 1:
            s_inputs = s_inputs.squeeze(0)

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
