import torch
from protenix.model.modules.embedders import InputFeatureEmbedder as ProtenixInputEmbedder
from protenix.model.modules.embedders import RelativePositionEncoding
import snoop
import torch.nn.functional as F

class ProtenixIntegration:
    """
    Integrates Protenix input embedding components for Stage B/C synergy.
    Builds single-token (s_inputs) and pair (z_init) embeddings from raw features.
    """

    def __init__(
        self,
        c_token=384,
        restype_dim=32,
        profile_dim=32,
        c_atom=128,
        c_pair=32,
        num_heads=4,
        num_layers=3,
        use_optimized=False,
        device=torch.device("cpu")
    ):
        """
        Args:
          c_token: dimension for single-token embedding
          restype_dim, profile_dim: dims for per-token features
          c_atom, c_pair: channels for atom/pair embeddings
          num_heads, num_layers: attention config
          use_optimized: whether to use optimized logic
          device: torch device
        """
        self.device = device

        self.input_embedder = ProtenixInputEmbedder(
            c_atom=c_atom,
            c_atompair=c_pair,
            c_token=c_token
        ).to(device)

        # Relative position encoding to initialize pair embeddings
        self.rel_pos_encoding = RelativePositionEncoding(
            r_max=32,
            s_max=2,
            c_z=c_token  # using c_token for pair embedding dimension
        ).to(device)

    @snoop
    def build_embeddings(self, input_features: dict) -> dict:
        """
        Given a dict of raw features, produce:
          - s_inputs: [N_token, c_token]
          - z_init: [N_token, N_token, c_token]

        input_features must include:
         'ref_pos', 'ref_charge', 'ref_element', 'ref_atom_name_chars', 'atom_to_token'
         'restype', 'profile', 'deletion_mean'
         Optionally 'residue_index'
        """
        # Ensure the required key "atom_to_token_idx" is present
        if "atom_to_token_idx" not in input_features and "atom_to_token" in input_features:
            input_features["atom_to_token_idx"] = input_features["atom_to_token"]
        # 1) single embedding from Protenix’s InputFeatureEmbedder
        # Ensure ref_mask exists
        if "ref_mask" not in input_features:
            # Here we assume all atoms are present. Adjust if you have actual mask logic.
            n_atom = input_features["ref_pos"].shape[0]
            input_features["ref_mask"] = torch.ones(n_atom, dtype=torch.bool, device=input_features["ref_pos"].device)

        # Before reshaping each feature, ensure it has at least 2 dims:
        for key in input_features.keys():
            val = input_features[key]
            if val.dim() == 1:
                val = val.unsqueeze(-1)
                input_features[key] = val
    
            if key == "ref_atom_name_chars":
                # Ensure shape is [N_atom, 256] by padding if needed
                if val.size(1) < 256:
                    pad_len = 256 - val.size(1)
                    val = F.pad(val, (0, pad_len), "constant", 0)
                elif val.size(1) > 256:
                    raise ValueError(
                        f"ref_atom_name_chars feature has dimension {val.size(1)}, expected <= 256"
                    )
                input_features[key] = val
    
            if val.dim() != 2:
                raise ValueError(
                    f"Expected feature '{key}' to have 2D shape [batch, feat_dim], "
                    f"but got {val.shape}."
                )

        # Now generate the single-token embedding (s_inputs)
        s_inputs = self.input_embedder(input_feature_dict=input_features)
        # Suppose s_inputs is [B, N_token, c_token]. If always B=1, we can squeeze:
        if s_inputs.dim() == 3 and s_inputs.size(0) == 1:
            s_inputs = s_inputs.squeeze(0)

        # 2) pair embedding from relative positions
        if "residue_index" in input_features:
            res_idx = input_features["residue_index"].to(self.device)
        else:
            # fallback: just arange
            N_token = s_inputs.shape[0]
            res_idx = torch.arange(N_token, device=self.device)

        N_token = res_idx.size(0)
        pair_input = res_idx.unsqueeze(0).expand(N_token, -1)  # [N_token, N_token]
        z_init = self.rel_pos_encoding(
            {
                "asym_id":    torch.zeros(N_token, dtype=torch.long, device=self.device),
                "residue_index": res_idx,
                "entity_id":  torch.zeros(N_token, dtype=torch.long, device=self.device),
                "sym_id":     torch.zeros(N_token, dtype=torch.long, device=self.device),
                "token_index": res_idx,
            }
        )
        # The above line is a placeholder example. Adjust your dict keys to match real usage.

        return {"s_inputs": s_inputs, "z_init": z_init}