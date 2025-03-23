import torch
from protenix.model.modules.embedders import InputFeatureEmbedder as ProtenixInputEmbedder
from protenix.model.modules.embedders import RelativePositionEncoding

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
            c_token=c_token,
            restype_dim=restype_dim,
            profile_dim=profile_dim,
            c_atom=c_atom,
            c_pair=c_pair,
            num_heads=num_heads,
            num_layers=num_layers,
            use_optimized=use_optimized
        ).to(device)

        # Relative position encoding to initialize pair embeddings
        self.rel_pos_encoding = RelativePositionEncoding(
            r_max=32,
            s_max=2,
            c_z=c_token  # using c_token for pair embedding dimension
        ).to(device)

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
        # 1) single embedding from Protenix’s InputFeatureEmbedder
        s_inputs = self.input_embedder(input_features)

        # 2) pair embedding from relative positions
        if "residue_index" in input_features:
            res_idx = input_features["residue_index"].to(self.device)
        else:
            # fallback: just arange
            N_token = s_inputs.shape[0]
            res_idx = torch.arange(N_token, device=self.device)

        N_token = res_idx.size(0)
        pair_input = res_idx.unsqueeze(0).expand(N_token, -1)  # [N_token, N_token]
        z_init = self.rel_pos_encoding(pair_input)
        return {"s_inputs": s_inputs, "z_init": z_init}