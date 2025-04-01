import torch
import torch.nn as nn

from rna_predict.models.encoder.atom_encoder import AtomAttentionEncoder

###############################################################################
# Input Feature Embedder
###############################################################################


class InputFeatureEmbedder(nn.Module):
    """
    High-level module merging:
      1) Atom-level embeddings from the AtomAttentionEncoder
      2) Extra per-token features (restype, profile, deletion_mean)
      3) (Optional) Pairformer for global pairwise refinement
    The final single-token embedding is the sum of all streams + a final LN.
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
        pairformer_blocks=48,
    ):
        super().__init__()
        import torch.nn as nn

        # We'll get our AtomAttentionEncoder from the same module
        from rna_predict.models.encoder.atom_encoder import (
            AtomAttentionEncoder,
            AtomEncoderConfig,
        )
        from rna_predict.pipeline.stageB.pairwise.pairformer_wrapper import (
            PairformerWrapper,
        )

        self.c_token = c_token
        self.c_pair = c_pair

        # Build the atom encoder
        encoder_config = AtomEncoderConfig(
            c_atom=c_atom,
            c_pair=c_pair,
            c_token=c_token,
            num_heads=num_heads,
            num_layers=num_layers,
            use_optimized=use_optimized,
        )
        self.atom_encoder = AtomAttentionEncoder(encoder_config)

        # Extra token-level linear
        in_extras = restype_dim + profile_dim + 1
        self.extra_linear = nn.Linear(in_extras, c_token)
        self.final_ln = nn.LayerNorm(c_token)

        # Pairformer for optional trunk_pair usage
        self.pairformer = PairformerWrapper(
            n_blocks=pairformer_blocks, c_z=c_pair, c_s=c_token, use_checkpoint=False
        )

    def forward(self, f, trunk_sing=None, trunk_pair=None, block_index=None):
        """
        Args:
          f: dict with:
            per-atom feats: "ref_pos", "ref_charge", "ref_element", ...
            per-token feats: "restype", "profile", "deletion_mean"
          trunk_sing, trunk_pair: optional external embeddings
          block_index: optional neighbor indices

        Returns: single_emb [N_token, c_token]
        """

        a_token, q_atom, c_atom0, p_lm = self.atom_encoder(
            f,
            trunk_sing=trunk_sing,
            trunk_pair=trunk_pair,
            block_index=block_index,
        )

        # Merge extra token-level feats
        restype = f["restype"]
        profile = f["profile"]
        deletion_mean = f["deletion_mean"].unsqueeze(-1)  # [N_token,1]
        extras = torch.cat(
            [restype, profile, deletion_mean], dim=-1
        )  # [N_token,in_extras]
        extras_emb = self.extra_linear(extras)  # [N_token,c_token]

        single_emb = a_token + extras_emb

        # If we have trunk_pair, run Pairformer
        if trunk_pair is not None:
            N = single_emb.size(0)
            device = single_emb.device
            # If no block_index, just use a full mask
            if block_index is None:
                pair_mask = torch.ones((1, N, N), device=device)
            else:
                # We won't do advanced block logic; just unify shape
                pair_mask = torch.ones((1, N, N), device=device)

            s_in = single_emb.unsqueeze(0)  # [1, N, c_token]
            z_in = trunk_pair.unsqueeze(0)  # [1, N, N, c_pair]
            s_out, z_out = self.pairformer(s_in, z_in, pair_mask)
            single_emb = s_out.squeeze(0)

        single_emb = self.final_ln(single_emb)
        return single_emb

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
    ):
        super().__init__()
        from rna_predict.models.encoder.atom_encoder import AtomEncoderConfig

        self.atom_encoder = AtomAttentionEncoder(
            AtomEncoderConfig(
                c_atom=c_atom,
                c_pair=c_pair,
                c_token=c_token,
                num_heads=num_heads,
                num_layers=num_layers,
                use_optimized=use_optimized,
            )
        )
        # Linear layer to embed extra token-level features.
        # For example: restype (32) + profile (32) + deletion_mean (1) = 65.
        in_extras = restype_dim + profile_dim + 1
        self.extra_linear = nn.Linear(in_extras, c_token)
        self.final_ln = nn.LayerNorm(c_token)

    def forward(self, f, trunk_sing=None, trunk_pair=None, block_index=None):
        """
        Args:
          f: dict containing:
             - Per-atom features: "ref_pos", "ref_charge", "ref_element", "ref_atom_name_chars", "atom_to_token"
             - Per-token features: "restype", "profile", "deletion_mean"
          trunk_sing, trunk_pair: optional recycled trunk embeddings
          block_index: [N_atom, block_size] for local attention
        Returns:
          single_emb: [N_token, c_token] – the final token-level embedding.
        """
        a_token, q_atom, c_atom0, p_lm = self.atom_encoder(
            f,
            trunk_sing=trunk_sing,
            trunk_pair=trunk_pair,
            block_index=block_index,
        )

        # Merge with extra token-level features.
        restype = f["restype"]  # [N_token, restype_dim]
        profile = f["profile"]  # [N_token, profile_dim]
        deletion_mean = f["deletion_mean"].unsqueeze(-1)  # [N_token, 1]

        extras = torch.cat(
            [restype, profile, deletion_mean], dim=-1
        )  # [N_token, in_extras]
        extras_emb = self.extra_linear(extras)  # [N_token, c_token]

        single_emb = a_token + extras_emb
        single_emb = self.final_ln(single_emb)
        return single_emb
