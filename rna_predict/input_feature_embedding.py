import torch
import torch.nn as nn
from rna_predict.atom_encoder import AtomAttentionEncoder

###############################################################################
# Input Feature Embedder
###############################################################################

class InputFeatureEmbedder(nn.Module):
    """
    High-level module that merges:
      1) Atom-level embeddings from the AtomAttentionEncoder
      2) Additional per-token features (e.g. residue type, MSA profile, deletion statistics)

    The final single-token embedding is then produced by combining these two streams.
    """
    def __init__(self, c_token=384, restype_dim=32, profile_dim=32,
                 c_atom=128, c_pair=32, num_heads=4, num_layers=3, use_optimized=False):
        super().__init__()
        self.atom_encoder = AtomAttentionEncoder(
            c_atom=c_atom, c_pair=c_pair, c_token=c_token,
            num_heads=num_heads, num_layers=num_layers,
            use_optimized=use_optimized
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
            f, trunk_sing=trunk_sing, trunk_pair=trunk_pair, block_index=block_index
        )

        # Merge with extra token-level features.
        restype = f["restype"]           # [N_token, restype_dim]
        profile = f["profile"]           # [N_token, profile_dim]
        deletion_mean = f["deletion_mean"].unsqueeze(-1)  # [N_token, 1]

        extras = torch.cat([restype, profile, deletion_mean], dim=-1)  # [N_token, in_extras]
        extras_emb = self.extra_linear(extras)  # [N_token, c_token]

        single_emb = a_token + extras_emb
        single_emb = self.final_ln(single_emb)
        return single_emb
