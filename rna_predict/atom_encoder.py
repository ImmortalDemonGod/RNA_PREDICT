import torch
import torch.nn as nn

from rna_predict.atom_transformer import AtomTransformer
from rna_predict.utils import scatter_mean

###############################################################################
# AtomAttentionEncoder
###############################################################################


class AtomAttentionEncoder(nn.Module):
    """
    Takes raw per-atom features and pairwise features to generate atom embeddings,
    performs local self-attention, and aggregates atom representations into tokens.

    Steps:
      1. Build per-atom embeddings from features such as coordinates, charge, element, and atom name.
      2. Build pairwise embeddings from (e.g.) relative positions and a same–entity indicator.
      3. Optionally add recycled trunk embeddings.
      4. Run local self-attention over atoms.
      5. Aggregate atoms to tokens (using a scatter–mean).
    """

    def __init__(
        self,
        c_atom=128,
        c_pair=32,
        c_token=384,
        num_heads=4,
        num_layers=3,
        use_optimized=False,
    ):
        super().__init__()
        self.c_atom = c_atom
        self.c_pair = c_pair
        self.c_token = c_token

        # (1) Per-atom input: example input dims = pos (3) + charge (1) + element (128) + name (16)
        in_atom_dim = 3 + 1 + 128 + 16
        self.atom_linear = nn.Linear(in_atom_dim, c_atom)

        # (2) Pairwise embedding: example input dims = delta (3) + same_entity (1)
        in_pair_dim = 3 + 1
        self.pair_linear = nn.Linear(in_pair_dim, c_pair)
        self.mlp_pair = nn.Sequential(
            nn.Linear(c_pair, 2 * c_pair), nn.ReLU(), nn.Linear(2 * c_pair, c_pair)
        )

        # (4) Atom transformer: local self-attention among atoms (naive or optimized).
        self.atom_transformer = AtomTransformer(
            c_atom=c_atom,
            num_heads=num_heads,
            num_layers=num_layers,
            use_optimized=use_optimized,
        )

        # (5) Final projection from atom embedding to token embedding.
        self.post_atom_proj = nn.Linear(c_atom, c_token)

    def forward(self, f, trunk_sing=None, trunk_pair=None, block_index=None):
        """
        Args:
          f: dict of Tensors containing:
             - "ref_pos": [N_atom, 3]
             - "ref_charge": [N_atom]
             - "ref_element": [N_atom, 128]
             - "ref_atom_name_chars": [N_atom, 16]
             - "atom_to_token": [N_atom]
             - "restype": [N_token, ...]   (to define the real N_token shape)
          trunk_sing: optional recycled token embedding [N_token, c_atom]
          trunk_pair: optional recycled pair embedding [N_token, N_token, c_pair]
          block_index: [N_atom, block_size] specifying local attention neighbors.
        """
        # Unpack per-atom features.
        pos = f["ref_pos"]  # [N_atom, 3]
        charge = f["ref_charge"].unsqueeze(-1).float()  # [N_atom, 1]
        elem = f["ref_element"]  # [N_atom, 128]
        aname = f["ref_atom_name_chars"]  # [N_atom, 16]
        token_ids = f["atom_to_token"]  # [N_atom]

        # (1) Build the per-atom input.
        x_atom_in = torch.cat([pos, charge, elem, aname], dim=-1)  # [N_atom, 148]
        c_atom0 = self.atom_linear(x_atom_in)  # [N_atom, c_atom]

        # (2) Build pairwise features.
        N_atom = pos.size(0)
        delta = pos.unsqueeze(1) - pos.unsqueeze(0)  # [N_atom, N_atom, 3]
        same_entity = (
            (token_ids.unsqueeze(1) == token_ids.unsqueeze(0)).float().unsqueeze(-1)
        )
        pair_in = torch.cat([delta, same_entity], dim=-1)  # [N_atom, N_atom, 4]
        p_lm = self.pair_linear(pair_in)  # [N_atom, N_atom, c_pair]
        p_lm = self.mlp_pair(p_lm)  # [N_atom, N_atom, c_pair]

        # (3) Optionally incorporate trunk embeddings from previous recycle passes.
        if trunk_sing is not None:
            c_atom0 = c_atom0 + trunk_sing[token_ids]  # [N_atom, c_atom]
        if trunk_pair is not None:
            i_l = token_ids.unsqueeze(-1)  # [N_atom, 1]
            i_m = token_ids.unsqueeze(0)  # [1, N_atom]
            trunk_pair_lm = trunk_pair[i_l, i_m]  # [N_atom, N_atom, c_pair]
            p_lm = p_lm + trunk_pair_lm

        # (4) Run local self-attention over atoms.
        q_atom = self.atom_transformer(c_atom0, p_lm, block_index)  # [N_atom, c_atom]

        # (5) Project and aggregate atoms to tokens — robust approach:
        #     Use the user-supplied number of tokens from f["restype"].size(0)
        #     (or whichever per-token feature is the definitive source).
        q_proj = self.post_atom_proj(q_atom)  # [N_atom, c_token]
        N_token_supplied = f["restype"].size(0)  # e.g. guaranteed shape
        a_token = scatter_mean(q_proj, token_ids, dim_size=N_token_supplied, dim=0)
        # a_token is now [N_token_supplied, c_token] — guaranteed to match extras later.

        return a_token, q_atom, c_atom0, p_lm
