import torch.nn as nn

from rna_predict.pipeline.stageB.pairwise.pairformer import PairformerStack


class PairformerWrapper(nn.Module):
    """
    Integrates Protenix's PairformerStack into our pipeline for global pairwise encoding.
    """

    def __init__(self, n_blocks=48, c_z=128, c_s=384, use_checkpoint=False, dropout=0.25):
        super().__init__()
        self.n_blocks = n_blocks
        self.c_z = c_z
        self.c_s = c_s
        self.use_checkpoint = use_checkpoint
        # Map use_checkpoint to blocks_per_ckpt parameter
        blocks_per_ckpt = 1 if use_checkpoint else None
        self.stack = PairformerStack(
            n_blocks=n_blocks,
            c_z=c_z,
            c_s=c_s,
            dropout=dropout,
            blocks_per_ckpt=blocks_per_ckpt
        )

    def forward(self, s, z, pair_mask):
        """
        s: [batch, N, c_s]
        z: [batch, N, N, c_z]
        pair_mask: [batch, N, N]
        returns updated s, z
        """
        s_updated, z_updated = self.stack(s, z, pair_mask)
        return s_updated, z_updated
