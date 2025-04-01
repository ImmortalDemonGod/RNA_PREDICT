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
        
        # Ensure c_z is a multiple of 16 to satisfy AttentionPairBias constraint
        # This ensures c_a % n_heads == 0 in the underlying AttentionPairBias initialization
        # (where default n_heads is 16 and c_a is derived from c_z)
        self.c_z_adjusted = max(16, ((c_z + 15) // 16) * 16)
        
        # Map use_checkpoint to blocks_per_ckpt parameter
        blocks_per_ckpt = 1 if use_checkpoint else None
        self.stack = PairformerStack(
            n_blocks=n_blocks,
            c_z=self.c_z_adjusted,
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
        # If c_z_adjusted != c_z, need to adapt the input z tensor
        if self.c_z_adjusted != self.c_z:
            # Pad or project z to match c_z_adjusted
            if self.c_z_adjusted > self.c_z:
                # Pad with zeros
                padding = torch.zeros(*z.shape[:-1], self.c_z_adjusted - self.c_z, 
                                     device=z.device, dtype=z.dtype)
                z_adjusted = torch.cat([z, padding], dim=-1)
            else:
                # This case shouldn't happen with our adjustment logic, but for completeness
                z_adjusted = z[..., :self.c_z_adjusted]
        else:
            z_adjusted = z
            
        s_updated, z_updated = self.stack(s, z_adjusted, pair_mask)
        
        # If we adjusted c_z, adjust the output accordingly
        if self.c_z_adjusted != self.c_z:
            z_updated = z_updated[..., :self.c_z]
            
        return s_updated, z_updated
