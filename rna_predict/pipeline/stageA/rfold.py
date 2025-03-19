# rna_predict/pipeline/stageA/rfold.py
"""
Fully updated RFold pipeline code, combining:
- RFold_Model (U-Net + Seq2Map attention)
- Row/column argmax "K-Rook" logic
- Base-type constraints
- StageARFoldPredictor class for easy usage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

###############################################################################
# 1) Helper Methods: row/col argmax, base-type constraints, etc.
###############################################################################

def build_distance_mask(length: int, device: torch.device) -> torch.Tensor:
    """
    Returns a [length x length] matrix with 1's for valid pairs
    and 0's for positions i,j where |i-j| < 4 (disallowing short loops).
    """
    mat = torch.ones((length, length), device=device)
    for i in range(length):
        st = max(0, i - 3)
        en = min(length - 1, i + 3)
        for j in range(st, en + 1):
            mat[i, j] = 0.0
    return mat

def constraint_matrix(seq_oh: torch.Tensor) -> torch.Tensor:
    """
    Compute a base-type & distance mask:
      - A–U / U–A / C–G / G–C / U–G / G–U are valid
      - disallow |i-j| < 4
    seq_oh: (batch, length, 4) one-hot for A/U/C/G
    Returns a float mask of shape (batch, length, length).
    """
    device = seq_oh.device
    b, L, _ = seq_oh.shape
    # Slicing: A=0, U=1, C=2, G=3
    base_a = seq_oh[..., 0]
    base_u = seq_oh[..., 1]
    base_c = seq_oh[..., 2]
    base_g = seq_oh[..., 3]

    # Outer products => shape (b, L, L)
    au = torch.einsum('bi,bj->bij', base_a, base_u)  # A–U
    ua = au.transpose(1,2)                           # U–A
    cg = torch.einsum('bi,bj->bij', base_c, base_g)  # C–G
    gc = cg.transpose(1,2)                           # G–C
    ug = torch.einsum('bi,bj->bij', base_u, base_g)  # U–G
    gu = ug.transpose(1,2)                           # G–U

    raw_mask = au + ua + cg + gc + ug + gu  # shape (b, L, L)

    # Also incorporate the "no pairs if within 3 positions"
    dist_mask = build_distance_mask(L, device=device)  # (L,L)
    dist_mask = dist_mask.unsqueeze(0)  # shape (1,L,L)

    final_mask = raw_mask * dist_mask  # broadcast over batch
    return final_mask

def row_col_softmax(y: torch.Tensor) -> torch.Tensor:
    """
    Softmax along rows and columns, then average => "bi-dimensional" distribution
    y: (batch, L, L)
    returns shape (batch, L, L)
    """
    row_sfm = F.softmax(y, dim=-1)  # row direction
    col_sfm = F.softmax(y, dim=-2)  # column direction
    return 0.5 * (row_sfm + col_sfm)

def row_col_argmax(y: torch.Tensor) -> torch.Tensor:
    """
    For each (batch, L, L), picks exactly one max in each row & col 
    by row-wise and col-wise argmax multiplication:
      - row argmax => (b, L)
      - col argmax => (b, L)
    We combine to get 'non-attacking rooks.'
    """
    y_sfm = row_col_softmax(y)
    # Add small noise to avoid ties
    y_perturbed = y_sfm + (torch.randn_like(y_sfm) * 1e-12)

    # Row direction => find argmax along dim=-1
    col_idx = torch.argmax(y_perturbed, dim=-1)  # shape (b, L)
    col_one = torch.zeros_like(y_perturbed).scatter_(
        2, col_idx.unsqueeze(-1), 1.0
    )  # (b, L, L)

    # Column direction => find argmax along dim=-2
    row_idx = torch.argmax(y_perturbed, dim=-2)  # shape (b, L)
    row_one = torch.zeros_like(y_perturbed).scatter_(
        1, row_idx.unsqueeze(1), 1.0
    )  # (b, L, L)

    return row_one * col_one

###############################################################################
# 2) RFold_Model: Seq2Map + U-Net (Encoder/Decoder)
###############################################################################

class ConvBlock(nn.Module):
    """
    Basic 2D conv block with BN + ReLU repeated twice.
    """
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.seq(x)

class UpConv(nn.Module):
    """
    Upsample by factor 2, then conv => shape doubling each dimension.
    """
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ch_in, ch_out, 3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)

class OffsetScale(nn.Module):
    """
    Offset/scale for the self-attention Q/K.
    """
    def __init__(self, dim, heads=2):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta  = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std=0.02)

    def forward(self, x):
        # x shape: (b, l, dim)
        # out => (b, l, heads, dim)
        out = torch.einsum('bld,hd->blhd', x, self.gamma) + self.beta
        return out.unbind(dim=2)  # if heads=2 => returns (q, k)

class Seq2MapAttn(nn.Module):
    """
    The self-attention block from the official code:
    - LN
    - Linear => dim
    - ReLU^2 of the sim
    """
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)
        self.offsetscale = OffsetScale(dim, heads=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (b,l,dim)
        b,l,d = x.shape
        x_ = self.norm(x)
        qk = self.linear(x_)
        q, k = self.offsetscale(qk)  # each => (b,l,d)
        # sim => (b,l,l)
        sim = torch.einsum('bld,bmd->blm', q, k) / l
        attn = F.relu(sim)**2
        return attn

class Seq2Map(nn.Module):
    """
    Map a one-hot-coded sequence [b, L] (with embedding) to a [b,L,L] contact map.
    """
    def __init__(self, vocab_size=4, num_hidden=128, dropout=0.1, max_len=3000):
        super().__init__()
        self.num_hidden = num_hidden
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, num_hidden)
        self.pos_embedding = nn.Embedding(max_len, num_hidden)
        self.scale = nn.Parameter(torch.sqrt(torch.tensor([float(num_hidden)])))
        self.attn = Seq2MapAttn(num_hidden, dropout=dropout)

    def forward(self, seq_idx):
        # seq_idx shape: (b, L)
        b, L = seq_idx.shape
        pos_idx = torch.arange(L, device=seq_idx.device).unsqueeze(0).expand(b, L)

        tok_emb = self.embedding(seq_idx) * self.scale
        pos_emb = self.pos_embedding(pos_idx)
        x = self.dropout(tok_emb + pos_emb)  # shape (b, L, hidden)

        # Self-attn => contact map
        contact_map = self.attn(x)           # (b, L, L)
        return contact_map

class Encoder(nn.Module):
    """
    U-Net Encoder: repeated conv + down-sample.
    e.g. channels [1,32,64,128,256,512]
    """
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(channels)-1):
            if i == 0:
                # first block: no pool
                self.layers.append(ConvBlock(channels[i], channels[i+1]))
            else:
                # pool -> conv
                blk = nn.Sequential(
                    nn.MaxPool2d(2),
                    ConvBlock(channels[i], channels[i+1])
                )
                self.layers.append(blk)

    def forward(self, x):
        skips = []
        for layer in self.layers:
            x = layer(x)
            skips.append(x)
        return x, skips[:-1]

class Decoder(nn.Module):
    """
    U-Net Decoder: up-conv + skip connection.
    e.g. [512,256,128,64,32]
    """
    def __init__(self, channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(len(channels)-1):
            up = UpConv(channels[i], channels[i+1])
            conv = ConvBlock(channels[i+1]*2, channels[i+1])
            self.blocks.append(nn.ModuleList([up, conv]))

    def forward(self, x, skips):
        # skip_list is in ascending depth => reverse for decode
        skips = skips[::-1]
        for i, (up, conv) in enumerate(self.blocks):
            x = up(x)
            x = torch.cat([x, skips[i]], dim=1)
            x = conv(x)
        return x

class RFold_Model(nn.Module):
    """
    Full pipeline from official approach:
      - Seq2Map
      - Gated representation
      - U-Net
      - symmetrical output
    """
    def __init__(self, num_hidden=128, dropout=0.3):
        super().__init__()
        self.num_hidden = num_hidden
        self.dropout = dropout

        # 1) Seq2Map
        self.seq2map = Seq2Map(
            vocab_size=4,
            num_hidden=num_hidden,
            dropout=dropout,
            max_len=3000
        )

        # 2) U-Net
        # Example channel config
        enc_channels = [1,32,64,128,256,512]
        dec_channels = [512,256,128,64,32]
        self.encoder = Encoder(enc_channels)
        self.decoder = Decoder(dec_channels)
        self.readout = nn.Conv2d(32, 1, kernel_size=1, padding=0)

    def forward(self, seq_idx):
        """
        seq_idx shape: (b, L)
        returns shape: (b, L, L)
        """
        b, L = seq_idx.shape
        # Step A: contact map from seq2map
        attn_map = self.seq2map(seq_idx)                # (b, L, L)
        # Step B: gating => multiply by sigmoid
        gated_map = attn_map * torch.sigmoid(attn_map)  # (b, L, L)
        # Step C: feed into U-Net
        x_4d = gated_map.unsqueeze(1)                   # (b,1,L,L)
        latent, skip_list = self.encoder(x_4d)
        dec = self.decoder(latent, skip_list)
        out_4d = self.readout(dec)                      # (b,1,L,L)
        out_2d = out_4d.squeeze(1)                      # (b,L,L)
        # Step D: symmetrical product => final
        out_final = out_2d.transpose(-1, -2) * out_2d   # (b,L,L)
        return out_final

###############################################################################
# 3) StageARFoldPredictor: load config, weights, run inference
###############################################################################

class StageARFoldPredictor:
    """
    Example usage:
      predictor = StageARFoldPredictor(
          config = {"num_hidden":128, "dropout":0.3},
          checkpoint_path = "./checkpoints/RNAStralign_trainset_pretrained.pth"
      )
      adjacency = predictor.predict_adjacency("AUGCUAG...")
    """
    def __init__(self, config: dict, checkpoint_path=None, device=None):
        """
        config: dict with keys e.g. {"num_hidden":128, "dropout":0.3}
        checkpoint_path: path to .pth or .pt file with weights
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        # Build model
        num_hidden = config.get("num_hidden", 128)
        dropout    = config.get("dropout", 0.3)
        self.model = RFold_Model(num_hidden=num_hidden, dropout=dropout)
        self.model.to(device)
        self.model.eval()

        # Optionally load weights
        if checkpoint_path is not None:
            ckp = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(ckp)
            print(f"RFold model weights loaded from {checkpoint_path}")

    def predict_adjacency(self, rna_sequence: str) -> np.ndarray:
        """
        Convert an RNA sequence (string of A/U/C/G) => adjacency [L,L].
        We do row/col argmax + constraint matrix as the final step.
        """
        # A=0,U=1,C=2,G=3
        mapping = {'A':0, 'U':1, 'C':2, 'G':3}
        seq_idx = [mapping.get(n,3) for n in rna_sequence]  # fallback to G=3
        seq_idx_t = torch.tensor(seq_idx, dtype=torch.long, device=self.device).unsqueeze(0)

        with torch.no_grad():
            raw_out = self.model(seq_idx_t)  # shape (1,L,L)

            # row/col argmax
            discrete_map = row_col_argmax(raw_out)

            # build a quick one-hot for constraints
            L = len(seq_idx)
            oh = torch.zeros((1,L,4), device=self.device)
            for i, bcode in enumerate(seq_idx):
                oh[0,i,bcode] = 1.0
            # apply base-type constraints
            base_mask = constraint_matrix(oh)
            final = discrete_map * base_mask

        # final shape => (1,L,L)
        return final[0].cpu().numpy()  # as Numpy