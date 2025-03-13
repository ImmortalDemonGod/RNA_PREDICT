import torch

###############################################################################
# Utility Functions
###############################################################################


def layernorm(x, eps=1e-5):
    """
    Simple layer normalization over the last dimension.
    Non-trainable; replace with nn.LayerNorm(...) if you want trainable params.
    """
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps)


def inverse_squared_dist(delta, eps=1e-8):
    """
    Given relative positions (delta) of shape [..., 3],
    compute the inverse squared distance as 1/(1 + ||delta||^2).
    """
    dist_sq = torch.sum(delta * delta, dim=-1, keepdim=True)
    return 1.0 / (1.0 + dist_sq + eps)


def scatter_mean(src, index, dim_size, dim=0):
    """
    A simple scatter-mean implementation.

    Args:
      src: Tensor of shape [N, c]
      index: Tensor of shape [N] containing integer indices (segment IDs)
      dim_size: Total number of segments (e.g. number of tokens)
    Returns:
      out: Tensor of shape [dim_size, c] containing the per-segment average.
    """
    device = src.device
    c = src.size(-1)
    out = torch.zeros(dim_size, c, device=device)
    counts = torch.zeros(dim_size, device=device)
    for i in range(src.size(0)):
        idx = index[i].item()
        out[idx] += src[i]
        counts[idx] += 1.0
    counts = torch.clamp(counts, min=1.0)
    out = out / counts.unsqueeze(-1)
    return out
