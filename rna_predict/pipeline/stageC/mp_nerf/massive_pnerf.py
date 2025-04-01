import numpy as np

# diff ml
import torch


def get_axis_matrix(a, b, c, norm=True):
    """Gets an orthonomal basis as a matrix of [e1, e2, e3].
    Useful for constructing rotation matrices between planes
    according to the first answer here:
    https://math.stackexchange.com/questions/1876615/rotation-matrix-from-plane-a-to-b
    Inputs:
    * a: (batch, 3) or (3, ). point(s) of the plane
    * b: (batch, 3) or (3, ). point(s) of the plane
    * c: (batch, 3) or (3, ). point(s) of the plane
    Outputs: orthonomal basis as a matrix of [e1, e2, e3]. calculated as:
        * e1_ = (c-b)
        * e2_proto = (b-a)
        * e3_ = e1_ ^ e2_proto
        * e2_ = e3_ ^ e1_
        * basis = normalize_by_vectors( [e1_, e2_, e3_] )
    Note: Could be done more by Grahm-Schmidt and extend to N-dimensions
          but this is faster and more intuitive for 3D.
    """
    v1_ = c - b
    v2_ = b - a
    v3_ = torch.cross(v1_, v2_, dim=-1)
    v2_ready = torch.cross(v3_, v1_, dim=-1)
    basis = torch.stack([v1_, v2_ready, v3_], dim=-2)
    # normalize if needed
    if norm:
        # Add small epsilon to avoid division by zero
        norm_values = torch.norm(basis, dim=-1, keepdim=True)
        # Clamp to avoid division by zero
        norm_values = torch.clamp(norm_values, min=1e-10)
        return basis / norm_values
    return basis


def mp_nerf_torch(a, b, c, l, theta, chi):
    """Custom Natural extension of Reference Frame.
    Inputs:
    * a: (batch, 3) or (3,). point(s) of the plane, not connected to d
    * b: (batch, 3) or (3,). point(s) of the plane, not connected to d
    * c: (batch, 3) or (3,). point(s) of the plane, connected to d
    * l: (batch,) or (float). bond length(s) between c-d
    * theta: (batch,) or (float).  angle(s) between b-c-d
    * chi: (batch,) or float. dihedral angle(s) between the a-b-c and b-c-d planes
    Outputs: d (batch, 3) or (float). the next point in the sequence, linked to c
    """
    # safety check
    if not ((-np.pi <= theta) * (theta <= np.pi)).all().item():
        # Clamp theta to valid range instead of raising error
        theta = torch.clamp(theta, -np.pi, np.pi)
    
    # calc vecs
    ba = b - a
    cb = c - b
    
    # Check for zero magnitude vectors and add small perturbation if needed
    ba_norm = torch.norm(ba, dim=-1)
    cb_norm = torch.norm(cb, dim=-1)
    
    if (ba_norm < 1e-10).any() or (cb_norm < 1e-10).any():
        # Add small perturbation to avoid zero vectors
        perturb = torch.tensor([1e-10, 1e-10, 1e-10], device=ba.device)
        if (ba_norm < 1e-10).any():
            ba = ba + perturb
        if (cb_norm < 1e-10).any():
            cb = cb + perturb
    
    # calc rotation matrix. based on plane normals and normalized
    n_plane = torch.cross(ba, cb, dim=-1)
    
    # Check if cross product resulted in zero vector (collinear ba and cb)
    n_plane_norm = torch.norm(n_plane, dim=-1)
    if (n_plane_norm < 1e-10).any():
        # Add small perturbation to ba to avoid collinearity
        perturb = torch.tensor([0.0, 1e-10, 1e-10], device=ba.device)
        ba = ba + perturb
        # Recalculate cross product
        n_plane = torch.cross(ba, cb, dim=-1)
    
    n_plane_ = torch.cross(n_plane, cb, dim=-1)
    rotate = torch.stack([cb, n_plane_, n_plane], dim=-1)
    
    # Safe normalization with epsilon to avoid division by zero
    norm = torch.norm(rotate, dim=-2, keepdim=True)
    norm = torch.clamp(norm, min=1e-10)  # Ensure no division by zero
    rotate = rotate / norm
    
    # calc proto point, rotate. add (-1 for sidechainnet convention)
    # https://github.com/jonathanking/sidechainnet/issues/14
    d = torch.stack(
        [
            -torch.cos(theta),
            torch.sin(theta) * torch.cos(chi),
            torch.sin(theta) * torch.sin(chi),
        ],
        dim=-1,
    ).unsqueeze(-1)
    
    # extend base point, set length
    # Handle both tensor and float inputs for l
    if isinstance(l, torch.Tensor):
        return c + l.unsqueeze(-1) * torch.matmul(rotate, d).squeeze()
    else:
        # If l is a float, we don't need to unsqueeze it
        return c + l * torch.matmul(rotate, d).squeeze()
