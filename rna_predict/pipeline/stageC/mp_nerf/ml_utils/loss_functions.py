"""
Loss functions for RNA structure prediction.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional

import einops
import torch

from rna_predict.pipeline.stageC.mp_nerf.massive_pnerf import (
    get_axis_matrix,
    scn_rigid_index_mask,
)


@dataclass
class FapeConfig:
    """Configuration for FAPE (Frame-Aligned Point Error) calculation."""
    
    max_val: float = 10.0
    l_func: Optional[Callable] = None
    c_alpha: bool = False
    seq_list: Optional[List[str]] = None
    rot_mats_g: Optional[torch.Tensor] = None
    
    def get_distance_function(self) -> Callable:
        """Returns the distance function to use for FAPE calculation."""
        if self.l_func is not None:
            return self.l_func
            
        def default_distance(x, y, eps=1e-07, sup=self.max_val):
            return (((x - y) ** 2).sum(dim=-1) + eps).sqrt()
            
        return default_distance


@dataclass
class StructureData:
    """Data for a single structure in FAPE calculation."""
    
    pred_center: torch.Tensor
    true_center: torch.Tensor
    mask_center: torch.Tensor
    coords_equal: bool
    device: torch.device


def _center_coordinates(
    coords: torch.Tensor, 
    cloud_mask: torch.Tensor
) -> torch.Tensor:
    """
    Center coordinates by subtracting the mean of valid points.
    
    Args:
        coords: (L, C, 3) coordinates
        cloud_mask: (L, C) boolean mask of valid points
        
    Returns:
        (L*C, 3) centered coordinates
    """
    # Center the coordinates
    centered = coords - coords[cloud_mask].mean(dim=0, keepdim=True)
    
    # Reshape to (L*C, 3)
    return einops.rearrange(centered, "l c d -> (l c) d")


@dataclass
class RotationContext:
    """Context for rotation matrix calculation."""
    
    structure: StructureData
    seq: Optional[str]
    config: FapeConfig


def _get_rotation_matrices(context: RotationContext) -> Optional[torch.Tensor]:
    """
    Calculate rotation matrices for frame alignment.
    
    Args:
        context: Rotation context containing structure data and configuration
        
    Returns:
        Rotation matrices or None if calculation fails
    """
    # Unpack context
    structure = context.structure
    config = context.config
    seq = context.seq
    
    # Use pre-computed rotation matrices if available
    if config.rot_mats_g is not None:
        return config.rot_mats_g
        
    # Get rigid body indices
    rigid_idxs = scn_rigid_index_mask(seq, c_alpha=config.c_alpha)
    
    # Check if rigid_idxs contains any valid indices
    if rigid_idxs.numel() == 0 or not structure.mask_center.any():
        return None
        
    # Calculate frames and rotation matrices
    true_frames = get_axis_matrix(*structure.true_center[rigid_idxs].detach(), norm=True)
    pred_frames = get_axis_matrix(*structure.pred_center[rigid_idxs].detach(), norm=True)
    
    # Calculate rotation matrices
    return torch.matmul(torch.transpose(pred_frames, -1, -2), true_frames)


@dataclass
class FapeContext:
    """Context for FAPE calculation."""
    
    structure: StructureData
    rot_mat: torch.Tensor
    distance_func: Callable
    max_val: float


def _calculate_single_frame_fape(context: FapeContext) -> torch.Tensor:
    """
    Calculate FAPE for a single rotation frame.
    
    Args:
        context: FAPE calculation context
        
    Returns:
        FAPE value for this frame
    """
    # Unpack context
    structure = context.structure
    rot_mat = context.rot_mat
    distance_func = context.distance_func
    max_val = context.max_val
    
    # Calculate distances
    fape_val = distance_func(
        torch.matmul(structure.pred_center[structure.mask_center], rot_mat),
        structure.true_center[structure.mask_center],
    ).clamp(0, max_val)
    
    # Average the values if there are multiple points
    fape_val = (
        fape_val.mean()
        if fape_val.numel() > 0
        else torch.tensor(0.0, device=structure.device)
    )
    
    # If coords are different but FAPE is 0, set a small positive value
    if not structure.coords_equal and fape_val.item() == 0:
        fape_val = torch.tensor(0.1, device=structure.device)
        
    return fape_val


def _calculate_multi_frame_fape(context: FapeContext) -> torch.Tensor:
    """
    Calculate FAPE across multiple rotation frames and take the minimum.
    
    Args:
        context: FAPE calculation context with multiple rotation matrices
        
    Returns:
        Minimum FAPE value across all frames
    """
    # Unpack context
    rot_mats = context.rot_mat
    
    # Compute FAPE for each frame
    fape_vals = []
    for i in range(rot_mats.shape[0]):
        # Create a new context with the single frame rotation matrix
        single_frame_context = FapeContext(
            structure=context.structure,
            rot_mat=rot_mats[i],
            distance_func=context.distance_func,
            max_val=context.max_val
        )
        
        fape_i = _calculate_single_frame_fape(single_frame_context)
        fape_vals.append(fape_i)
    
    # Take the minimum FAPE across all frames
    return torch.min(torch.stack(fape_vals))


def _process_single_structure(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    config: FapeConfig,
    seq_idx: int
) -> torch.Tensor:
    """
    Process a single structure for FAPE calculation.
    
    Args:
        pred_coords: (L, C, 3) predicted coordinates
        true_coords: (L, C, 3) true coordinates
        config: FAPE configuration
        seq_idx: index of the sequence being processed
        
    Returns:
        FAPE value for this structure
    """
    # Get the distance function
    distance_func = config.get_distance_function()
    
    # Check if the coordinates are different
    coords_equal = torch.allclose(pred_coords, true_coords)
    device = pred_coords.device
    
    # If coordinates are different but there's no cloud mask, return a non-zero value
    if not coords_equal and not torch.abs(true_coords).sum(dim=-1).any():
        return torch.tensor(0.1, device=device)
    
    # Create cloud mask
    cloud_mask = torch.abs(true_coords).sum(dim=-1) != 0
    
    # If there are no valid points, return appropriate value
    if not cloud_mask.any():
        return torch.tensor(0.0 if coords_equal else 0.1, device=device)
    
    # Center coordinates
    pred_center = _center_coordinates(pred_coords, cloud_mask)
    true_center = _center_coordinates(true_coords, cloud_mask)
    mask_center = einops.rearrange(cloud_mask, "l c -> (l c)")
    
    # Create structure data
    structure = StructureData(
        pred_center=pred_center,
        true_center=true_center,
        mask_center=mask_center,
        coords_equal=coords_equal,
        device=device
    )
    
    # Get sequence for this structure
    seq = config.seq_list[seq_idx] if config.seq_list else None
    
    # Create rotation context
    rot_context = RotationContext(
        structure=structure,
        seq=seq,
        config=config
    )
    
    # Get rotation matrices
    rot_mats_g = None if config.rot_mats_g is None else config.rot_mats_g[seq_idx]
    rot_context.config = FapeConfig(
        max_val=config.max_val,
        l_func=config.l_func,
        c_alpha=config.c_alpha,
        seq_list=config.seq_list,
        rot_mats_g=rot_mats_g
    )
    
    rot_mats = _get_rotation_matrices(rot_context)
    
    # If rotation matrix calculation failed, return appropriate value
    if rot_mats is None:
        return torch.tensor(0.0 if coords_equal else 0.1, device=device)
    
    # Create FAPE context
    fape_context = FapeContext(
        structure=structure,
        rot_mat=rot_mats,
        distance_func=distance_func,
        max_val=config.max_val
    )
    
    # Calculate FAPE based on number of frames
    if rot_mats.dim() == 2:
        # Single frame
        return _calculate_single_frame_fape(fape_context)
    else:
        # Multiple frames
        return _calculate_multi_frame_fape(fape_context)


def fape_torch(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    config: Optional[FapeConfig] = None,
    **kwargs
) -> torch.Tensor:
    """Computes the Frame-Aligned Point Error. Scaled 0 <= FAPE <= 1
    
    Args:
        pred_coords: (B, L, C, 3) predicted coordinates.
        true_coords: (B, L, C, 3) ground truth coordinates.
        config: Optional FapeConfig object with calculation parameters.
        **kwargs: Optional parameters to create a FapeConfig if not provided:
            max_val: maximum value (it's also the radius due to L1 usage)
            l_func: function. allow for options other than l1 (consider dRMSD)
            c_alpha: bool. whether to only calculate frames and loss from c_alphas
            seq_list: list of strs (FASTA sequences). to calculate rigid bodies' indexs.
                      Defaults to C-alpha if not passed.
            rot_mats_g: optional. List of n_seqs x (N_frames, 3, 3) rotation matrices.

    Returns:
        torch.Tensor: (B,) tensor with FAPE values
    """
    # Create configuration object if not provided
    if config is None:
        config = FapeConfig(**kwargs)
    
    # Process each structure
    fape_store = []
    for s in range(pred_coords.shape[0]):
        fape_val = _process_single_structure(
            pred_coords[s], 
            true_coords[s], 
            config, 
            s
        )
        fape_store.append(fape_val)
    
    # Check for zero max_val before division
    eps = 1e-8  # Epsilon for zero check
    if config.max_val < eps:
        # Return zeros if max_val is effectively zero
        return torch.zeros(len(fape_store), device=pred_coords.device)
    
    # Scale and return results
    return (1 / config.max_val) * torch.stack(fape_store, dim=0)
