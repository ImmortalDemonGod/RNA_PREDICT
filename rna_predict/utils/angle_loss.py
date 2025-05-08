import torch
import torch.nn.functional as F

def angle_loss(pred_angles: torch.Tensor, target_angles: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Compute the loss between predicted and target angles.
    
    Args:
        pred_angles: Predicted angles tensor of shape (batch_size, seq_len, num_angles)
        target_angles: Target angles tensor of shape (batch_size, seq_len, num_angles)
        mask: Optional mask tensor of shape (batch_size, seq_len) indicating valid positions
        
    Returns:
        Loss value as a scalar tensor
    """
    # Ensure inputs are on the same device
    if pred_angles.device != target_angles.device:
        target_angles = target_angles.to(pred_angles.device)
    
    # Compute MSE loss
    loss = F.mse_loss(pred_angles, target_angles, reduction='none')
    
    # Apply mask if provided
    if mask is not None:
        if mask.device != pred_angles.device:
            mask = mask.to(pred_angles.device)
        # Expand mask to match loss dimensions
        mask = mask.unsqueeze(-1)
        loss = loss * mask
        # Average over masked elements
        return loss.sum() / (mask.sum() + 1e-8)
    
    # Average over all elements if no mask
    return loss.mean() 