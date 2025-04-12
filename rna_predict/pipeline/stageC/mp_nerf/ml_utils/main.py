"""
Main module for running ml_utils functionality directly.
"""

from typing import List, Optional, Tuple
import einops
import torch

from .coordinate_transforms import (
    CombineNoiseConfig,
    NoiseInternalsConfig,
    combine_noise,
    noise_internals,
)


def _load_protein_data(data_path: str) -> Optional[List]:
    """
    Load protein data from a serialized file.
    
    Args:
        data_path: Path to the serialized protein data file
        
    Returns:
        Loaded protein data or None if loading fails
    """
    import joblib
    
    try:
        prots = joblib.load(data_path)
        return prots
    except FileNotFoundError:
        print(f"Error: Could not find the data file '{data_path}'.")
        print("This script requires a specific data file to run its main logic.")
        return None
    except Exception as e:
        print(f"Error loading data file: {e}")
        return None


def _validate_protein_data(prots: List) -> Optional[Tuple]:
    """
    Validate protein data format and extract the last protein.
    
    Args:
        prots: List of protein data
        
    Returns:
        Tuple of protein data components or None if validation fails
    """
    # Ensure prots is not empty and contains the expected tuple structure
    is_valid = (
        prots and 
        isinstance(prots[-1], tuple) and 
        len(prots[-1]) == 7
    )
    
    if not is_valid:
        print("Error: Loaded data does not have the expected format.")
        return None
        
    # Unpack the last protein
    protein_data = prots[-1]
    seq, int_seq, true_coords, angles, padding_seq, mask, pid = protein_data
    
    # Basic type/shape validation
    if not isinstance(seq, str) or not isinstance(true_coords, torch.Tensor):
        print("Error: Unexpected data types after unpacking.")
        return None
        
    return protein_data


def _prepare_tensors_for_device(
    true_coords: torch.Tensor,
    int_seq: Optional[torch.Tensor],
    angles: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Prepare tensors for processing by moving them to the appropriate device.
    
    Args:
        true_coords: Tensor of coordinates
        int_seq: Optional tensor of integer sequence
        angles: Optional tensor of angles
        
    Returns:
        Tuple of prepared tensors
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move tensors to device
    true_coords = true_coords.unsqueeze(0).to(device)
    
    if angles is not None:
        angles = angles.to(device)
        
    if int_seq is not None:
        int_seq = int_seq.to(device)
        
    return true_coords, int_seq, angles


def _test_noise_internals(
    seq: str,
    true_coords: torch.Tensor,
    angles: Optional[torch.Tensor]
) -> None:
    """
    Test the noise_internals function with the provided data.
    
    Args:
        seq: Amino acid sequence
        true_coords: Tensor of coordinates
        angles: Optional tensor of angles
    """
    try:
        coords_scn = einops.rearrange(true_coords, "b (l c) d -> b l c d", c=14)
        
        # Create configuration
        config = NoiseInternalsConfig(
            seq=seq,
            angles=angles,
            coords=coords_scn[0],
            noise_scale=1.0,
        )
        
        # Call noise_internals with config
        cloud, cloud_mask = noise_internals(config)
        print("cloud.shape", cloud.shape)
    except Exception as e:
        print(f"Error during noise_internals check: {e}")


def _test_combine_noise(
    true_coords: torch.Tensor,
    seq: Optional[str] = None,
    int_seq: Optional[torch.Tensor] = None
) -> None:
    """
    Test the combine_noise function with either sequence or integer sequence data.
    
    Args:
        true_coords: Tensor of coordinates
        seq: Optional amino acid sequence string
        int_seq: Optional integer sequence tensor
    """
    input_type = "seq" if seq is not None else "int_seq"
    
    try:
        # Create configuration
        config = CombineNoiseConfig(
            true_coords=true_coords,
            seq=seq,
            int_seq=int_seq,
            angles=None,
            noise_internals_scale=1e-2,
            sidechain_reconstruct=True,
        )
        
        # Call combine_noise with config
        integral, mask_out = combine_noise(config)
        print(f"integral.shape (with {input_type})", integral.shape)
    except Exception as e:
        print(f"Error during combine_noise check (with {input_type}): {e}")


def _run_main_logic():
    """
    Contains the logic originally in the `if __name__ == '__main__':` block.
    Loads data, sets parameters, and performs checks on noise_internals and combine_noise.
    """
    # Load protein data
    data_path = "some_route_to_local_serialized_file_with_prots"
    prots = _load_protein_data(data_path)
    if not prots:
        return
    
    # Validate and extract protein data
    protein_data = _validate_protein_data(prots)
    if not protein_data:
        return
        
    # Unpack validated protein data
    seq, int_seq, true_coords, angles, padding_seq, mask, pid = protein_data
    
    # Prepare tensors for processing
    true_coords, int_seq, angles = _prepare_tensors_for_device(true_coords, int_seq, angles)
    
    # Run tests
    _test_noise_internals(seq, true_coords, angles)
    _test_combine_noise(true_coords, seq=seq)
    
    if int_seq is not None:
        _test_combine_noise(true_coords, int_seq=int_seq)


if __name__ == "__main__":
    _run_main_logic()
