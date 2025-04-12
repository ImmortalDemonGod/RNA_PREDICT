"""
Atom manipulation utilities for RNA structure prediction.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import einops
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from rna_predict.pipeline.stageC.mp_nerf.protein_utils import (
    AAS2INDEX,
    AMBIGUOUS,
    INDEX2AAS,
    SUPREME_INFO,
)
from rna_predict.pipeline.stageC.mp_nerf.proteins import scn_cloud_mask


class AtomSelectionOption(Enum):
    """Enum for atom selection options."""

    BACKBONE = "backbone"
    BACKBONE_WITH_OXYGEN = "backbone-with-oxygen"
    BACKBONE_WITH_CBETA = "backbone-with-cbeta"
    BACKBONE_WITH_CBETA_AND_OXYGEN = "backbone-with-cbeta-and-oxygen"
    ALL = "all"

    @classmethod
    def from_string(cls, option_str: str) -> "AtomSelectionOption":
        """Convert string to enum value."""
        option_str = option_str.lower()
        for option in cls:
            if option.value == option_str:
                return option
        # Handle "backbone-only" as an alias for "backbone"
        if option_str == "backbone-only":
            return cls.BACKBONE
        raise ValueError(f"Invalid option: {option_str}")


# Pre-computed atom masks for different selection options
ATOM_MASKS = {
    AtomSelectionOption.BACKBONE: torch.tensor(
        [True, True, True] + [False] * 11, dtype=torch.bool
    ),
    AtomSelectionOption.BACKBONE_WITH_OXYGEN: torch.tensor(
        [True, True, True, True] + [False] * 10, dtype=torch.bool
    ),
    AtomSelectionOption.BACKBONE_WITH_CBETA: torch.tensor(
        [True, True, True, False, True] + [False] * 9, dtype=torch.bool
    ),
    AtomSelectionOption.BACKBONE_WITH_CBETA_AND_OXYGEN: torch.tensor(
        [True, True, True, True, True] + [False] * 9, dtype=torch.bool
    ),
    AtomSelectionOption.ALL: torch.tensor([True] * 14, dtype=torch.bool),
}


@dataclass
class AtomSelectionConfig:
    """Configuration for atom selection."""

    option: Union[str, torch.Tensor, AtomSelectionOption]
    discard_absent: bool = True

    def get_atom_mask(self) -> torch.Tensor:
        """
        Get the atom mask based on the selection option.

        Returns:
            torch.Tensor: Boolean mask of shape (14,) indicating which atoms to select
        """
        # Handle tensor option
        if isinstance(self.option, torch.Tensor):
            return self.option.bool()

        # Convert string to enum if needed
        if isinstance(self.option, str):
            try:
                option_enum = AtomSelectionOption.from_string(self.option)
            except ValueError:
                raise ValueError(
                    f"Invalid option: {self.option}. Available options: backbone, backbone-with-oxygen, "
                    "backbone-with-cbeta, backbone-with-cbeta-and-oxygen, all"
                )
        else:
            option_enum = self.option

        # Return pre-computed mask
        return ATOM_MASKS[option_enum]


def _create_presence_mask(
    scn_seq: Union[List[str], List[torch.Tensor]],
    coords: Optional[torch.Tensor] = None,
    discard_absent: bool = True,
) -> torch.Tensor:
    """
    Create a mask indicating which atoms are present in the structure.

    Args:
        scn_seq: List of sequences in string or tensor format
        coords: Optional coordinates tensor to check for absent atoms
        discard_absent: Whether to discard absent atoms

    Returns:
        torch.Tensor: Boolean mask of shape (batch_size, seq_len, 14)
    """
    # Get mask for each sequence
    present = []
    for i, seq in enumerate(scn_seq):
        pass_x = coords[i] if discard_absent and coords is not None else None

        # Convert tensor sequence to string if needed
        if pass_x is None and isinstance(seq, torch.Tensor):
            seq = "".join([INDEX2AAS[x] for x in seq.cpu().detach().tolist()])

        # Try/except to handle potential shape errors in scn_cloud_mask
        try:
            present.append(scn_cloud_mask(seq, coords=pass_x))
        except Exception:
            # If we get an error, use a simplified approach
            present.append(torch.ones(len(seq), 14, dtype=torch.bool))

    # Stack masks into a batch
    present = torch.stack(present, dim=0).bool()  # Shape: (B, L, 14)
    return present


@dataclass
class SequenceProcessor:
    """Processor for sequence data."""

    mask: torch.Tensor

    def process_string_seq(self, seq: str, batch_idx: int, processor: Callable) -> None:
        """Process a string sequence."""
        for i, char in enumerate(seq):
            processor(self.mask, batch_idx, i, char)

    def process_tensor_seq(
        self, seq: torch.Tensor, batch_idx: int, processor: Callable
    ) -> None:
        """Process a tensor sequence."""
        processor(self.mask, batch_idx, seq)


def _mark_padding_in_string(
    mask: torch.Tensor, batch_idx: int, pos: int, char: str
) -> None:
    """Mark padding in string sequence."""
    if char == "_":
        mask[batch_idx, pos, :] = False


def _mark_padding_in_tensor(
    mask: torch.Tensor, batch_idx: int, seq: torch.Tensor
) -> None:
    """Mark padding in tensor sequence."""
    padding_indices = (seq == AAS2INDEX["_"]).nonzero(as_tuple=True)[0]
    if padding_indices.numel() > 0:
        mask[batch_idx, padding_indices, :] = False


def _handle_padding_in_mask(
    present: torch.Tensor, scn_seq: Union[List[str], List[torch.Tensor]]
) -> torch.Tensor:
    """
    Update the presence mask to handle padding characters.

    Args:
        present: Presence mask of shape (batch_size, seq_len, 14)
        scn_seq: List of sequences in string or tensor format

    Returns:
        torch.Tensor: Updated presence mask
    """
    # Make a copy to avoid modifying the input
    updated_mask = present.clone()

    # Create processor
    processor = SequenceProcessor(updated_mask)

    # Handle each sequence in the batch
    for b, seq_item in enumerate(scn_seq):
        if isinstance(seq_item, str):
            processor.process_string_seq(seq_item, b, _mark_padding_in_string)
        elif isinstance(seq_item, torch.Tensor):
            processor.process_tensor_seq(seq_item, b, _mark_padding_in_tensor)

    return updated_mask


def _mark_glycine_in_string(
    mask: torch.Tensor, batch_idx: int, pos: int, char: str
) -> None:
    """Mark glycine in string sequence."""
    if char == "G":
        mask[batch_idx, pos, 4] = False


def _mark_glycine_in_tensor(
    mask: torch.Tensor, batch_idx: int, seq: torch.Tensor
) -> None:
    """Mark glycine in tensor sequence."""
    glycine_indices = (seq == AAS2INDEX["G"]).nonzero(as_tuple=True)[0]
    if glycine_indices.numel() > 0:
        mask[batch_idx, glycine_indices, 4] = False


def _handle_glycine_special_case(
    mask: torch.Tensor,
    scn_seq: Union[List[str], List[torch.Tensor]],
    option: Union[str, AtomSelectionOption],
) -> torch.Tensor:
    """
    Handle the special case for Glycine which doesn't have a CB atom.

    Args:
        mask: Combined mask of shape (batch_size, seq_len, 14)
        scn_seq: List of sequences in string or tensor format
        option: Selection option

    Returns:
        torch.Tensor: Updated mask with CB removed for Glycine residues
    """
    # Check if we need to handle glycine
    if isinstance(option, str):
        needs_glycine_handling = "backbone-with-cbeta" in option
    else:
        needs_glycine_handling = option in [
            AtomSelectionOption.BACKBONE_WITH_CBETA,
            AtomSelectionOption.BACKBONE_WITH_CBETA_AND_OXYGEN,
        ]

    if not needs_glycine_handling:
        return mask

    # Make a copy to avoid modifying the input
    updated_mask = mask.clone()

    # Create processor
    processor = SequenceProcessor(updated_mask)

    # Handle each sequence in the batch
    for i, seq_item in enumerate(scn_seq):
        if isinstance(seq_item, str):
            processor.process_string_seq(seq_item, i, _mark_glycine_in_string)
        elif isinstance(seq_item, torch.Tensor):
            processor.process_tensor_seq(seq_item, i, _mark_glycine_in_tensor)

    return updated_mask


def _create_token_tensor(aa: str, pad_token_id: int = 0) -> torch.Tensor:
    """
    Create a token tensor for a single amino acid.

    Args:
        aa: Amino acid character
        pad_token_id: Padding token ID

    Returns:
        torch.Tensor: Token tensor of shape (14,)
    """
    if aa == "_":
        # Assign padding token ID for '_'
        mask_array = np.full(14, pad_token_id, dtype=np.int64)
    elif aa in SUPREME_INFO:
        # Get token mask from SUPREME_INFO
        mask_array = np.array(SUPREME_INFO[aa]["atom_token_mask"], dtype=np.int64)
    else:
        # Handle unexpected characters by using padding
        mask_array = np.full(14, pad_token_id, dtype=np.int64)

    return torch.tensor(mask_array, dtype=torch.long)


def scn_atom_embedd(seq_list: List[str]) -> torch.Tensor:
    """
    Convert a list of amino acid sequences to atom-level token embeddings.

    Args:
        seq_list: List of amino acid sequences

    Returns:
        torch.Tensor: Token embeddings for each atom in each sequence, padded to the maximum sequence length. Shape: (batch_size, max_seq_len, 14)
    """
    # Create a list to hold the final 2D tensors for each sequence
    batch_tensors: List[torch.Tensor] = []
    pad_token_id = 0  # Use 0 as padding token ID

    for seq in seq_list:
        # Create a list to hold 1D tensors for the current sequence
        token_tensors: List[torch.Tensor] = []

        # Process each amino acid in the sequence
        for aa in seq:
            token_tensors.append(_create_token_tensor(aa, pad_token_id))

        # Stack the list of 1D tensors into a 2D tensor (seq_len, 14)
        if token_tensors:  # Avoid stacking empty list
            seq_tensor = torch.stack(token_tensors, dim=0)
            batch_tensors.append(seq_tensor)
        else:
            # Handle empty sequence case if necessary, e.g., append an empty tensor
            batch_tensors.append(torch.empty((0, 14), dtype=torch.long))

    # Pad the sequences to the maximum length in the batch (outside the loop)
    # pad_sequence expects (L, *) input, batch_first=True gives (B, L_max, *) output
    padded_batch = pad_sequence(
        batch_tensors, batch_first=True, padding_value=pad_token_id
    )

    return padded_batch.long()


def _swap_coordinates_and_features(
    coords: torch.Tensor, features: Optional[torch.Tensor], idx1: int, idx2: int
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Swap coordinates and features between two atom indices.

    Args:
        coords: Coordinates tensor
        features: Optional features tensor
        idx1: First atom index
        idx2: Second atom index

    Returns:
        Tuple of (updated_coords, updated_features)
    """
    # Make copies to avoid modifying the input
    updated_coords = coords.clone()
    updated_features = features.clone() if features is not None else None

    # Swap coordinates
    updated_coords[idx1], updated_coords[idx2] = (
        updated_coords[idx2],
        updated_coords[idx1],
    )

    # Swap features if provided
    if updated_features is not None:
        updated_features[idx1], updated_features[idx2] = (
            updated_features[idx2],
            updated_features[idx1],
        )

    return updated_coords, updated_features


def _process_symmetric_pair(
    renamed_coors: torch.Tensor,
    renamed_feats: Optional[torch.Tensor],
    res_idx: int,
    pair: List[int],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Process a symmetric atom pair.

    Args:
        renamed_coors: Coordinates tensor
        renamed_feats: Optional features tensor
        res_idx: Residue index
        pair: List of atom indices

    Returns:
        Tuple of (updated_coords, updated_features)
    """
    # Calculate flattened indices for the atoms in the pair for the current residue
    atom_idx1 = res_idx * 14 + pair[0]
    atom_idx2 = res_idx * 14 + pair[1]

    # Check if indices are within bounds
    if atom_idx1 >= renamed_coors.shape[0] or atom_idx2 >= renamed_coors.shape[0]:
        return renamed_coors, renamed_feats

    # Get coordinates for both atoms
    coord1 = renamed_coors[atom_idx1]
    coord2 = renamed_coors[atom_idx2]

    # Skip if either coordinate is all zeros (missing atom)
    if torch.all(coord1 == 0) or torch.all(coord2 == 0):
        return renamed_coors, renamed_feats

    # Calculate distances to determine which atom is closer to its ideal position
    dist1 = torch.norm(coord1)
    dist2 = torch.norm(coord2)

    # Swap coordinates if the second atom is closer to the ideal position
    if dist2 < dist1:
        renamed_coors, renamed_feats = _swap_coordinates_and_features(
            renamed_coors, renamed_feats, atom_idx1, atom_idx2
        )

    return renamed_coors, renamed_feats


def rename_symmetric_atoms(
    pred_coors: torch.Tensor,
    pred_feats: Optional[torch.Tensor] = None,
    seq: Optional[str] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Rename symmetric atoms in the predicted coordinates and features.

    Args:
        pred_coors: Predicted coordinates [num_atoms, 3] (expects single sequence data)
        pred_feats: Optional predicted features [num_atoms, num_features] (expects single sequence data)
        seq: Optional sequence string

    Returns:
        Tuple of (renamed_coors, renamed_feats)
    """
    if seq is None:
        # If no sequence provided, cannot determine ambiguous atoms, return original
        return pred_coors, pred_feats

    # Create a copy of the input tensors to avoid modifying the originals
    renamed_coors = pred_coors.clone()
    renamed_feats = pred_feats.clone() if pred_feats is not None else None

    # Validate input shapes
    num_residues = len(seq)
    num_atoms_expected = num_residues * 14

    if pred_coors.shape[0] != num_atoms_expected:
        print(
            f"Warning: pred_coors shape {pred_coors.shape} mismatch for sequence length {num_residues}"
        )

    # Process each residue
    for res_idx in range(num_residues):
        res_char = seq[res_idx]

        # Skip residues without symmetric atoms
        if res_char not in AMBIGUOUS:
            continue

        # Process each symmetric pair for this residue
        for pair_indices in cast(List[Any], AMBIGUOUS[res_char]["indexs"]):
            # Ensure pair_indices are integers
            pair = list(map(int, pair_indices))

            # Process this symmetric pair
            renamed_coors, renamed_feats = _process_symmetric_pair(
                renamed_coors, renamed_feats, res_idx, pair
            )

    return renamed_coors, renamed_feats


def _process_residue_pairs(
    aa: str, res_idx: int, result: Dict[str, List[Tuple[int, int]]]
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Process symmetric pairs for a single residue.

    Args:
        aa: Amino acid character
        res_idx: Residue index
        result: Current result dictionary

    Returns:
        Dict[str, List[Tuple[int, int]]]: Updated result dictionary
    """
    if aa not in AMBIGUOUS:
        return result

    # Convert residue index to string key
    key = str(res_idx)

    # Get the pairs of atom indices for this residue
    pairs: List[Tuple[int, int]] = []

    # Process each pair
    for pair_indices in cast(List[Any], AMBIGUOUS[aa]["indexs"]):
        # Convert to tuple of integers
        pair_tuple = tuple(map(int, pair_indices))

        # Ensure the tuple has exactly two elements
        if len(pair_tuple) == 2:
            pairs.append(cast(Tuple[int, int], pair_tuple))

    # Add to result if we found any pairs
    if pairs:
        result[key] = pairs

    return result


def get_symmetric_atom_pairs(seq: str) -> Dict[str, List[Tuple[int, int]]]:
    """
    Get symmetric atom pairs for a given sequence.

    Args:
        seq: String of amino acid sequence

    Returns:
        Dictionary mapping residue indices (as strings) to lists of atom index pairs
        that are symmetric within that residue.
    """
    result = {}

    # Process each residue in the sequence
    for i, aa in enumerate(seq):
        result = _process_residue_pairs(aa, i, result)

    return result


def atom_selector(
    scn_seq: Union[List[str], List[torch.Tensor]],
    x: torch.Tensor,
    option: Union[str, torch.Tensor] = "all",
    discard_absent: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns a selection of the atoms in a protein.

    Args:
        scn_seq: (batch, len) sidechainnet format or list of strings
        x: (batch, (len * n_aa), dims) sidechainnet format
        option: one of [torch.tensor, 'backbone-only', 'backbone-with-cbeta',
                'all', 'backbone-with-oxygen', 'backbone-with-cbeta-and-oxygen']
        discard_absent: bool. Whether to discard the points for which
                        there are no labels (bad recordings)

    Returns:
        Tuple of (selected_atoms, mask)
    """
    # Create configuration
    config = AtomSelectionConfig(option=option, discard_absent=discard_absent)

    # Get atom mask based on option
    atom_mask = config.get_atom_mask()

    # Create presence mask
    present = _create_presence_mask(scn_seq, x, discard_absent)

    # Handle padding in mask
    present = _handle_padding_in_mask(present, scn_seq)

    # Combine atom mask with presence mask
    combined_mask = present * atom_mask.unsqueeze(0).unsqueeze(0).bool()

    # Handle special case for Glycine
    combined_mask = _handle_glycine_special_case(combined_mask, scn_seq, option)

    # Reshape mask for selection
    try:
        mask = einops.rearrange(combined_mask, "b l c -> b (l c)")
        return x[mask], mask
    except Exception:
        # If rearrange fails, fall back to a simpler approach
        mask = combined_mask.reshape(combined_mask.shape[0], -1)
        return x[mask], mask
