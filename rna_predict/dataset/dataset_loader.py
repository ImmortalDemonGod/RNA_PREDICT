# DEPRECATED: This file is not used by train.py or the main data pipeline. Retained for reference only. Remove if confirmed obsolete.

import torch
from datasets import load_dataset
from datasets.iterable_dataset import IterableDataset


def stream_bprna_dataset(split: str = "train") -> IterableDataset:
    """
    Streams the "bprna-spot" RNA dataset from the Hugging Face Hub as an iterable dataset.
    
    Args:
        split: The dataset split to stream (e.g., "train", "validation", "test"). Defaults to "train".
    
    Returns:
        An IterableDataset object for the specified split, enabling streaming access to the data.
    """
    ds_iter = load_dataset("multimolecule/bprna-spot", split=split, streaming=True)
    return ds_iter


def build_rna_token_metadata(num_tokens: int, device):
    """
    Constructs token-level metadata tensors for a single-chain RNA molecule.
    
    Creates a dictionary of tensors representing metadata for each token (residue) in a single-chain, single-entity RNA. All tensors are allocated on the specified device, which must be provided explicitly.
    
    Args:
        num_tokens: Number of tokens (residues) in the RNA.
        device: Device on which to allocate tensors (must not be "cpu" or None).
    
    Returns:
        A dictionary with the following keys:
            - "asym_id": Tensor of zeros, shape [num_tokens], indicating a single chain.
            - "residue_index": Tensor of consecutive integers from 1 to num_tokens, shape [num_tokens].
            - "entity_id": Tensor of zeros, shape [num_tokens], indicating a single entity.
            - "sym_id": Tensor of zeros, shape [num_tokens], indicating a single chain.
            - "token_index": Tensor of consecutive integers from 0 to num_tokens - 1, shape [num_tokens].
    """
    # Assert device is not hardcoded default
    assert device is not None and device != "cpu", "Device argument must be provided from Hydra config; do not use hardcoded defaults."
    asym_id = torch.zeros((num_tokens,), dtype=torch.long, device=device)
    entity_id = torch.zeros((num_tokens,), dtype=torch.long, device=device)
    sym_id = torch.zeros((num_tokens,), dtype=torch.long, device=device)

    residue_index = torch.arange(start=1, end=num_tokens + 1, device=device)
    token_index = torch.arange(num_tokens, device=device)

    return {
        "asym_id": asym_id,
        "residue_index": residue_index,
        "entity_id": entity_id,
        "sym_id": sym_id,
        "token_index": token_index,
    }


def build_atom_to_token_idx(num_atoms: int, num_tokens: int, device):
    """
    Creates a tensor mapping each atom to a token by partitioning atoms into contiguous blocks.
    
    Each of the `num_atoms` atoms is assigned to one of the `num_tokens` tokens, distributing atoms as evenly as possible. The mapping tensor is allocated on the specified device and has shape `[num_atoms]`, where each entry indicates the token index for the corresponding atom.
    """
    assert device is not None and device != "cpu", "Device argument must be provided from Hydra config; do not use hardcoded defaults."
    atom_to_token = torch.empty((num_atoms,), dtype=torch.long, device=device)
    atoms_per_token = num_atoms // num_tokens
    leftover = num_atoms % num_tokens
    start = 0
    for t_idx in range(num_tokens):
        block_size = atoms_per_token + (1 if t_idx < leftover else 0)
        end = start + block_size
        atom_to_token[start:end] = t_idx
        start = end
    return atom_to_token


def validate_input_features(input_feature_dict: dict):
    """
    Optional checker ensuring required keys exist and have non-empty shapes.
    Raises ValueError if anything is invalid.
    """
    required_atom_keys = ["atom_to_token_idx", "ref_pos", "ref_space_uid"]
    required_token_keys = [
        "asym_id",
        "residue_index",
        "entity_id",
        "sym_id",
        "token_index",
    ]
    for key in required_atom_keys + required_token_keys:
        if key not in input_feature_dict:
            raise ValueError(f"Missing required key: {key}")

    # Basic shape checks
    if (
        input_feature_dict["ref_pos"].ndim != 3
        or input_feature_dict["ref_pos"].shape[-1] != 3
    ):
        raise ValueError("ref_pos must have shape [batch, N_atom, 3].")
    # Additional shape and content checks can be added here if needed.
    return True


# #####@snoop
def load_rna_data_and_features(
    rna_filepath: str, device, override_num_atoms: int | None = None
):
    """
    Simulates loading RNA structure data and generates atom- and token-level feature dictionaries.
    
    This placeholder function creates synthetic atom coordinates and metadata for a single-chain RNA structure, assembling feature dictionaries suitable for downstream processing. The number of atoms defaults to 40 unless overridden. No actual file parsing is performed.
    
    Args:
        rna_filepath: Path to the RNA structure file (not used; included for interface compatibility).
        override_num_atoms: If provided, sets the number of atoms to this value.
    
    Returns:
        A tuple containing:
            - atom_feature_dict: Dictionary of atom-level features and metadata.
            - token_feature_dict: Dictionary of token-level features.
    """
    # Suppose we parse the file and determine we have 40 atoms by default:
    default_num_atoms = 40
    num_atoms = (
        override_num_atoms if override_num_atoms is not None else default_num_atoms
    )

    # For demonstration, we keep a fixed number of tokens.
    num_tokens = 10

    coords = torch.randn((1, num_atoms, 3), device=device)  # [batch=1, num_atoms, 3]
    # Unique ID for each atom
    ref_space_uid = torch.arange(num_atoms, device=device)[None, :]

    # Build token metadata
    token_meta = build_rna_token_metadata(num_tokens, device=device)

    # Build atom-to-token mapping
    atom_to_tok = build_atom_to_token_idx(num_atoms, num_tokens, device=device)
    atom_to_tok = atom_to_tok.unsqueeze(0)  # shape [batch=1, num_atoms]

    # Build the atom-level feature dictionary (only features that should be processed by the atom-level encoder)
    atom_feature_dict = {
        "atom_to_token_idx": atom_to_tok,  # [1, num_atoms]
        "ref_pos": coords,  # [1, num_atoms, 3]
        "ref_space_uid": ref_space_uid,  # [1, num_atoms]
        "ref_charge": torch.zeros((1, num_atoms, 1), device=device),
        "ref_mask": torch.ones((1, num_atoms, 1), device=device),
        "ref_element": torch.zeros((1, num_atoms, 128), device=device),
        "ref_atom_name_chars": torch.zeros((1, num_atoms, 256), device=device),
    }

    # Insert token-level metadata (IDs, etc.) as metadata in the atom dictionary.
    atom_feature_dict["asym_id"] = token_meta["asym_id"].unsqueeze(0)
    atom_feature_dict["residue_index"] = token_meta["residue_index"].unsqueeze(0)
    atom_feature_dict["entity_id"] = token_meta["entity_id"].unsqueeze(0)
    atom_feature_dict["sym_id"] = token_meta["sym_id"].unsqueeze(0)
    atom_feature_dict["token_index"] = token_meta["token_index"].unsqueeze(0)

    # Build a separate dictionary for token-level features that should be concatenated after the atom-level encoding.
    token_feature_dict = {
        "restype": torch.zeros((1, num_tokens, 32), device=device),
        "profile": torch.zeros((1, num_tokens, 32), device=device),
        "deletion_mean": torch.zeros((1, num_tokens, 1), device=device),
    }

    return atom_feature_dict, token_feature_dict
