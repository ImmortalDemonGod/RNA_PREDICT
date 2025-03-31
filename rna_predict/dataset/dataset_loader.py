import torch
from datasets import load_dataset
from datasets.iterable_dataset import IterableDataset


def stream_bprna_dataset(split: str = "train") -> IterableDataset:
    """
    Stream the bprna-spot dataset from the HF Hub.

    Args:
        split (str): The dataset split to stream, defaults to "train".

    Returns:
        IterableDataset: An iterable dataset object for the specified split.
    """
    ds_iter = load_dataset("multimolecule/bprna-spot", split=split, streaming=True)
    return ds_iter


def build_rna_token_metadata(num_tokens: int, device="cpu"):
    """
    Construct basic token-level metadata for a single-chain RNA.

    Args:
        num_tokens (int): Number of tokens (e.g., residues in the RNA).
        device (str): CPU or GPU device.

    Returns:
        dict[str, torch.Tensor]:
           {
             "asym_id":       shape [num_tokens], all zeros for a single chain
             "residue_index": shape [num_tokens], e.g. 1..N
             "entity_id":     shape [num_tokens], all zeros if single entity
             "sym_id":        shape [num_tokens], all zeros for single chain
             "token_index":   shape [num_tokens], 0..N-1
           }
    """
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


def build_atom_to_token_idx(num_atoms: int, num_tokens: int, device="cpu"):
    """
    Simplest possible mapping: partition 'num_atoms' equally among 'num_tokens'.

    In reality, we might parse the PDB/CIF to know exactly which atoms
    belong to each residue. For demonstration, we assign ~equal block
    sizes of atoms to each token.

    Args:
        num_atoms (int): Total number of atoms.
        num_tokens (int): Total number of tokens (residues).
        device (str): Device.

    Returns:
        torch.Tensor: Shape [num_atoms], mapping each atom i -> token index j.
    """
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


# @snoop
def load_rna_data_and_features(
    rna_filepath: str, device="cpu", override_num_atoms: int | None = None
):
    """
    Example "high-level" routine that loads an RNA structure,
    builds the input feature dictionaries, and returns them for the pipeline.
    This is a placeholder — in real code, parse PDB/CIF properly.

    override_num_atoms (int | None): If provided, force the number of atoms to match partial_coords.

    Returns:
        tuple: (atom_feature_dict, token_feature_dict)
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
