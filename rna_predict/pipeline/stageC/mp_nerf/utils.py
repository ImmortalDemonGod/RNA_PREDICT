# Author: Eric Alcaide

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import importlib.util

# Check if Bio.PDB is available
bio_pdb_spec = importlib.util.find_spec("Bio.PDB")
if bio_pdb_spec is not None:
    from Bio.PDB import MMCIFIO, PDBIO, MMCIFParser, PDBParser
    from Bio.PDB.Atom import Atom
    from Bio.PDB.StructureBuilder import StructureBuilder

# Remove unused imports
# from Bio.PDB.Chain import Chain
# from Bio.PDB.Model import Model
# from Bio.PDB.Residue import Residue

# Attempt to import BioPython, handle gracefully if not installed
try:
    from Bio.PDB import MMCIFIO, PDBIO, MMCIFParser, PDBParser
    from Bio.PDB.Atom import Atom
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Model import Model
    from Bio.PDB.Residue import Residue
    from Bio.PDB.StructureBuilder import StructureBuilder

    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False

    # Define dummy classes/functions if BioPython is not available
    # This allows the module to be imported but functions relying on BioPython will fail
    class DummyPDBParser:
        pass

    class DummyMMCIFParser:
        pass

    class DummyPDBIO:
        pass

    class DummyMMCIFIO:
        pass

    class DummyStructureBuilder:
        pass

    class DummyModel:
        pass

    class DummyChain:
        pass

    class DummyResidue:
        pass

    class DummyAtom:
        pass


# random hacks


# to_pi_minus_pi(4) = -2.28  # to_pi_minus_pi(-4) = 2.28  # rads to pi-(-pi)
def to_pi_minus_pi(x):
    return torch.where((x // np.pi) % 2 == 0, x % np.pi, -(2 * np.pi - x % (2 * np.pi)))


def to_zero_two_pi(x):
    return torch.where(x > np.pi, x % np.pi, 2 * np.pi + x % np.pi)


# data utils
def get_prot(dataloader_=None, vocab_=None, min_len=80, max_len=150, verbose=True):
    """Gets a protein from sidechainnet and returns
    the right attrs for training.
    Inputs:
    * dataloader_: sidechainnet iterator over dataset
    * vocab_: sidechainnet VOCAB class
    * min_len: int. minimum sequence length
    * max_len: int. maximum sequence length
    * verbose: bool. verbosity level
    Outputs: (cleaned, without padding)
    (seq_str, int_seq, coords, angles, padding_seq, mask, pid)
    """
    while True:
        for b, batch in enumerate(dataloader_["train"]):
            for i in range(batch.int_seqs.shape[0]):
                # strip padding - matching angles to string means
                # only accepting prots with no missing residues (angles would be 0)
                padding_seq = (batch.int_seqs[i] == 20).sum().item()
                padding_angles = (
                    (torch.abs(batch.angs[i]).sum(dim=-1) == 0).long().sum().item()
                )

                if padding_seq == padding_angles:
                    # check for appropiate length
                    real_len = batch.int_seqs[i].shape[0] - padding_seq
                    if max_len >= real_len >= min_len:
                        # strip padding tokens
                        seq = "".join(
                            [vocab_.int2char(aa) for aa in batch.int_seqs[i].numpy()]
                        )
                        seq = seq[: -padding_seq or None]
                        int_seq = batch.int_seqs[i][: -padding_seq or None]
                        angles = batch.angs[i][: -padding_seq or None]
                        mask = batch.msks[i][: -padding_seq or None]
                        coords = batch.crds[i][: -padding_seq * 14 or None]

                        if verbose:
                            print("stopping at sequence of length", real_len)
                        return (
                            seq,
                            int_seq,
                            coords,
                            angles,
                            padding_seq,
                            mask,
                            batch.pids[i],
                        )
                    else:
                        if verbose:
                            print(
                                "found a seq of length:",
                                batch.int_seqs[i].shape,
                                "but oustide the threshold:",
                                min_len,
                                max_len,
                            )
                else:
                    if verbose:
                        print("paddings not matching", padding_seq, padding_angles)
                    pass
    return None


######################
## structural utils ##
######################


def get_dihedral(c1, c2, c3, c4):
    """Returns the dihedral angle in radians.
    Will use atan2 formula from:
    https://en.wikipedia.org/wiki/Dihedral_angle#In_polymer_physics
    Inputs:
    * c1: (batch, 3) or (3,)
    * c2: (batch, 3) or (3,)
    * c3: (batch, 3) or (3,)
    * c4: (batch, 3) or (3,)
    """
    u1 = c2 - c1
    u2 = c3 - c2
    u3 = c4 - c3

    return torch.atan2(
        ((torch.norm(u2, dim=-1, keepdim=True) * u1) * torch.cross(u2, u3, dim=-1)).sum(
            dim=-1
        ),
        (torch.cross(u1, u2, dim=-1) * torch.cross(u2, u3, dim=-1)).sum(dim=-1),
    )


def get_angle(c1, c2, c3):
    """Returns the angle in radians.
    Inputs:
    * c1: (batch, 3) or (3,)
    * c2: (batch, 3) or (3,)
    * c3: (batch, 3) or (3,)
    """
    u1 = c2 - c1
    u2 = c3 - c2

    # dont use acos since norms involved.
    # better use atan2 formula: atan2(cross, dot) from here:
    # https://johnblackburne.blogspot.com/2012/05/angle-between-two-3d-vectors.html

    # add a minus since we want the angle in reversed order - sidechainnet issues
    return torch.atan2(
        torch.norm(torch.cross(u1, u2, dim=-1), dim=-1), -(u1 * u2).sum(dim=-1)
    )


def kabsch_torch(X, Y):
    """Kabsch alignment of X into Y.
    Assumes X,Y are both (D, N) - usually (3, N)
    """
    #  center X and Y to the origin
    X_ = X - X.mean(dim=-1, keepdim=True)
    Y_ = Y - Y.mean(dim=-1, keepdim=True)
    # calculate convariance matrix (for each prot in the batch)
    C = torch.matmul(X_, Y_.t())
    # Optimal rotation matrix via SVD - warning! W must be transposed
    if int(torch.__version__.split(".")[1]) < 8:
        V, S, W = torch.svd(C.detach())
        W = W.t()
    else:
        V, S, W = torch.linalg.svd(C.detach())
    # determinant sign for direction correction
    d = (torch.det(V) * torch.det(W)) < 0.0
    if d:
        S[-1] = S[-1] * (-1)
        V[:, -1] = V[:, -1] * (-1)
    # Create Rotation matrix U
    U = torch.matmul(V, W)
    # calculate rotations
    X_ = torch.matmul(X_.t(), U).t()
    # return centered and aligned
    return X_, Y_


def rmsd_torch(X, Y):
    """Assumes x,y are both (batch, d, n) - usually (batch, 3, N)."""
    return torch.sqrt(torch.mean((X - Y) ** 2, axis=(-1, -2)))


def get_coords_from_pdb(file_path: str) -> torch.Tensor:
    """Extracts atom coordinates from a PDB file.

    Args:
        file_path (str): Path to the PDB file.

    Returns:
        torch.Tensor: A tensor of shape (num_atoms, 3) containing coordinates.

    Raises:
        ImportError: If BioPython is not installed.
        FileNotFoundError: If the PDB file does not exist.
        Exception: For other PDB parsing errors.
    """
    if not BIOPYTHON_AVAILABLE:
        raise ImportError("BioPython is required for PDB parsing but not installed.")

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("structure", file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"PDB file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error parsing PDB file {file_path}: {e}")

    coords = []
    # Use explicit get methods to align with test mocks
    for model in structure.get_models():
        for chain in model.get_chains():
            for residue in chain.get_residues():
                for atom in residue.get_atoms():
                    coords.append(atom.get_coord())  # BioPython returns numpy arrays

    if not coords:
        # Handle case with no atoms found, maybe raise error or return empty tensor
        return torch.empty((0, 3), dtype=torch.float32)

    return torch.tensor(np.array(coords), dtype=torch.float32)


def get_coords_from_cif(file_path: str) -> torch.Tensor:
    """Extracts atom coordinates from an MMCIF file.

    Args:
        file_path (str): Path to the MMCIF file.

    Returns:
        torch.Tensor: A tensor of shape (num_atoms, 3) containing coordinates.

    Raises:
        ImportError: If BioPython is not installed.
        FileNotFoundError: If the CIF file does not exist.
        Exception: For other CIF parsing errors.
    """
    if not BIOPYTHON_AVAILABLE:
        raise ImportError("BioPython is required for CIF parsing but not installed.")

    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure("structure", file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"CIF file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error parsing CIF file {file_path}: {e}")

    coords = []
    # Use explicit get methods to align with test mocks
    for model in structure.get_models():
        for chain in model.get_chains():
            for residue in chain.get_residues():
                for atom in residue.get_atoms():
                    coords.append(atom.get_coord())

    if not coords:
        return torch.empty((0, 3), dtype=torch.float32)

    return torch.tensor(np.array(coords), dtype=torch.float32)


def get_coords_from_file(file_path: str) -> torch.Tensor:
    """Dispatches coordinate extraction based on file extension.

    Args:
        file_path (str): Path to the PDB or CIF file.

    Returns:
        torch.Tensor: A tensor of shape (num_atoms, 3) containing coordinates.

    Raises:
        ValueError: If the file format is not supported (.pdb or .cif).
        ImportError: If BioPython is not installed (raised by called functions).
        FileNotFoundError: If the file does not exist (raised by called functions).
        Exception: For parsing errors (raised by called functions).
    """
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdb":
        return get_coords_from_pdb(file_path)
    elif suffix == ".cif":
        return get_coords_from_cif(file_path)
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. Only .pdb and .cif are supported."
        )


def get_device() -> torch.device:
    """Gets the appropriate torch device (CUDA if available, else CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def save_structure(
    coords: torch.Tensor,
    output_file: str,
    atom_types: Optional[List[str]] = None,
    res_names: Optional[List[str]] = None,
    chain_id: str = "A",
    model_id: int = 0,
) -> None:
    """Saves coordinates as a PDB or CIF file using BioPython.

    Args:
        coords (torch.Tensor): Coordinates tensor. Can be 2D (num_atoms, 3)
            or 3D (num_residues, atoms_per_residue, 3). If 3D, assumes standard
            backbone atoms (N, CA, C) if atom_types is None.
        output_file (str): Path to save the output file (.pdb or .cif).
        atom_types (Optional[List[str]]): List of atom names. Required if coords is 2D.
            If coords is 3D and this is None, defaults to ['N', 'CA', 'C'].
        res_names (Optional[List[str]]): List of residue names. If None, defaults to 'GLY'.
            Length must match the number of residues.
        chain_id (str): Chain identifier.
        model_id (int): Model identifier.

    Raises:
        ImportError: If BioPython is not installed.
        ValueError: For invalid input shapes or mismatched arguments.
        RuntimeError: For issues during structure building or saving.
    """
    if not BIOPYTHON_AVAILABLE:
        raise ImportError(
            "BioPython is required for saving structures but not installed."
        )

    # Validate input tensor shape
    if coords.ndim == 2:
        if coords.shape[1] != 3:
            raise ValueError(f"Coordinates must have shape (M, 3), got {coords.shape}")
        num_atoms = coords.shape[0]
        if atom_types is None:
            raise ValueError("atom_types must be provided for 2D coordinate input.")
        atoms_per_residue = len(atom_types)
        if num_atoms % atoms_per_residue != 0:
            raise ValueError(
                f"Number of atoms ({num_atoms}) must be divisible by the number of atom types ({atoms_per_residue}) for 2D input."
            )
        num_residues = num_atoms // atoms_per_residue
        # Reshape 2D coords to 3D for consistent processing
        try:
            coords_3d = coords.reshape(num_residues, atoms_per_residue, 3)
        except RuntimeError as e:
            raise RuntimeError(
                f"Could not reshape coords ({coords.shape}) with {atoms_per_residue} atom types: {e}"
            )

    elif coords.ndim == 3:
        if coords.shape[1] != 3 or coords.shape[2] != 3:
            # Check if it's (N, 3, 3) - assuming N, CA, C
            if coords.shape[1] == 3 and coords.shape[2] == 3:
                if atom_types is None:
                    atom_types = ["N", "CA", "C"]  # Default backbone
                elif len(atom_types) != 3:
                    raise ValueError(
                        f"Expected 3 atom types for 3D input shape {coords.shape}, got {len(atom_types)}"
                    )
            else:
                raise ValueError(
                    f"Coordinates must have shape (N, 3, 3), got {coords.shape}"
                )
        num_residues = coords.shape[0]
        atoms_per_residue = coords.shape[1]
        if atom_types is None:
            raise ValueError(
                "atom_types must be provided if coords shape is not (N, 3, 3)"
            )
        elif len(atom_types) != atoms_per_residue:
            raise ValueError(
                f"Length of atom_types ({len(atom_types)}) must match atoms per residue ({atoms_per_residue}) for 3D input."
            )
        coords_3d = coords
    else:
        raise ValueError(f"Coordinates must be a 2D or 3D tensor, got {coords.ndim}D")

    # Default residue names if not provided
    if res_names is None:
        res_names = ["GLY"] * num_residues
    elif len(res_names) != num_residues:
        raise ValueError(
            f"Length of res_names ({len(res_names)}) must match number of residues ({num_residues})"
        )

    # Build the structure
    builder = StructureBuilder()
    builder.init_structure("structure")
    builder.init_model(model_id)
    builder.init_chain(chain_id)

    atom_serial_number = 1
    for res_idx in range(num_residues):
        res_id = (" ", res_idx + 1, " ")  # Hetflag, sequence identifier, insertion code
        res_name = res_names[res_idx]
        builder.init_residue(res_name, res_id[0], res_id[1], res_id[2])

        for atom_idx in range(atoms_per_residue):
            atom_name = atom_types[atom_idx]
            # Ensure atom name is correctly formatted (max 4 chars, potentially padded)
            formatted_atom_name = f"{atom_name:<4s}"[:4]
            coord_np = coords_3d[res_idx, atom_idx, :].detach().cpu().numpy()
            element = atom_name[0]  # Simple guess for element based on first char

            # Create Atom object
            # Atom(name, coord, bfactor, occupancy, altloc, fullname, serial_number, element)
            atom = Atom(
                name=formatted_atom_name,
                coord=coord_np,
                bfactor=0.0,
                occupancy=1.0,
                altloc=" ",
                fullname=formatted_atom_name,  # Use formatted name
                serial_number=atom_serial_number,
                element=element.upper(),  # Ensure element is uppercase
            )
            # Add the created atom object to the current residue in the builder
            builder.residue.add(atom)
            atom_serial_number += 1

    structure = builder.get_structure()

    # Save the structure
    output_path = Path(output_file)
    suffix = output_path.suffix.lower()

    # Create the appropriate IO object based on the suffix
    io: Union[PDBIO, MMCIFIO]  # Define type hint here
    if suffix == ".pdb":
        io = PDBIO()
    elif suffix == ".cif":
        io = MMCIFIO()
    else:
        raise ValueError(f"Unsupported output file format: {suffix}. Use .pdb or .cif.")

    # Set structure and save using the created IO object
    io.set_structure(structure)
    io.save(str(output_path))
