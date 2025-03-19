Directory Structure:

└── ./
    ├── src
    │   ├── enums
    │   │   ├── __init__.py
    │   │   └── atoms.py
    │   ├── helper
    │   │   ├── __init__.py
    │   │   ├── computation_helper.py
    │   │   ├── extractor_helper.py
    │   │   └── rna_torsionBERT_helper.py
    │   ├── metrics
    │   │   ├── __init__.py
    │   │   └── mcq.py
    │   ├── __init__.py
    │   ├── rna_torsionBERT_cli.py
    │   ├── tb_mcq_cli.py
    │   └── utils.py
    └── README.md



---
File: /src/enums/__init__.py
---




---
File: /src/enums/atoms.py
---

import numpy as np
import os

ALL_ATOMS = [
    "P",
    "OP1",
    "OP2",
    "O5'",
    "C5'",
    "C4'",
    "O4'",
    "C3'",
    "O3'",
    "C1'",
    "C2'",
    "N1",
    "C2",
    "N9",
    "C4",
]

ANGLES = {
    "alpha": {"atoms": ["O3'", "P", "O5'", "C5'"], "index": [-1, 0, 0, 0]},
    "beta": {"atoms": ["P", "O5'", "C5'", "C4'"], "index": [0, 0, 0, 0]},
    "gamma": {"atoms": ["O5'", "C5'", "C4'", "C3'"], "index": [0, 0, 0, 0]},
    "delta": {"atoms": ["C5'", "C4'", "C3'", "O3'"], "index": [0, 0, 0, 0]},
    "epsilon": {"atoms": ["C4'", "C3'", "O3'", "P"], "index": [0, 0, 0, 1]},
    "zeta": {"atoms": ["C3'", "O3'", "P", "O5'"], "index": [0, 0, 1, 1]},
    "chi": {"atoms": ["O4'", "C1'", "N1", "C2"], "index": [0, 0, 0, 0]},
    "eta": {"atoms": ["C4'", "P", "C4'", "P"], "index": [-1, 0, 0, 1]},
    "theta": {"atoms": ["P", "C4'", "P", "C4'"], "index": [0, 0, 1, 1]},
    "eta'": {"atoms": ["C1'", "P", "C1'", "P"], "index": [-1, 0, 0, 1]},
    "theta'": {"atoms": ["P", "C1'", "P", "C1'"], "index": [0, 0, 1, 1]},
    "v0": {"atoms": ["C4'", "O4'", "C1'", "C2'"], "index": [0, 0, 0, 0]},
    "v1": {"atoms": ["O4'", "C1'", "C2'", "C3'"], "index": [0, 0, 0, 0]},
    "v2": {"atoms": ["C1'", "C2'", "C3'", "C4'"], "index": [0, 0, 0, 0]},
    "v3": {"atoms": ["C2'", "C3'", "C4'", "O4'"], "index": [0, 0, 0, 0]},
    "v4": {"atoms": ["C3'", "C4'", "O4'", "C1'"], "index": [0, 0, 0, 0]},
}



---
File: /src/helper/__init__.py
---




---
File: /src/helper/computation_helper.py
---

import numpy as np
from typing import List

from src.enums.atoms import ANGLES, ALL_ATOMS
from src.utils import compute_torsion_angle


class ComputationHelper:
    def __init__(self, matrix: np.ndarray, sequence: str):
        self.matrix = matrix
        self.sequence = sequence

    def compute_angles(self, angle_name: str) -> List:
        """
        Compute all the angles for the given structure.
        :param angle_name: the angle to compute values from
        :return: a list with the angle values
        """
        c_angle_dict = ANGLES.get(angle_name, {})
        atoms = c_angle_dict.get("atoms", [])
        atoms_position = [ALL_ATOMS.index(atom) for atom in atoms]
        indexes = c_angle_dict.get("index", [])
        angles_out = []
        for i, c_atoms in enumerate(self.matrix):
            if angle_name == "chi" and self.sequence[i] in ["A", "G"]:
                atoms_position = [
                    ALL_ATOMS.index(atom) for atom in ["O4'", "C1'", "N9", "C4"]
                ]
            if angle_name == "chi" and self.sequence[i] in ["C", "U"]:
                atoms_position = [
                    ALL_ATOMS.index(atom) for atom in ["O4'", "C1'", "N1", "C2"]
                ]
            specific_atoms = [
                self.matrix[i + offset, atom_pos]
                for offset, atom_pos in zip(indexes, atoms_position)
                if i + offset < len(self.matrix) and i + offset >= 0
            ]
            angle = (
                compute_torsion_angle(*specific_atoms)
                if len(specific_atoms) == 4
                else np.nan
            )
            angles_out.append(angle)
        return angles_out



---
File: /src/helper/extractor_helper.py
---

from typing import Dict, List, Optional
import numpy as np

import pandas as pd

from src.enums.atoms import ALL_ATOMS, ANGLES
from src.helper.computation_helper import ComputationHelper
from src.utils import read_all_atoms, get_sequence


class ExtractorHelper:
    def __init__(self, all_atoms: List = ALL_ATOMS):
        self.all_atoms = all_atoms

    def extract_all(
        self, in_pdb: str, save_to_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Extract all the torsional angles and bond angles from the pdb file
        :param in_pdb: path to a .pdb file
        :param save_to_path: path where to save the output
        :return: a .csv file with the torsional and bond angles.
        """
        all_atoms = read_all_atoms(in_pdb)
        matrix = self.convert_atoms_to_matrix(all_atoms)
        sequence = [element for element in get_sequence(in_pdb)]
        computation_helper = ComputationHelper(matrix, sequence)
        torsion_angles = {
            angle: computation_helper.compute_angles(angle) for angle in ANGLES
        }
        sequence = [element for element in get_sequence(in_pdb)]
        df = pd.DataFrame(
            {**{"sequence": sequence}, **torsion_angles},
            index=range(1, len(sequence) + 1),
        )
        if save_to_path:
            df.to_csv(save_to_path)
        return df

    def convert_atoms_to_matrix(self, all_atoms: Dict) -> np.ndarray:
        """
        Convert the different atoms into a matrix of size (L, N, 3) where:
            L: the number of nucleotides
            N: the number of atoms per nucleotide
            3: the x,y,z coordinates
        :param all_atoms: list of atoms with their coordinates
        :return: a np.array matrix
        """
        output = np.nan * np.ones((len(all_atoms), len(self.all_atoms), 3))
        for index, atoms in enumerate(all_atoms):
            for atom in atoms:
                if atom in self.all_atoms:
                    output[index, self.all_atoms.index(atom)] = np.array(atoms[atom])
        return output



---
File: /src/helper/rna_torsionBERT_helper.py
---

import transformers
from transformers import AutoModel, AutoTokenizer
import numpy as np
import pandas as pd
from typing import Optional, Dict
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

transformers.logging.set_verbosity_error()


BACKBONE = [
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "chi",
    "eta",
    "theta",
    "eta'",
    "theta'",
    "v0",
    "v1",
    "v2",
    "v3",
    "v4",
]


class RNATorsionBERTHelper:
    def __init__(self):
        self.model_name = "sayby/rna_torsionbert"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.params_tokenizer = {
            "return_tensors": "pt",
            "padding": "max_length",
            "max_length": 512,
            "truncation": True,
        }
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)

    def predict(self, sequence: str):
        sequence_tok = self.convert_raw_sequence_to_k_mers(sequence)
        inputs = self.tokenizer(sequence_tok, **self.params_tokenizer)
        outputs = self.model(inputs)["logits"]
        outputs = self.convert_sin_cos_to_angles(
            outputs.cpu().detach().numpy(), inputs["input_ids"]
        )
        output_angles = self.convert_logits_to_dict(
            outputs[0, :], inputs["input_ids"][0, :].cpu().detach().numpy()
        )
        output_angles.index = list(sequence)[:-2]  # Because of the 3-mer representation
        return output_angles

    def convert_raw_sequence_to_k_mers(self, sequence: str, k_mers: int = 3):
        """
        Convert a raw RNA sequence into sequence readable for the tokenizer.
        It converts the sequence into k-mers, and replace U by T
        :return: input readable by the tokenizer
        """
        sequence = sequence.upper().replace("U", "T")
        k_mers_sequence = [
            sequence[i : i + k_mers]
            for i in range(len(sequence))
            if len(sequence[i : i + k_mers]) == k_mers
        ]
        return " ".join(k_mers_sequence)

    def convert_sin_cos_to_angles(
        self, output: np.ndarray, input_ids: Optional[np.ndarray] = None
    ):
        """
        Convert the raw predictions of the RNA-TorsionBERT into angles.
        It converts the cos and sinus into angles using:
            alpha = arctan(sin(alpha)/cos(alpha))
        :param output: Dictionary with the predictions of the RNA-TorsionBERT per angle
        :param input_ids: the input_ids of the RNA-TorsionBERT. It allows to only select the of the sequence,
            and not the special tokens.
        :return: a np.ndarray with the angles for the sequence
        """
        if input_ids is not None:
            output[
                (input_ids == 0)
                | (input_ids == 2)
                | (input_ids == 3)
                | (input_ids == 4)
            ] = np.nan
        pair_indexes, impair_indexes = np.arange(0, output.shape[-1], 2), np.arange(
            1, output.shape[-1], 2
        )
        sin, cos = output[:, :, impair_indexes], output[:, :, pair_indexes]
        tan = np.arctan2(sin, cos)
        angles = np.degrees(tan)
        return angles

    def convert_logits_to_dict(self, output: np.ndarray, input_ids: np.ndarray) -> Dict:
        """
        Convert the raw predictions into dictionary format.
        It removes the special tokens and only keeps the predictions for the sequence.
        :param output: predictions from the models in angles
        :param input_ids: input ids from the tokenizer
        :return: a dictionary with the predictions for each angle
        """
        index_start, index_end = (
            np.where(input_ids == 2)[0][0],
            np.where(input_ids == 3)[0][0],
        )
        output_non_pad = output[index_start + 1 : index_end, :]
        output_angles = {
            angle: output_non_pad[:, angle_index]
            for angle_index, angle in enumerate(BACKBONE)
        }
        out = pd.DataFrame(output_angles)
        return out



---
File: /src/metrics/__init__.py
---




---
File: /src/metrics/mcq.py
---

from typing import Union, List
import numpy as np
import pandas as pd

import warnings

warnings.filterwarnings("ignore")

TORSION_TO_ANGLES = {
    "BACKBONE": ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi", "phase"],
    "PSEUDO": ["eta", "theta"],
}


class MCQ:
    """
    MCQ Helper. Reproduce the MCQ computation from https://github.com/tzok/mcq4structures.
    """

    def mod(self, values: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """
        Compute the mod(t) = t + 2pi % 2pi
        Computation is done in degree
        :param values:
        :return:
        """
        return (values + 360) % 360

    def difference(self, x: float, y: float):
        """
        Compute the distance between two angles
        :param x: the true angle
        :param x: the predicted atom
        :return: the distance based on MCQ computation
        """
        if np.isnan(x) and np.isnan(y):
            return 0
        elif np.isnan(x) or np.isnan(y):
            return 180
        else:
            return min(
                abs(self.mod(x) - self.mod(y)), 360 - abs(self.mod(x) - self.mod(y))
            )

    def get_phase(self, values: np.ndarray):
        """
        Compute the phase P = arctan(v1 + v4 - v0 - v3, 2v2(sin 36 + sin72))
        :param values:
        :return:
        """
        try:
            riboses = np.radians(values[["v0", "v1", "v2", "v3", "v4"]])
            P = np.arctan(
                riboses["v1"] + riboses["v4"] - riboses["v0"] - riboses["v3"],
                2 * riboses["v2"] * (np.sin(np.radians(36)) + np.sin(np.radians(72))),
            )
        except KeyError:
            P = np.nan
        return P

    def compute_mcq(
        self,
        true_values: np.ndarray,
        pred_values: np.ndarray,
        torsion: str = "BACKBONE",
    ):
        """
        Compute the MCQ between two sets of angles.
        :param true_values: experimental inferred angles from the native structure
        :param pred_values: predicted angles
        :param torsion: the type of angles to use. Default to BACKBONE.
        :return:
        """
        diff, angles = self._get_diff_angles(true_values, pred_values, torsion)
        sin_mod, cos_mod = (
            np.sin(np.radians(diff)).sum(),
            np.cos(np.radians(diff)).sum(),
        )
        mcq = np.arctan2(sin_mod, cos_mod)
        mcq = np.degrees(mcq)
        return mcq

    def get_current_angles(self, torsion: str, pred_cols: List):
        """
        Get the current angles values from all the angles available.
        """
        angles = []
        for angle in TORSION_TO_ANGLES[torsion]:
            if angle == "phase" and "v0" in pred_cols:
                angles.append(angle)
            elif angle in pred_cols:
                angles.append(angle)
        return angles

    def _get_diff_angles(
        self, true_values: pd.DataFrame, pred_values: pd.DataFrame, torsion: str
    ):
        """
        Compute the differences to be used for the MCQ computation
        :return:
        """
        if len(pred_values) < len(true_values):
            true_values = true_values[: len(pred_values)]
        else:
            pred_values = pred_values[: len(true_values)]
        angles = self.get_current_angles(torsion, pred_values.columns)
        if torsion == "BACKBONE":
            phase_true = np.degrees(self.get_phase(true_values))
            phase_pred = np.degrees(self.get_phase(pred_values))
            true_values["phase"] = phase_true
            pred_values["phase"] = phase_pred
        true_angles = true_values[angles].values
        pred_angles = pred_values[angles].values
        diff_fn = np.vectorize(self.difference)
        diff = diff_fn(true_angles, pred_angles)
        return diff, angles

    def compute_mcq_per_angle(self, true_values, pred_values, torsion: str):
        """
        Compute the MCQ for a given angle.
        :param true_values: experimental inferred angles from the native structure
        :param pred_values: predicted angles
        :param torsion: the type of angles to use. Default to BACKBONE.
        :return: MCQ per angle
        """
        diff, angles = self._get_diff_angles(true_values, pred_values, torsion)
        sin_mod, cos_mod = np.sin(np.radians(diff)).sum(axis=0), np.cos(
            np.radians(diff)
        ).sum(axis=0)
        mcq = np.arctan2(sin_mod, cos_mod)
        mcq = np.degrees(mcq)
        output = {angle: mcq[i] for i, angle in enumerate(angles)}
        return output

    def compute_mcq_per_sequence(self, true_values, pred_values, torsion: str):
        """
        Compute the MCQ for a given position.
        :return:
        """
        diff, angles = self._get_diff_angles(true_values, pred_values, torsion)
        sin_mod, cos_mod = np.sin(np.radians(diff)).sum(axis=1), np.cos(
            np.radians(diff)
        ).sum(axis=1)
        mcq = np.arctan2(sin_mod, cos_mod)
        mcq = np.degrees(mcq)
        return mcq.tolist()



---
File: /src/__init__.py
---




---
File: /src/rna_torsionBERT_cli.py
---

import argparse
from typing import Optional

from src.helper.rna_torsionBERT_helper import RNATorsionBERTHelper
from src.utils import read_fasta
from loguru import logger


class RNATorsionBERTCLI:
    def __init__(
        self,
        in_seq: Optional[str],
        in_fasta: Optional[str],
        out_path: Optional[str],
        *args,
        **kwargs,
    ):
        self.sequence = self._init_inputs(in_seq, in_fasta)
        self.out_path = out_path

    def _init_inputs(self, in_seq: Optional[str], in_fasta: Optional[str]) -> str:
        """
        Initialise the inputs given the arguments
        :return: the sequence
        """
        if in_seq is None and in_fasta is None:
            raise ValueError("You must provide either a sequence or a fasta file.")
        if in_seq is not None and in_fasta is not None:
            raise ValueError(
                "Please provide only the sequence or the fasta file, not both."
            )
        if in_seq is not None:
            sequence = in_seq
        elif in_fasta is not None:
            sequence = read_fasta(in_fasta)
        return sequence

    def run(self):
        output = RNATorsionBERTHelper().predict(self.sequence)
        if self.out_path is not None:
            output.to_csv(self.out_path)
            logger.info(f"Saved the output to {self.out_path}")
        return output

    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser(
            description="Prediction of Torsional angles for RNA structures"
        )
        # Add command line arguments
        parser.add_argument(
            "--in_seq",
            dest="in_seq",
            type=str,
            help="RNA Input sequence.",
            default=None,
        )
        parser.add_argument(
            "--in_fasta",
            dest="in_fasta",
            type=str,
            help="Path to a fasta file.",
            default=None,
        )
        parser.add_argument(
            "--out_path",
            dest="out_path",
            type=str,
            help="Path to a .csv file to save the prediction",
            default=None,
        )
        # Parse the command line arguments
        args = parser.parse_args()
        return args


if __name__ == "__main__":
    args = RNATorsionBERTCLI.get_args()
    rna_torsionBERT_cli = RNATorsionBERTCLI(**vars(args))
    rna_torsionBERT_cli.run()



---
File: /src/tb_mcq_cli.py
---

import argparse
import os

from loguru import logger
import tqdm
import pandas as pd
from typing import Optional, List

from src.helper.extractor_helper import ExtractorHelper
from src.helper.rna_torsionBERT_helper import RNATorsionBERTHelper
from src.metrics.mcq import MCQ


class TBMCQCLI:
    def __init__(self, in_pdb: str, out_path: Optional[str], *args, **kwargs):
        self.list_files = self._init_pdb(in_pdb)
        self.out_path = out_path

    def _init_pdb(self, in_pdb: Optional[str]) -> List:
        """
        Initialise the inputs structures.
        :param in_pdb: a path to either a .pdb file or a directory of .pdb files
        :return: a list of path to .pdb files
        """
        if os.path.isdir(in_pdb):
            list_files = os.listdir(in_pdb)
            list_files = [os.path.join(in_pdb, file_) for file_ in list_files]
        elif os.path.isfile(in_pdb):
            list_files = [in_pdb]
        else:
            logger.info(f"NO INPUTS FOUND FOR INPUT .PDB: {in_pdb}")
        return list_files

    def run(self):
        all_scores = {"RNA": [], "TB-MCQ": []}
        for in_path in tqdm.tqdm(self.list_files):
            score = self.compute_tb_mcq(in_path)
            all_scores["RNA"].append(os.path.basename(in_path))
            all_scores["TB-MCQ"].append(score)
        all_scores = pd.DataFrame(all_scores, columns=["TB-MCQ", "RNA"]).set_index(
            "RNA"
        )
        if self.out_path is not None:
            logger.info(f"Saved the output to {self.out_path}")
            all_scores.to_csv(self.out_path, index=True)
        return all_scores

    def compute_tb_mcq(self, pred_path: str) -> float:
        """
        Compute the TB-MCQ with RNA-TorsionBERT model
        :param pred_path: the path to the .pdb file of a prediction.
                It could be a native or a predicted structure.
        """
        experimental_angles = ExtractorHelper().extract_all(pred_path)
        sequence = "".join(experimental_angles["sequence"].values)
        torsionBERT_helper = RNATorsionBERTHelper()
        torsionBERT_output = torsionBERT_helper.predict(sequence)
        mcq = MCQ().compute_mcq(experimental_angles, torsionBERT_output)
        return mcq

    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser(
            description="Prediction of Torsional angles for RNA structures"
        )
        # Add command line arguments
        parser.add_argument(
            "--in_pdb",
            dest="in_pdb",
            type=str,
            help="Path a .pdb file or a directory of .pdb files.",
            default=None,
        )
        parser.add_argument(
            "--out_path",
            dest="out_path",
            type=str,
            help="Path to a .csv file to save the predictions.",
            default=None,
        )
        # Parse the command line arguments
        args = parser.parse_args()
        return args


if __name__ == "__main__":
    args = TBMCQCLI.get_args()
    tb_mcq_cli = TBMCQCLI(**vars(args))
    tb_mcq_cli.run()



---
File: /src/utils.py
---

from typing import Any, List, Dict
from Bio.PDB import Atom, Model, Chain, Residue, Structure, PDBIO
import Bio
import numpy as np
from Bio.PDB import Atom, Residue, PDBParser
import warnings

warnings.filterwarnings("ignore")


def read_fasta(in_path: str) -> str:
    """
    Read a fasta file to get the sequence
    :param in_path: path to a .fasta file
    :return: the RNA sequence
    """
    with open(in_path, "r") as f:
        lines = f.readlines()
    sequence = "".join([line.strip() for line in lines[1:]])
    return sequence


def read_all_atoms(in_pdb: str) -> Any:
    """
    Read and return the coordinates of all the  atoms from a pdb file
    :param in_pdb: path to a .pdb file
    """
    parser = PDBParser()
    all_atoms = []
    structure = parser.get_structure("", in_pdb)
    for model in structure:
        for chain in model:
            for residue in chain:
                res = residue.get_resname().replace(" ", "")
                if res in ["A", "C", "G", "U"]:
                    atoms = get_atoms_torsion(residue)
                    c_atom = {
                        atom.get_name(): atom.get_coord().tolist() for atom in atoms
                    }
                    all_atoms.append(c_atom)
    return all_atoms


def get_atoms_torsion(residue: Bio.PDB.Residue.Residue):
    """
    Return the atoms coordinates for a given residue.
    :param residue: the residue to get the atoms from
    """
    atoms = []
    for atom in residue:
        atoms.append(atom)
    return atoms


def compute_torsion_angle(
    atom1: np.ndarray, atom2: np.ndarray, atom3: np.ndarray, atom4: np.ndarray
) -> float:
    """
    Compute torsional angles between 4 atoms
    :return: the torsional angles between the atoms
    """
    v12 = atom1 - atom2
    v23 = atom2 - atom3
    v34 = atom3 - atom4
    e1 = np.cross(v12, v23)
    e2 = np.cross(v23, v34)
    sign = +1 if np.dot(v23, np.cross(e1, e2)) < 0 else -1
    angle_in_radians = np.arccos(
        np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
    )
    angle_in_degrees = sign * np.degrees(angle_in_radians)
    return angle_in_degrees


def get_sequence(in_pdb: str) -> str:
    """
    Return the RNA sequence from a .pdb file
    :param in_pdb: path to a pdb file
    :return: RNA sequence of nucleotides
    """
    parser = PDBParser()
    structure = parser.get_structure("structure", in_pdb)
    sequence = ""
    for model in structure:
        for chain in model:
            for residue in chain:
                res = residue.get_resname().replace(" ", "")
                if res in ["A", "C", "G", "U"]:
                    sequence += res
    return sequence



---
File: /README.md
---

# RNA-TorsionBERT

`RNA-TorsionBERT` is a 86.9 MB parameter BERT-based language model that predicts RNA torsional and pseudo-torsional angles from the sequence.

![](./img/dnabert_architecture_final.png)


`RNA-TorsionBERT` is a DNABERT model that was pre-trained on ~4200 RNA structures.

It provides improvement of [MCQ](https://github.com/tzok/mcq4structures) over the previous state-of-the-art models like 
[SPOT-RNA-1D](https://github.com/jaswindersingh2/SPOT-RNA-1D) or inferred angles from existing methods, on the Test Set (composed of RNA-Puzzles and CASP-RNA).

## Installation

To install RNA-TorsionBERT and it's dependencies following commands can be used in terminal:

```bash
pip install -r requirements.txt 
```


## RNA-TorsionBERT usage

To run the RNA-TorsionBERT, you can use the following command line:
```bash
python -m src.rna_torsionBERT_cli [--seq_file] [--in_fasta] [--out_path]
```

The arguments are the following:
- `--seq_file`: RNA Sequence. 
- `--in_fasta`: Path to the input sequence fasta file. 
- `--out_path`: Path to a `.csv` file where the output will be saved. 

You can also import in your python code the class `RNATorsionBERTCLI` from `src.rna_torsionBERT_cli`. 


## TB-MCQ

TB-MCQ stands for TorsionBERT-MCQ, which is a scoring function to assess the quality of a predicted structure in torsional angle space.
Given the inferred angles from the structures and the predicted angles from the model, TB-MCQ computes the quality of the predicted angles using 
the [MCQ](https://github.com/tzok/mcq4structures) (mean of circular quantities) metric.

![](./img/torsion_bert_mcq_T.png)

To run the TB-MCQ scoring function, you can use the following command line:
```bash
python -m src.rna_torsion_cli [--in_pdb] [--out_path]
```
with:

- `--in_pdb`: Path to the input PDB file.
- `--out_path`: Path to a .csv file where the output will be saved.


## Docker 
To run the code using `Docker`, you can use the following command line:
```bash
docker build -t rna_torsionbert .
docker run -it rna_torsionbert 
```

It will enter into a bash console where you could execute the previous commands with all the installations done. 

To have example of commands, you can look at the `Makefile`.


## Citation

```bibtex
@article{rna_torsion_bert,
    author = {Bernard, Clément and Postic, Guillaume and Ghannay, Sahar and Tahi, Fariza},
    title = {RNA-TorsionBERT: leveraging language models for RNA 3D torsion angles prediction},
    journal = {Bioinformatics},
    volume = {41},
    number = {1},
    pages = {btaf004},
    year = {2025},
    month = {01},
    issn = {1367-4811},
    doi = {10.1093/bioinformatics/btaf004},
    url = {https://doi.org/10.1093/bioinformatics/btaf004},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/41/1/btaf004/61381586/btaf004.pdf},
}
```

====

Hugging Face's logo Hugging Face

Models
Datasets
Spaces
Docs
Enterprise
Pricing

sayby
/
rna_torsionBERT
Token Classification
Transformers
PyTorch
Safetensors
rna_torsionbert
feature-extraction
biology
RNA
Torsional
Angles
custom_code
Model card
Files
Community

RNA-TorsionBERT
Model Description

RNA-TorsionBERT is a 86.9 MB parameter BERT-based language model that predicts RNA torsional and pseudo-torsional angles from the sequence.

RNA-TorsionBERT is a DNABERT model that was pre-trained on ~4200 RNA structures.

It provides improvement of MCQ over the previous state-of-the-art models like SPOT-RNA-1D or inferred angles from existing methods, on the Test Set (composed of RNA-Puzzles and CASP-RNA).

Key Features

    Torsional and Pseudo-torsional angles prediction
    Predict sequences up to 512 nucleotides

Usage

Get started generating text with RNA-TorsionBERT by using the following code snippet:

from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("sayby/rna_torsionbert", trust_remote_code=True)
model = AutoModel.from_pretrained("sayby/rna_torsionbert", trust_remote_code=True)

sequence = "ACG CGG GGT GTT"
params_tokenizer = {
    "return_tensors": "pt",
    "padding": "max_length",
    "max_length": 512,
    "truncation": True,
}
inputs = tokenizer(sequence, **params_tokenizer)
output = model(inputs)["logits"]

    Please note that it was fine-tuned from a DNABERT-3 model and therefore the tokenizer is the same as the one used for DNABERT. Nucleotide U should therefore be replaced by T in the input sequence.
    The output is the sinus and the cosine for each angle. The angles are in the following order: alpha, beta,gamma,delta,epsilon,zeta,chi,eta,theta,eta',theta',v0,v1,v2,v3,v4.

To convert the predictions into angles, you can use the following code snippet:

import transformers
from transformers import AutoModel, AutoTokenizer
import numpy as np
import pandas as pd
from typing import Optional, Dict
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

transformers.logging.set_verbosity_error()


BACKBONE = [
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "chi",
    "eta",
    "theta",
    "eta'",
    "theta'",
    "v0",
    "v1",
    "v2",
    "v3",
    "v4",
]


class RNATorsionBERTHelper:
    def __init__(self):
        self.model_name = "sayby/rna_torsionbert"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.params_tokenizer = {
            "return_tensors": "pt",
            "padding": "max_length",
            "max_length": 512,
            "truncation": True,
        }
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)

    def predict(self, sequence: str):
        sequence_tok = self.convert_raw_sequence_to_k_mers(sequence)
        inputs = self.tokenizer(sequence_tok, **self.params_tokenizer)
        outputs = self.model(inputs)["logits"]
        outputs = self.convert_sin_cos_to_angles(
            outputs.cpu().detach().numpy(), inputs["input_ids"]
        )
        output_angles = self.convert_logits_to_dict(
            outputs[0, :], inputs["input_ids"][0, :].cpu().detach().numpy()
        )
        output_angles.index = list(sequence)[:-2]  # Because of the 3-mer representation
        return output_angles

    def convert_raw_sequence_to_k_mers(self, sequence: str, k_mers: int = 3):
        """
        Convert a raw RNA sequence into sequence readable for the tokenizer.
        It converts the sequence into k-mers, and replace U by T
        :return: input readable by the tokenizer
        """
        sequence = sequence.upper().replace("U", "T")
        k_mers_sequence = [
            sequence[i : i + k_mers]
            for i in range(len(sequence))
            if len(sequence[i : i + k_mers]) == k_mers
        ]
        return " ".join(k_mers_sequence)

    def convert_sin_cos_to_angles(
        self, output: np.ndarray, input_ids: Optional[np.ndarray] = None
    ):
        """
        Convert the raw predictions of the RNA-TorsionBERT into angles.
        It converts the cos and sinus into angles using:
            alpha = arctan(sin(alpha)/cos(alpha))
        :param output: Dictionary with the predictions of the RNA-TorsionBERT per angle
        :param input_ids: the input_ids of the RNA-TorsionBERT. It allows to only select the of the sequence,
            and not the special tokens.
        :return: a np.ndarray with the angles for the sequence
        """
        if input_ids is not None:
            output[
                (input_ids == 0)
                | (input_ids == 2)
                | (input_ids == 3)
                | (input_ids == 4)
            ] = np.nan
        pair_indexes, impair_indexes = np.arange(0, output.shape[-1], 2), np.arange(
            1, output.shape[-1], 2
        )
        sin, cos = output[:, :, impair_indexes], output[:, :, pair_indexes]
        tan = np.arctan2(sin, cos)
        angles = np.degrees(tan)
        return angles

    def convert_logits_to_dict(self, output: np.ndarray, input_ids: np.ndarray) -> Dict:
        """
        Convert the raw predictions into dictionary format.
        It removes the special tokens and only keeps the predictions for the sequence.
        :param output: predictions from the models in angles
        :param input_ids: input ids from the tokenizer
        :return: a dictionary with the predictions for each angle
        """
        index_start, index_end = (
            np.where(input_ids == 2)[0][0],
            np.where(input_ids == 3)[0][0],
        )
        output_non_pad = output[index_start + 1 : index_end, :]
        output_angles = {
            angle: output_non_pad[:, angle_index]
            for angle_index, angle in enumerate(BACKBONE)
        }
        out = pd.DataFrame(output_angles)
        return out


if __name__ == "__main__":
    sequence = "AGGGCUUUAGUCUUUGGAG"
    rna_torsionbert_helper = RNATorsionBERTHelper()
    output_angles = rna_torsionbert_helper.predict(sequence)
    print(output_angles)

Downloads last month
    988 

Safetensors
Model size
86.9M params
Tensor type
F32
Inference Providers
NEW
Token Classification
This model is not currently available via any of the supported Inference Providers.
The model cannot be deployed to the HF Inference API: The HF Inference API does not support model that require custom code execution.
Model tree for sayby/rna_torsionBERT

Base model
zhihan1996/DNA_bert_3
Finetuned
(1)
this model
TOS
Privacy
About
Jobs
Models
Datasets
Spaces
Pricing
Docs
