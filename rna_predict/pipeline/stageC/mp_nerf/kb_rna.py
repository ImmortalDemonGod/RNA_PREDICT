"""
kb_rna.py

Holds standard RNA geometry references (backbone + base).
Optionally references final_kb_rna.py or a bridging file if numeric data is large.
"""

from typing import Dict, Any

# If you want to import from final_kb_rna or do bridging, you can do so here.
# from .final_kb_rna import ...

# A minimal dictionary demonstrating how to store geometry data for each base type.
# You can expand or replace these placeholders with real numeric values from final_kb_rna.py.

RNA_BUILD_INFO: Dict[str, Any] = {
    "A": {
        "backbone_atoms": ["P","O5'","C5'","C4'","O4'","C3'","O3'"],
        "bond_lengths": {
            ("P","O5'"): 1.59,
            ("O5'","C5'"): 1.44,
            ("C5'","C4'"): 1.51,
            ("C4'","C3'"): 1.52,
            ("C3'","O3'"): 1.42
        },
        "bond_angles": {
            ("P","O5'","C5'"): 105.0,
            ("O5'","C5'","C4'"): 115.0,
            ("C5'","C4'","C3'"): 102.0,
            ("C4'","C3'","O3'"): 118.0,
        },
        # Torsions might map alpha..zeta + chi
        "torsions": ["alpha","beta","gamma","delta","epsilon","zeta","chi"],
        # Optional defaults (in degrees) if Stage B doesn't provide them
        "default_torsion_degs": {
            "alpha": 300.0,
            "beta": 180.0,
            "gamma": 50.0,
            "delta": 85.0,
            "epsilon": 180.0,
            "zeta": 290.0,
            "chi": 210.0
        },
        # Base atom names for sidechain-like logic
        "base_atoms": ["N9","C8","N7","C5","C6","N6","C4","N3","C2"]
    },
    "U": {
        "backbone_atoms": ["P","O5'","C5'","C4'","O4'","C3'","O3'"],
        "bond_lengths": {
            ("P","O5'"): 1.59,
            ("O5'","C5'"): 1.44,
            ("C5'","C4'"): 1.51,
            ("C4'","C3'"): 1.52,
            ("C3'","O3'"): 1.42
        },
        "bond_angles": {
            ("P","O5'","C5'"): 105.0,
            ("O5'","C5'","C4'"): 115.0,
            ("C5'","C4'","C3'"): 102.0,
            ("C4'","C3'","O3'"): 118.0,
        },
        "torsions": ["alpha","beta","gamma","delta","epsilon","zeta","chi"],
        "default_torsion_degs": {
            "alpha": 300.0,
            "beta": 180.0,
            "gamma": 50.0,
            "delta": 85.0,
            "epsilon": 180.0,
            "zeta": 290.0,
            "chi": 210.0
        },
        "base_atoms": ["N1","C2","N3","C4","C5","C6","O2","O4"]
    },
    "G": {
        "backbone_atoms": ["P","O5'","C5'","C4'","O4'","C3'","O3'"],
        "bond_lengths": {
            ("P","O5'"): 1.59,
            ("O5'","C5'"): 1.44,
            ("C5'","C4'"): 1.51,
            ("C4'","C3'"): 1.52,
            ("C3'","O3'"): 1.42
        },
        "bond_angles": {
            ("P","O5'","C5'"): 105.0,
            ("O5'","C5'","C4'"): 115.0,
            ("C5'","C4'","C3'"): 102.0,
            ("C4'","C3'","O3'"): 118.0,
        },
        "torsions": ["alpha","beta","gamma","delta","epsilon","zeta","chi"],
        "default_torsion_degs": {
            "alpha": 300.0,
            "beta": 180.0,
            "gamma": 50.0,
            "delta": 85.0,
            "epsilon": 180.0,
            "zeta": 290.0,
            "chi": 210.0
        },
        "base_atoms": ["N9","C8","N7","C5","C6","O6","N1","C2","N2","N3","C4"]
    },
    "C": {
        "backbone_atoms": ["P","O5'","C5'","C4'","O4'","C3'","O3'"],
        "bond_lengths": {
            ("P","O5'"): 1.59,
            ("O5'","C5'"): 1.44,
            ("C5'","C4'"): 1.51,
            ("C4'","C3'"): 1.52,
            ("C3'","O3'"): 1.42
        },
        "bond_angles": {
            ("P","O5'","C5'"): 105.0,
            ("O5'","C5'","C4'"): 115.0,
            ("C5'","C4'","C3'"): 102.0,
            ("C4'","C3'","O3'"): 118.0,
        },
        "torsions": ["alpha","beta","gamma","delta","epsilon","zeta","chi"],
        "default_torsion_degs": {
            "alpha": 300.0,
            "beta": 180.0,
            "gamma": 50.0,
            "delta": 85.0,
            "epsilon": 180.0,
            "zeta": 290.0,
            "chi": 210.0
        },
        "base_atoms": ["N1","C2","O2","N3","C4","N4","C5","C6"]
    }
}