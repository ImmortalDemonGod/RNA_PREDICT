"""
kb_rna.py

Holds standard RNA geometry references (backbone + base).
Optionally references final_kb_rna.py or a bridging file if numeric data is large.
"""

from typing import Dict, Any

# A minimal dictionary demonstrating how to store geometry data for each base type,
# now expanded to include the sugar ring (C2', O2', C1') in the backbone_atoms.

RNA_BUILD_INFO: Dict[str, Any] = {
    "A": {
        # 10 backbone atoms: P, O5', C5', C4', O4', C3', O3', C2', O2', C1'
        "backbone_atoms": ["P","O5'","C5'","C4'","O4'","C3'","O3'","C2'","O2'","C1'"],
        "bond_lengths": {
            ("P","O5'"): 1.59,
            ("O5'","C5'"): 1.44,
            ("C5'","C4'"): 1.51,
            ("C4'","O4'"): 1.45,
            ("C4'","C3'"): 1.52,
            ("C3'","O3'"): 1.42,
            ("C3'","C2'"): 1.52,
            ("C2'","O2'"): 1.41,
            ("C1'","C2'"): 1.53,
            # If needed, you can define additional ring bonds (e.g. O4'-C1') explicitly
        },
        # Some standard angles, mostly placeholders or typical A-form references
        "bond_angles": {
            ("P","O5'","C5'"): 105.0,
            ("O5'","C5'","C4'"): 115.0,
            ("C5'","C4'","O4'"): 106.0,
            ("C5'","C4'","C3'"): 102.0,
            ("C4'","C3'","O3'"): 118.0,

            ("C4'","C3'","C2'"): 110.0,
            ("C3'","C2'","O2'"): 109.0,
            ("C1'","C2'","C3'"): 101.0,
            ("C1'","C2'","O2'"): 110.0,

            # bridging angles for reference:
            ("O3'","P","O5'"): 105.0
        },
        # Torsions might map alpha..zeta + chi, plus ring torsions if flexible
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
        # We can keep base_atoms for future expansions
        "base_atoms": ["N9","C8","N7","C5","C6","N6","C4","N3","C2"]
    },

    "U": {
        "backbone_atoms": ["P","O5'","C5'","C4'","O4'","C3'","O3'","C2'","O2'","C1'"],
        "bond_lengths": {
            ("P","O5'"): 1.59,
            ("O5'","C5'"): 1.44,
            ("C5'","C4'"): 1.51,
            ("C4'","O4'"): 1.45,
            ("C4'","C3'"): 1.52,
            ("C3'","O3'"): 1.42,
            ("C3'","C2'"): 1.52,
            ("C2'","O2'"): 1.41,
            ("C1'","C2'"): 1.53,
        },
        "bond_angles": {
            ("P","O5'","C5'"): 105.0,
            ("O5'","C5'","C4'"): 115.0,
            ("C5'","C4'","O4'"): 106.0,
            ("C5'","C4'","C3'"): 102.0,
            ("C4'","C3'","O3'"): 118.0,

            ("C4'","C3'","C2'"): 110.0,
            ("C3'","C2'","O2'"): 109.0,
            ("C1'","C2'","C3'"): 101.0,
            ("C1'","C2'","O2'"): 110.0,

            ("O3'","P","O5'"): 105.0
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
        "backbone_atoms": ["P","O5'","C5'","C4'","O4'","C3'","O3'","C2'","O2'","C1'"],
        "bond_lengths": {
            ("P","O5'"): 1.59,
            ("O5'","C5'"): 1.44,
            ("C5'","C4'"): 1.51,
            ("C4'","O4'"): 1.45,
            ("C4'","C3'"): 1.52,
            ("C3'","O3'"): 1.42,
            ("C3'","C2'"): 1.52,
            ("C2'","O2'"): 1.41,
            ("C1'","C2'"): 1.53,
        },
        "bond_angles": {
            ("P","O5'","C5'"): 105.0,
            ("O5'","C5'","C4'"): 115.0,
            ("C5'","C4'","O4'"): 106.0,
            ("C5'","C4'","C3'"): 102.0,
            ("C4'","C3'","O3'"): 118.0,

            ("C4'","C3'","C2'"): 110.0,
            ("C3'","C2'","O2'"): 109.0,
            ("C1'","C2'","C3'"): 101.0,
            ("C1'","C2'","O2'"): 110.0,

            ("O3'","P","O5'"): 105.0
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
        "backbone_atoms": ["P","O5'","C5'","C4'","O4'","C3'","O3'","C2'","O2'","C1'"],
        "bond_lengths": {
            ("P","O5'"): 1.59,
            ("O5'","C5'"): 1.44,
            ("C5'","C4'"): 1.51,
            ("C4'","O4'"): 1.45,
            ("C4'","C3'"): 1.52,
            ("C3'","O3'"): 1.42,
            ("C3'","C2'"): 1.52,
            ("C2'","O2'"): 1.41,
            ("C1'","C2'"): 1.53,
        },
        "bond_angles": {
            ("P","O5'","C5'"): 105.0,
            ("O5'","C5'","C4'"): 115.0,
            ("C5'","C4'","O4'"): 106.0,
            ("C5'","C4'","C3'"): 102.0,
            ("C4'","C3'","O3'"): 118.0,

            ("C4'","C3'","C2'"): 110.0,
            ("C3'","C2'","O2'"): 109.0,
            ("C1'","C2'","C3'"): 101.0,
            ("C1'","C2'","O2'"): 110.0,

            ("O3'","P","O5'"): 105.0
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
    },

    # Example extension for a hypothetical modified base "m5C" or pseudouridine "PSU"
    # "m5C": {
    #     "backbone_atoms": [...],
    #     "bond_lengths": {...},
    #     "bond_angles": {...},
    #     ...
    # }

}