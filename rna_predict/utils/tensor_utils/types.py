"""
Type definitions and constants for tensor utilities.

This module provides type aliases and constants used throughout the tensor utilities
package, including standard RNA atom definitions and residue-atom mapping types.
"""

import logging
from typing import List, TypeAlias

# Type alias for residue-to-atom mapping
ResidueAtomMap: TypeAlias = List[List[int]]

# Define standard RNA atom names for each residue type
# These are the heavy atoms typically found in RNA nucleotides
STANDARD_RNA_ATOMS = {
    'A': ['P', 'OP1', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
          'N9', 'C8', 'N7', 'C5', 'C6', 'N6', 'N1', 'C2', 'N3', 'C4'],  # 22 atoms
    'U': ['P', 'OP1', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
          'N1', 'C2', 'O2', 'N3', 'C4', 'O4', 'C5', 'C6'],  # 20 atoms
    'G': ['P', 'OP1', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
          'N9', 'C8', 'N7', 'C5', 'C6', 'O6', 'N1', 'C2', 'N2', 'N3', 'C4'],  # 23 atoms
    'C': ['P', 'OP1', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
          'N1', 'C2', 'O2', 'N3', 'C4', 'N4', 'C5', 'C6'],  # 20 atoms
}

# Logger setup
logger = logging.getLogger(__name__)
