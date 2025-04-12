"""
Residue-to-atom bridging utilities for Stage D diffusion.

This package provides functions for bridging between residue-level and atom-level
representations in the Stage D diffusion process.
"""

from .residue_atom_bridge import BridgingInput, bridge_residue_to_atom

__all__ = ["bridge_residue_to_atom", "BridgingInput"]
