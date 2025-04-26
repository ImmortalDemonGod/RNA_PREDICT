"""
run_full_pipeline.py

This module re-exports the run_full_pipeline function from rna_predict.runners.full_pipeline
for backward compatibility with existing tests and scripts.
"""

from rna_predict.runners.full_pipeline import run_full_pipeline

__all__ = ["run_full_pipeline"]
