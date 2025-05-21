import torch
import logging
from omegaconf import OmegaConf
from rna_predict.interface import RNAPredictor

TEST_SEQS  = "/kaggle/input/stanford-rna-3d-folding/test_sequences.csv"
SAMPLE_SUB = "/kaggle/input/stanford-rna-3d-folding/sample_submission.csv"
OUTPUT_CSV = "submission.csv"

def create_predictor():
    """Instantiate RNAPredictor with local checkpoints & GPU/CPU autodetect, matching Hydra config structure."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}") # Assuming logging is configured
    cfg = OmegaConf.create({
        # Top-level keys consistent with a full Hydra config (e.g., default.yaml)
        "device": device,
        "seed": 42, # Good for reproducibility if used by models
        "atoms_per_residue": 44, # Standard value
        "extraction_backend": "dssr", # Or "mdanalysis" as needed

        "pipeline": { # General pipeline settings
            "verbose": True,
            "save_intermediates": True,
            # output_dir is usually set by Hydra's run directory or overridden
        },

        "prediction": { # Prediction-specific settings
            "repeats": 5,
            "residue_atom_choice": 0,
            "enable_stochastic_inference_for_submission": True, # CRITICAL: Ensures unique predictions
            # "submission_seeds": [42, 101, 2024, 7, 1991],  # Optional: for reproducible stochastic runs
        },

        "model": {
            # ── Stage B: torsion-angle prediction ──────────────────────────
            "stageB": {
                "torsion_bert": {
                    "model_name_or_path": "/kaggle/working/rna_torsionBERT", # Path to local TorsionBERT model
                    "device": device,
                    "angle_mode": "degrees",   # CHANGED: Set to "degrees" for consistency with StageC
                                               # This ensures StageBTorsionBertPredictor outputs angles in degrees.
                    "num_angles": 7,
                    "max_length": 512,
                    "checkpoint_path": None,   # Can be overridden if a specific checkpoint is needed
                    "debug_logging": True,     # Set to False if logs are too verbose
                    "init_from_scratch": False, # Assumes using pretrained TorsionBERT
                    "lora": {                  # LoRA config (currently disabled)
                        "enabled": False,
                        "r": 8,
                        "alpha": 16,
                        "dropout": 0.1,
                        "target_modules": ["query", "value"],
                    },
                }
                # Pairformer config would go here if used: "pairformer": { ... }
            },
            # ── Stage C: 3D reconstruction (MP-NeRF) ──────────────────────
            "stageC": {
                "enabled": True,
                "method": "mp_nerf",
                "do_ring_closure": False,       # Consistent with default.yaml; notebook log showed True, adjust if needed.
                "place_bases": True,
                "sugar_pucker": "C3'-endo",
                "device": device,
                "debug_logging": True,          # Set to False if logs are too verbose
                "angle_representation": "degrees", # StageC expects angles in degrees from StageB
                "use_metadata": False,
                "use_memory_efficient_kernel": False,
                "use_deepspeed_evo_attention": False,
                "use_lma": False,
                "inplace_safe": False,          # Consistent with default.yaml; notebook log showed True.
                "chunk_size": None,
            },
            # ── Stage D: Diffusion refinement (minimal placeholder) ────────
            # Add full StageD config if it's actively used in this notebook
            "stageD": {
                "enabled": False, # Set to True if StageD is part of this specific notebook's pipeline
                "mode": "inference",
                "device": device,
                "debug_logging": True,
                # Placeholder for other essential StageD keys if enabled:
                # "ref_element_size": 128,
                # "ref_atom_name_chars_size": 256,
                # "profile_size": 32,
                # "model_architecture": { ... },
                # "diffusion": { ... }
            },
        }
    })
    return RNAPredictor(cfg)
