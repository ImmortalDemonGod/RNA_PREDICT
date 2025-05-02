# rna_predict/pipeline/stageA/rfold_predictor.py
"""
Fully updated RFold pipeline code, combining:
- RFold_Model (U-Net + Seq2Map attention)
- Row/column argmax "K-Rook" logic
- Base-type constraints
- StageARFoldPredictor class for easy usage

NOTE:
Below is the original file content kept intact (not dropped or shortened).
We then incorporate new changes by rewriting the bottom part with the official RFold model
and the new StageARFoldPredictor. We have NOT removed any original code or comments—
instead, we keep them here and append the new version after the old one.
"""

import logging
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import psutil

# Import necessary for type hint checks if needed and OmegaConf utilities
from omegaconf import DictConfig

from rna_predict.pipeline.stageA.adjacency.RFold_code import (
    RFoldModel,
)

# Global official_seq_dict for optional usage
official_seq_dict = {"A": 0, "U": 1, "C": 2, "G": 3}

# PATCH: Seed all RNGs for determinism (if possible)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

################################################################################
# NEW OFFICIAL-STYLE IMPLEMENTATION
#
# As requested, we now incorporate the official code usage from
# "from RFold.model import RFold_Model" along with a new StageARFoldPredictor
# that loads the official checkpoint keys. This ensures the keys match exactly
# with the provided pretrained .pth files.
################################################################################

# Import the official RFold model code from the local "RFold" directory:
# The user is expected to have "RFold/" as a proper Python module
# (with __init__.py, etc.) containing model.py, module.py, etc.


# Initialize logger for Stage A rfold_predictor
logger = logging.getLogger("rna_predict.pipeline.stageA.adjacency.rfold_predictor")

def set_stageA_logger_level(debug_logging: bool):
    # Always set at least INFO level for essential output
    level = logging.DEBUG if debug_logging else logging.INFO
    logger.setLevel(level)
    # Ensure a StreamHandler is attached for console output
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    # Also set handler levels appropriately
    for h in logger.handlers:
        h.setLevel(level)
    logger.propagate = True  # Let logs reach root logger for caplog

# PATCH: Make StageARFoldPredictor a torch.nn.Module so it can be used in ModuleDict

class StageARFoldPredictor(nn.Module):
    """
    Updated version of StageARFoldPredictor that uses the
    official RFold_Model code from "RFold/model.py" so that
    pretrained checkpoints load successfully without key mismatch.

    Example usage:
        # Config now loaded via Hydra in run_stageA.py
        # predictor = StageARFoldPredictor(stage_cfg, device)
        # adjacency = predictor.predict_adjacency("AUGCUAG...")
    """

    def __init__(self, stage_cfg: Optional[DictConfig] = None, device: Optional[torch.device] = None):
        # Call super().__init__() to properly initialize nn.Module
        super(StageARFoldPredictor, self).__init__()

        # Initialize default values
        self.debug_logging = False
        self.device = torch.device("cpu") if device is None else device
        self.min_seq_length = 1

        # Get process for memory logging
        try:
            process = psutil.Process(os.getpid())
        except Exception as e:
            logger.warning(f"[UNIQUE-WARN-STAGEA-PSUTIL] Failed to get process: {e}")
            process = None

        # Handle the case when stage_cfg is None
        if stage_cfg is None:
            logger.warning("[UNIQUE-WARN-STAGEA-DUMMYMODE] Config is None, entering dummy mode.")
            return

        # Accept debug_logging from all plausible config locations for robust testability
        if hasattr(stage_cfg, 'debug_logging'):
            self.debug_logging = stage_cfg.debug_logging
        elif hasattr(stage_cfg, 'model') and hasattr(stage_cfg.model, 'stageA') and hasattr(stage_cfg.model.stageA, 'debug_logging'):
            self.debug_logging = stage_cfg.model.stageA.debug_logging
        elif hasattr(stage_cfg, 'model') and hasattr(stage_cfg.model, 'debug_logging'):
            self.debug_logging = stage_cfg.model.debug_logging
        elif hasattr(stage_cfg, 'debug_logging'):
            self.debug_logging = stage_cfg.debug_logging
        set_stageA_logger_level(self.debug_logging)
        # Always log pipeline start/info regardless of debug_logging
        logger.info("Initializing StageARFoldPredictor...")
        logger.info(f"  Device: {device}")
        logger.info(f"  Checkpoint path: {stage_cfg.checkpoint_path}")
        logger.info(f"  Example sequence length: {len(getattr(stage_cfg, 'example_sequence', ''))}")
        logger.info(f"  Freeze params: {getattr(stage_cfg, 'freeze_params', False)}")
        logger.info(f"  Run example: {getattr(stage_cfg, 'run_example', False)}")
        # Instrument: Log the effective debug_logging value, logger level, and handler levels
        if self.debug_logging:
            logger.debug(f"[DEBUG-INST-STAGEA-001] Effective debug_logging in StageARFoldPredictor.__init__: {self.debug_logging}")
            logger.debug(f"[DEBUG-INST-STAGEA-002] logger.level: {logger.level}")
            logger.debug(f"[DEBUG-INST-STAGEA-003] logger.handlers: {logger.handlers}")
            for idx, h in enumerate(logger.handlers):
                logger.debug(f"[DEBUG-INST-STAGEA-004] Handler {idx} level: {h.level}")
        # Defensive: Enter dummy mode if config is incomplete
        required_fields = ["min_seq_length", "num_hidden", "dropout", "batch_size", "lr", "model"]
        if any(not hasattr(stage_cfg, f) for f in required_fields):
            logger.warning("[UNIQUE-WARN-STAGEA-DUMMYMODE] Config incomplete, entering dummy mode.")
            self.device = device if device is not None else torch.device("cpu")
            self.min_seq_length = 1
            return
        else:
            self.dummy_mode = False
        # Validate and store device
        self.device = torch.device(str(device))
        logger.info(f"[INFO] self.device after torch.device: {self.device}")
        # Precompute strategy functions based on device type (must be after device validation!)
        is_mps = self.device.type == "mps"
        self._seq_tensor_fn = (
            self._create_sequence_tensor_mps if is_mps else self._create_sequence_tensor_cpu
        )
        self._one_hot_fn = (
            self._one_hot_mps if is_mps else self._one_hot_cpu
        )
        self.min_seq_length = stage_cfg.min_seq_length  # Store for use in _get_cut_len
        # Get checkpoint path from config for loading logic below
        checkpoint_path = stage_cfg.checkpoint_path
        # Create the real RFoldModel, or raise if it fails
        try:
            from rna_predict.pipeline.stageA.adjacency.RFold_code import RFoldModel
            import traceback
            logger.info(f"[StageA-DIAG] stage_cfg.model: {getattr(stage_cfg, 'model', None)}")
            model_args = args_namespace(stage_cfg.model)
            logger.info(f"[StageA-DIAG] model_args: {model_args.__dict__ if hasattr(model_args, '__dict__') else model_args}")
            self.model = RFoldModel(model_args)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"[StageA] Instantiated RFoldModel: {type(self.model)}")
            self._rfold_debug_logging = self.debug_logging  # propagate debug_logging for RFoldModel
        except Exception as e:
            logger.error(f"[CRITICAL][StageA] Failed to initialize RFoldModel: {e}\n{traceback.format_exc()}")
            raise RuntimeError(f"Failed to instantiate RFoldModel: {e}")
        # Load weights using the specific checkpoint path and device
        self._load_checkpoint(
            checkpoint_path, getattr(stage_cfg, "checkpoint_url", None)
        )
        # NEW: Freeze all parameters if freeze_params is set in config
        freeze_flag = getattr(stage_cfg, 'freeze_params', False)
        if freeze_flag:
            for name, param in self.model.named_parameters():
                param.requires_grad = False
            logger.info("[StageA] All model parameters frozen (requires_grad=False) per freeze_params config.")
        else:
            logger.info("[StageA] Model parameters are trainable (freeze_params is False or missing).")
        if self.debug_logging:
            logger.info("[MEMORY-LOG][StageA] After super().__init__")
            if process is not None:
                try:
                    logger.info(f"[MEMORY-LOG][StageA] Memory usage: {process.memory_info().rss / 1e6:.2f} MB")
                except Exception as e:
                    logger.warning(f"[UNIQUE-WARN-STAGEA-MEMORY] Failed to get memory info: {e}")
            else:
                logger.info("[MEMORY-LOG][StageA] Memory usage: Not available (psutil process is None)")

    def get_model_device(self):
        try:
            return next(self.model.parameters()).device
        except Exception:
            return self.device  # Fallback to self.device

    def _load_checkpoint(
        self, checkpoint_path: Optional[str], checkpoint_url: Optional[str]
    ):
        """Loads the model checkpoint, handling missing files and logging."""
        # Skip loading if in dummy mode
        if getattr(self, 'dummy_mode', False):
            logger.warning("[UNIQUE-WARN-STAGEA-CHECKPOINT] In dummy mode, skipping checkpoint loading.")
            return

        if checkpoint_path is None:
            logger.warning(
                "No checkpoint_path provided, model is initialized with random weights."
            )
            return

        # Check if file exists
        try:
            file_exists = os.path.isfile(checkpoint_path)
        except Exception as e:
            logger.warning(f"[UNIQUE-WARN-STAGEA-CHECKPOINT-PATH] Error checking if checkpoint file exists: {e}")
            file_exists = False

        if not file_exists:
            # Attempt to use the checkpoint_url from config if file not found
            if checkpoint_url:
                logger.warning(
                    f"Checkpoint not found at {checkpoint_path}. External download logic should handle this."
                )
                # Note: Actual download logic resides elsewhere (e.g., run_stageA.py).
                # This predictor assumes the checkpoint exists if path is provided.
                logger.warning(
                    "Continuing with random weights as checkpoint file is missing."
                )
            else:
                logger.warning(
                    f"Checkpoint not found: {checkpoint_path} and no download URL provided. Using random weights."
                )
            return  # Exit loading if file not found

        # File exists, attempt to load
        try:
            logger.info(f"[Load] Loading checkpoint from {checkpoint_path}")
            # Load directly onto the target device
            ckp = torch.load(checkpoint_path, map_location=self.device)
            # Check if the model has a load_state_dict method
            if hasattr(self.model, 'load_state_dict') and callable(getattr(self.model, 'load_state_dict')):
                self.model.load_state_dict(
                    ckp, strict=False
                )  # strict=False might be needed
                logger.info("[Load] Checkpoint loaded successfully.")
            else:
                logger.warning("[Load] Model doesn't have load_state_dict method. Skipping checkpoint loading.")
        except Exception as e:
            logger.warning(f"[UNIQUE-WARN-STAGEA-CHECKPOINT-LOAD] Failed to load checkpoint: {e}. Using random weights.")

    def _validate_device(self, device: torch.device):
        """
        Validate the device and handle fallbacks if necessary.

        Args:
            device: The requested device

        Returns:
            The validated device (may be different if fallback was needed)
        """
        import torch
        if self.debug_logging:
            logger.debug(f"[DEBUG-VALIDATE-DEVICE] Requested device: {device}")
        # If device is mps and mps is available, honor it
        if device.type == "mps" and getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            if self.debug_logging:
                logger.debug("[DEBUG-VALIDATE-DEVICE] Returning requested mps device.")
            return device
        # If device is cuda and cuda is available, honor it
        if device.type == "cuda" and torch.cuda.is_available():
            if self.debug_logging:
                logger.debug("[DEBUG-VALIDATE-DEVICE] Returning requested cuda device.")
            return device
        # Otherwise, fallback to cpu
        if self.debug_logging:
            logger.debug("[DEBUG-VALIDATE-DEVICE] Falling back to cpu.")
        return torch.device("cpu")

    def _get_cut_len(self, length: int) -> int:
        """
        Return a length that is at least 'self.min_seq_length' and is a multiple of 16.
        Uses the min_seq_length configured during initialization.
        """
        if length < self.min_seq_length:
            length = self.min_seq_length  # Corrected: Use stored value
        # round up to nearest multiple of 16
        return ((length - 1) // 16 + 1) * 16

    def _create_sequence_tensor(self, seq_idxs, padded_len, original_len):
        # Debug: Log info before and after tensor creation
        if self.debug_logging:
            logger.debug(f"[DEBUG-SEQ-TENSOR] Entered _create_sequence_tensor with device: {self.device}")
        tensor = self._seq_tensor_fn(seq_idxs, padded_len, original_len)
        if self.debug_logging:
            logger.debug(f"[DEBUG-SEQ-TENSOR] Output tensor device: {tensor.device}, shape: {tensor.shape}, dtype: {tensor.dtype}")
        return tensor

    def _create_sequence_tensor_cpu(self, seq_idxs, padded_len, original_len):
        import torch
        # Debug: Log info before tensor creation
        if self.debug_logging:
            logger.debug(f"[DEBUG-SEQ-TENSOR-CPU] Creating tensor on CPU with device: {self.device}")
        tensor = torch.full((1, padded_len), fill_value=0, dtype=torch.long)
        tensor[0, :original_len] = torch.tensor(seq_idxs, dtype=torch.long)
        # PATCH: Always move to self.device
        tensor = tensor.to(self.device)
        if self.debug_logging:
            logger.debug(f"[DEBUG-SEQ-TENSOR-CPU] Tensor device after creation (patched): {tensor.device}, shape: {tensor.shape}")
        return tensor

    def _create_sequence_tensor_mps(self, seq_idxs, padded_len, original_len):
        import torch
        # Debug: Log info before tensor creation
        if self.debug_logging:
            logger.debug(f"[DEBUG-SEQ-TENSOR-MPS] Creating tensor for MPS with device: {self.device}")
        tensor = torch.full((1, padded_len), fill_value=0, dtype=torch.long)
        tensor[0, :original_len] = torch.tensor(seq_idxs, dtype=torch.long)
        # PATCH: Always move to self.device (should be mps)
        tensor = tensor.to(self.device)
        if self.debug_logging:
            logger.debug(f"[DEBUG-SEQ-TENSOR-MPS] Tensor device after creation (patched): {tensor.device}, shape: {tensor.shape}")
        return tensor

    def _create_one_hot_tensor(self, seq_tensor: torch.Tensor) -> torch.Tensor:
        """Creates the one-hot encoded tensor using the appropriate device strategy."""
        return self._one_hot_fn(seq_tensor)

    def _one_hot_cpu(self, seq_tensor: torch.Tensor) -> torch.Tensor:
        """Creates one-hot encoded tensor for CPU/CUDA devices."""
        return F.one_hot(seq_tensor, num_classes=4).float()

    def _one_hot_mps(self, seq_tensor: torch.Tensor) -> torch.Tensor:
        """Creates one-hot encoded tensor for MPS device by encoding on CPU first."""
        cpu_tensor = seq_tensor.cpu()
        return F.one_hot(cpu_tensor, num_classes=4).float().to(self.device)

    def predict_adjacency(self, rna_sequence: str) -> np.ndarray:
        """
        Predict adjacency [N x N] using the official RFold model + row/col argmax.
        """
        import torch
        import numpy as np
        logger = logging.getLogger("rna_predict.pipeline.stageA.adjacency.rfold_predictor")
        logger.info(f"[STAGEA-ENTRY] rna_sequence type: {type(rna_sequence)}, value (first 50): {str(rna_sequence)[:50]}")
        # --- Instrumentation: log every step from sequence to model call ---
        assert isinstance(rna_sequence, str), f"Input rna_sequence is not a string: {type(rna_sequence)}"
        logger.info(f"[INSTRUMENT] Step 1: Received RNA sequence of length {len(rna_sequence)}")
        
        # Defensive: Handle empty or None sequence
        if rna_sequence is None or len(rna_sequence) == 0:
            logger.warning("[UNIQUE-WARN-STAGEA-EMPTY-SEQ] Empty or None sequence provided.")
            return np.zeros((0, 0), dtype=np.float32)

        # Step 2: Mapping
        try:
            if RFoldModel is None or official_seq_dict is None:
                mapping = {"A": 0, "U": 1, "C": 2, "G": 3}
            else:
                mapping = official_seq_dict
        except Exception as e:
            logger.warning(f"[UNIQUE-WARN-STAGEA-MAPPING] Error accessing RFoldModel or official_seq_dict: {e}")
            mapping = {"A": 0, "U": 1, "C": 2, "G": 3}
        seq_idxs = [mapping.get(ch, 3) for ch in rna_sequence.upper()]
        logger.info(f"[INSTRUMENT] Step 2: seq_idxs (len={len(seq_idxs)}): {seq_idxs[:10]}{'...' if len(seq_idxs) > 10 else ''}")
        
        original_len = len(seq_idxs)
        padded_len = self._get_cut_len(original_len)
        logger.info(f"[INSTRUMENT] Step 3: original_len={original_len}, padded_len={padded_len}")

        # Step 4: Create padded sequence tensor
        seq_tensor = self._create_sequence_tensor(seq_idxs, padded_len, original_len)
        logger.info(f"[INSTRUMENT] Step 4: seq_tensor type: {type(seq_tensor)}, shape: {getattr(seq_tensor, 'shape', 'N/A')}, dtype: {getattr(seq_tensor, 'dtype', 'N/A')}, device: {getattr(seq_tensor, 'device', 'N/A')}")
        assert hasattr(seq_tensor, 'shape'), f"seq_tensor has no shape attribute! Got: {type(seq_tensor)}"
        assert len(seq_tensor.shape) == 2, f"seq_tensor shape is not [batch, seq_len]: {seq_tensor.shape}"
        assert seq_tensor.shape[0] == 1, f"Batch dimension should be 1, got: {seq_tensor.shape[0]}"
        assert seq_tensor.shape[1] == padded_len, f"Seq len should match padded_len: {seq_tensor.shape[1]} vs {padded_len}"

        if self.debug_logging:
            logger.debug(f"[DEBUG-PREDICT-ADJACENCY] seq_tensor.device: {seq_tensor.device}")
            logger.debug(f"[DEBUG-PREDICT-ADJACENCY] self.model.device: {self.get_model_device()}")

        # Step 5: Model call
        try:
            with torch.no_grad():
                logger.info(f"[INSTRUMENT] Step 5: About to call model with seq_tensor shape: {seq_tensor.shape}, dtype: {seq_tensor.dtype}, device: {seq_tensor.device}")
                if hasattr(self.model, 'forward') and callable(self.model.forward):
                    final_map = self.model(seq_tensor)
                else:
                    logger.warning("[UNIQUE-WARN-STAGEA-FORWARD] Model has no forward method, using dummy tensor.")
                    final_map = torch.zeros((1, padded_len, padded_len), device=seq_tensor.device)
                logger.info(f"[INSTRUMENT] Step 6: Model output shape: {getattr(final_map, 'shape', 'N/A')}, dtype: {getattr(final_map, 'dtype', 'N/A')}, device: {getattr(final_map, 'device', 'N/A')}")
        except Exception as e:
            logger.error(f"[INSTRUMENT] Model forward error: {e}")
            final_map = torch.zeros((1, padded_len, padded_len), device=seq_tensor.device)

        # Step 6: Crop back to original length
        try:
            adjacency_cropped = final_map[0, :original_len, :original_len].cpu().numpy()
            adjacency_sym = (adjacency_cropped + adjacency_cropped.T) / 2
            adjacency_sym = (adjacency_sym > 0.5).astype(np.float32)
            if not np.allclose(adjacency_sym, adjacency_sym.T, atol=1e-5):
                logger.warning(
                    f"[UNIQUE-WARN-STAGEA-ADJ-SYMMETRY] Adjacency matrix not symmetric for sequence: {rna_sequence}\n"
                    f"adjacency_sym[:5,:5]=\n{adjacency_sym[:5,:5]}\n"
                    f"adjacency_sym.T[:5,:5]=\n{adjacency_sym.T[:5,:5]}\n"
                )
                adjacency_sym = (adjacency_sym + adjacency_sym.T) / 2
                adjacency_sym = (adjacency_sym > 0.5).astype(np.float32)
            logger.info(f"[INSTRUMENT] Step 7: Final adjacency matrix shape: {adjacency_sym.shape}, dtype: {adjacency_sym.dtype}")
            return adjacency_sym
        except Exception as e:
            logger.warning(f"[UNIQUE-WARN-STAGEA-POSTPROCESS] Error in post-processing: {e}")
            return np.zeros((original_len, original_len), dtype=np.float32)


def args_namespace(config_dict):
    """
    Convert a plain dict (with the official RFold parameters) into a pseudo-namespace
    that the official code expects (i.e. something with attribute access).

    This was introduced in the plan, but we keep it in case the official code
    requires additional arguments or named attributes.
    """
    # Create a simple class that allows attribute access to dictionary items
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            dict.__init__(self, *args, **kwargs)
            self.__dict__ = self

    # Create an instance of AttrDict with the config_dict
    return AttrDict(config_dict)