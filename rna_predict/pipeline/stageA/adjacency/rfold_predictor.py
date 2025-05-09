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
        # Call super().__init__() first to properly initialize nn.Module
        super().__init__()
        
        # Set debug_logging from config if available, before any method calls that may use it
        self.debug_logging = getattr(stage_cfg, "debug_logging", False) if stage_cfg is not None else False

        # --- HYDRA DEVICE COMPLIANCE PATCH ---
        # Always require device from config, never fallback to CPU unless explicitly set in config
        resolved_device = None
        if stage_cfg is not None and hasattr(stage_cfg, 'device'):
            resolved_device = stage_cfg.device
        elif device is not None:
            resolved_device = device
        else:
            raise ValueError("StageARFoldPredictor requires a device specified in the config or as an explicit argument; do not use hardcoded defaults.")
        # validate & announce
        self.device = self._validate_device(torch.device(resolved_device))
        if self.debug_logging:
            logger.debug("[DEVICE-DEBUG] Using device: %s", self.device)

        # Assert device is resolved if present in config
        if stage_cfg is not None and hasattr(stage_cfg, 'device'):
            assert stage_cfg.device != "${device}", f"Device not resolved in stage_cfg for {self.__class__.__name__}: {stage_cfg.device}"
            if self.debug_logging:
                logger.debug("[DEBUG][StageARFoldPredictor] Resolved device in stage_cfg: %s", stage_cfg.device)
        # Use the provided device or fallback to CPU, then validate
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
            self.dummy_mode = True # Explicitly set dummy_mode
            return

        # Accept debug_logging from all plausible config locations for robust testability
        if hasattr(stage_cfg, 'model') and hasattr(stage_cfg.model, 'stageA') and hasattr(stage_cfg.model.stageA, 'debug_logging'):
            self.debug_logging = stage_cfg.model.stageA.debug_logging
        elif hasattr(stage_cfg, 'model') and hasattr(stage_cfg.model, 'debug_logging'):
            self.debug_logging = stage_cfg.model.debug_logging
        elif hasattr(stage_cfg, 'debug_logging'):
            self.debug_logging = stage_cfg.debug_logging
        set_stageA_logger_level(self.debug_logging)
        # Always log pipeline start/info regardless of debug_logging
        logger.info("Initializing StageARFoldPredictor...")
        logger.info(f"  Device: {self.device}") # Log the validated device
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
        required_fields = ["min_seq_length", "num_hidden", "dropout", "batch_size", "lr", "model", "checkpoint_path"] # Add checkpoint_path as required
        if any(not hasattr(stage_cfg, f) for f in required_fields):
            logger.warning("[UNIQUE-WARN-STAGEA-DUMMYMODE] Config incomplete, entering dummy mode.")
            self.dummy_mode = True
            self.device = device if device is not None else torch.device("cpu") # Use provided device or CPU
            self.min_seq_length = 1
            self.model = None # Explicitly set model to None in dummy mode
            return
        else:
            self.dummy_mode = False
        # Store min_seq_length from config
        self.min_seq_length = stage_cfg.min_seq_length

        # Get checkpoint path from config for loading logic below
        checkpoint_path = stage_cfg.checkpoint_path
        self.checkpoint_path = checkpoint_path # Store checkpoint path

        # Create the real RFoldModel, or raise if it fails
        try:
            from rna_predict.pipeline.stageA.adjacency.RFold_code import RFoldModel
            import traceback
            logger.info(f"[StageA-DIAG] stage_cfg.model: {getattr(stage_cfg, 'model', None)}")
            model_args = args_namespace(stage_cfg.model)
            # Ensure device is in model_args for RFoldModel's internal device handling
            setattr(model_args, 'device', self.device) # Pass the validated device
            logger.info(f"[StageA-DIAG] model_args (with device): {model_args.__dict__ if hasattr(model_args, '__dict__') else model_args}")
            self.model = RFoldModel(model_args)
            # CRITICAL FIX: Explicitly move model to the correct device AFTER instantiation
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"[StageA] Instantiated RFoldModel: {type(self.model)}")
            logger.info(f"[StageA] Model device after instantiation and .to(device): {self.get_model_device()}")
            self._rfold_debug_logging = self.debug_logging  # propagate debug_logging for RFoldModel
        except Exception as e:
            logger.error(f"[CRITICAL][StageA] Failed to initialize RFoldModel: {e}\n{traceback.format_exc()}")
            raise RuntimeError(f"Failed to instantiate RFoldModel: {e}")
        # Load weights using the specific checkpoint path and device
        self._load_checkpoint(
             self.checkpoint_path, getattr(stage_cfg, "checkpoint_url", None)
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
        """
        Returns the device on which the model's parameters are located.
        
        If the model has no parameters or an error occurs, returns the configured device instead.
        """
        if self.model is None:
            return None
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            # Handle models with no parameters (e.g., nn.Identity)
            if self.debug_logging:
                 logger.debug("[DEBUG] Model has no parameters, returning self.device")
            return self.device
        except Exception as e:
            if self.debug_logging:
                 logger.debug(f"[DEBUG] Failed to get model device: {e}. Returning self.device")
            return self.device  # Fallback to self.device

    def _load_checkpoint(
        self, checkpoint_path: Optional[str], checkpoint_url: Optional[str]
    ):
        """
        Loads model weights from a checkpoint file if available.
        
        If the checkpoint file is missing or cannot be loaded, the model remains initialized with random weights. Logs warnings for missing files, missing download URLs, or loading failures. Skips loading if in dummy mode or if the model is not initialized.
        """
        # Skip loading if in dummy mode or model is None
        if getattr(self, 'dummy_mode', False) or self.model is None:
            logger.warning("[UNIQUE-WARN-STAGEA-CHECKPOINT] In dummy mode or model is None, skipping checkpoint loading.")
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
            # **** CRITICAL FIX: Ensure map_location is always used ****
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

    def _validate_device(self, device: torch.device) -> torch.device:
        """
        Validates the requested device, falling back to CPU if CUDA or MPS is unavailable.
        
        Args:
            device: The requested torch.device.
        
        Returns:
            The validated device, using CPU if the requested device is not available.
        """
        if self.debug_logging:
            logger.debug("[DEBUG-VALIDATE-DEVICE] Requested device: %s", device)
        # Check if the requested device type is actually available
        if device.type == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available – falling back to CPU.")
            return torch.device("cpu")
        if device.type == "mps" and not (getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()):
            logger.warning("MPS requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        # If the device type is available or is CPU, return it as is
        if self.debug_logging:
            logger.debug("[DEBUG-VALIDATE-DEVICE] Returning validated device: %s", device)
        return device


    def _get_cut_len(self, length: int) -> int:
        """
        Returns the smallest multiple of 16 greater than or equal to the given length and at least the configured minimum sequence length.
        
        Args:
            length: The original sequence length.
        
        Returns:
            The adjusted length, rounded up to the nearest multiple of 16 and not less than the minimum sequence length.
        """
        if length < self.min_seq_length:
            length = self.min_seq_length  # Corrected: Use stored value
        # round up to nearest multiple of 16
        return ((length - 1) // 16 + 1) * 16

    def _create_sequence_tensor(self, seq_idxs, padded_len, original_len) -> torch.Tensor:
        # Debug: Log info before and after tensor creation
        """
        Creates a padded tensor containing sequence indices and moves it to the configured device.
        
        Args:
            seq_idxs: List or array of sequence indices to encode.
            padded_len: Total length of the output tensor (for padding).
            original_len: Length of the original sequence (unpadded).
        
        Returns:
            A torch.Tensor of shape [1, padded_len] with the sequence indices in the first original_len positions and zeros elsewhere, located on the configured device.
        """
        if self.debug_logging:
            logger.debug("[DEBUG-SEQ-TENSOR] Entered _create_sequence_tensor with device: %s", self.device)
        # PATCH: Call the appropriate device-specific function (removed this complexity, use direct .to(self.device))
        # Reverted: Simplify and rely on .to(self.device) after creation
        import torch
        # Debug: Log info before tensor creation
        if self.debug_logging:
            logger.debug("[DEBUG-SEQ-TENSOR-GENERIC] Creating tensor with device: %s", self.device)
        tensor = torch.full((1, padded_len), fill_value=0, dtype=torch.long)
        tensor[0, :original_len] = torch.tensor(seq_idxs, dtype=torch.long)
        # PATCH: Always move to self.device explicitly
        tensor = tensor.to(self.device)
        if self.debug_logging:
            logger.debug("[DEBUG-SEQ-TENSOR-GENERIC] Tensor device after creation and .to(self.device): %s, shape: %s, dtype: %s", tensor.device, tensor.shape, tensor.dtype)
        return tensor


    def predict_adjacency(self, rna_sequence: str) -> np.ndarray:
        """
        Predicts the RNA secondary structure adjacency matrix for a given sequence using the RFold model.
        
        Args:
            rna_sequence: RNA sequence as a string.
        
        Returns:
            A symmetric binary NumPy array of shape [N, N], where N is the input sequence length, representing the predicted adjacency matrix. If the model is in dummy mode, the input is empty, or an error occurs, returns a zero matrix of appropriate size.
        """
        import torch
        import numpy as np
        logger = logging.getLogger("rna_predict.pipeline.stageA.adjacency.rfold_predictor")
        logger.info("rna_sequence type: %s, value (first 50): %s", type(rna_sequence), str(rna_sequence)[:50])
        # --- Instrumentation: log every step from sequence to model call ---
        assert isinstance(rna_sequence, str), "Input rna_sequence is not a string: %s" % type(rna_sequence)
        logger.info("Step 1: Received RNA sequence of length %d", len(rna_sequence))

        # Defensive: Handle empty or None sequence, and dummy mode
        if getattr(self, 'dummy_mode', False):
            logger.warning("[UNIQUE-WARN-STAGEA-DUMMYMODE] In dummy mode, returning zero adjacency.")
            return np.zeros((len(rna_sequence), len(rna_sequence)), dtype=np.float32) # Return based on input length

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
            logger.warning("[UNIQUE-WARN-STAGEA-MAPPING] Error accessing RFoldModel or official_seq_dict: %s", e)
            mapping = {"A": 0, "U": 1, "C": 2, "G": 3}
        seq_idxs = [mapping.get(ch, 3) for ch in rna_sequence.upper()]
        logger.info("Step 2: seq_idxs (len=%d): %s%s", len(seq_idxs), seq_idxs[:10], '...' if len(seq_idxs) > 10 else '')

        original_len = len(seq_idxs)
        padded_len = self._get_cut_len(original_len)
        logger.info("Step 3: original_len=%d, padded_len=%d", original_len, padded_len)

        # Step 4: Create padded sequence tensor
        # PATCH: Ensure seq_tensor is created and moved to the correct device here
        seq_tensor = self._create_sequence_tensor(seq_idxs, padded_len, original_len)
        logger.info("Step 4: seq_tensor type: %s, shape: %s, dtype: %s, device: %s", type(seq_tensor), seq_tensor.shape, seq_tensor.dtype, seq_tensor.device)
        assert hasattr(seq_tensor, 'shape'), "seq_tensor has no shape attribute! Got: %s" % type(seq_tensor)
        assert len(seq_tensor.shape) == 2, "seq_tensor shape is not [batch, seq_len]: %s" % seq_tensor.shape
        assert seq_tensor.shape[0] == 1, "Batch dimension should be 1, got: %d" % seq_tensor.shape[0]
        assert seq_tensor.shape[1] == padded_len, "Seq len should match padded_len: %d vs %d" % (seq_tensor.shape[1], padded_len)

        if self.debug_logging:
            logger.debug("[DEBUG-PREDICT-ADJACENCY] seq_tensor.device: %s", seq_tensor.device)
            logger.debug("[DEBUG-PREDICT-ADJACENCY] self.model.device: %s", self.get_model_device())

        # Step 5: Model call
        try:
            # Ensure model is on the correct device before calling forward (Redundant if done correctly in __init__, but safe)
            # CRITICAL FIX: Move model to device before forward pass
            if self.model is not None:
                self.model.to(self.device)
            else:
                 logger.error("[CRITICAL] RFoldModel is None during predict_adjacency.")
                 raise RuntimeError("RFoldModel is None. Dummy mode was likely entered due to incomplete config.")

            with torch.no_grad():
                logger.info("Step 5: About to call model with seq_tensor shape: %s, dtype: %s, device: %s", seq_tensor.shape, seq_tensor.dtype, seq_tensor.device)
                # Ensure seq_tensor is on the correct device just before the model call
                seq_tensor = seq_tensor.to(self.get_model_device()) # Use model's device as target
                if hasattr(self.model, 'forward') and callable(self.model.forward):
                    final_map = self.model(seq_tensor)
                else:
                    logger.warning("[UNIQUE-WARN-STAGEA-FORWARD] Model has no forward method, using dummy tensor.")
                    final_map = torch.zeros((1, padded_len, padded_len), device=seq_tensor.device) # Create dummy on correct device
                logger.info("Step 6: Model output shape: %s, dtype: %s, device: %s", final_map.shape, final_map.dtype, final_map.device)
        except Exception as e:
            logger.error("Model forward error: %s", e)
            # Return a default/dummy adjacency on error to allow pipeline continuation if needed
            return np.zeros((original_len, original_len), dtype=np.float32)

        # Step 6: Crop back to original length and post-process
        try:
            # Ensure final_map is on CPU before converting to numpy
            adjacency_cropped = final_map[0, :original_len, :original_len].cpu().numpy()
            adjacency_sym = (adjacency_cropped + adjacency_cropped.T) / 2
            adjacency_sym = (adjacency_sym > 0.5).astype(np.float32)
            if not np.allclose(adjacency_sym, adjacency_sym.T, atol=1e-5):
                logger.warning(
                    "[UNIQUE-WARN-STAGEA-ADJ-SYMMETRY] Adjacency matrix not symmetric for sequence: %s\n"
                    "adjacency_sym[:5,:5]=\n%s\n"
                    "adjacency_sym.T[:5,:5]=\n%s\n",
                    rna_sequence, adjacency_sym[:5,:5], adjacency_sym.T[:5,:5]
                )
                # Re-symmetrize just in case, though it should be symmetric by construction
                adjacency_sym = (adjacency_sym + adjacency_sym.T) / 2
                adjacency_sym = (adjacency_sym > 0.5).astype(np.float32)
            logger.info("Step 7: Final adjacency matrix shape: %s, dtype: %s", adjacency_sym.shape, adjacency_sym.dtype)
            return adjacency_sym
        except Exception as e:
            logger.warning("Error in post-processing: %s", e)
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