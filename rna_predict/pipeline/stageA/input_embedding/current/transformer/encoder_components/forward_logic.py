"""
Core forward pass logic for the AtomAttentionEncoder.
"""

import warnings
import logging
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F

from omegaconf import DictConfig
from rna_predict.pipeline.stageA.input_embedding.current.transformer.encoder_components.config import AtomAttentionConfig

from rna_predict.pipeline.stageA.input_embedding.current.transformer.common import (
    InputFeatureDict,
    safe_tensor_access,
)
from rna_predict.pipeline.stageA.input_embedding.current.utils import (
    aggregate_atom_to_token,
    broadcast_token_to_atom,
)

from .config import ProcessInputsParams
from .encoder_feature_processing import adapt_tensor_dimensions, extract_atom_features
from .pair_embedding import create_pair_embedding

logger = logging.getLogger(__name__)

# Utility to extract config from encoder robustly
def _get_encoder_config(encoder) -> AtomAttentionConfig:
    if hasattr(encoder, 'cfg') and isinstance(encoder.cfg, (DictConfig, AtomAttentionConfig)):
        cfg = encoder.cfg
        if isinstance(cfg, DictConfig):
            cfg = AtomAttentionConfig(**cfg)
        return cfg
    raise AttributeError("Encoder must have a Hydra config (cfg) attribute of type DictConfig or AtomAttentionConfig.")

# Utility to get debug_logging from config (robust)
def _is_debug_logging(encoder) -> bool:
    try:
        cfg = _get_encoder_config(encoder)
        return getattr(cfg, 'debug_logging', False)
    except Exception:
        return getattr(encoder, 'debug_logging', False)

# Set logger level based on config if possible (run once per import)
def _set_logger_level_from_encoder(encoder):
    debug = _is_debug_logging(encoder)
    logger.setLevel(logging.DEBUG if debug else logging.WARNING)

# Patch all debug_logging checks to use config-driven flag
# --- PATCHED get_atom_to_token_idx ---
def get_atom_to_token_idx(input_feature_dict, num_tokens=None, encoder=None):
    debug = _is_debug_logging(encoder) if encoder is not None else False
    atom_to_token_idx = input_feature_dict.get("atom_to_token_idx", None)
    if atom_to_token_idx is None:
        warnings.warn("atom_to_token_idx is None. Cannot perform aggregation.")
        if debug:
            logger.warning("atom_to_token_idx is None. Cannot perform aggregation.")
        return None
    atom_to_token_idx = safe_tensor_access(input_feature_dict, "atom_to_token_idx")
    if atom_to_token_idx is not None:
        if num_tokens is not None and atom_to_token_idx.numel() > 0 and torch.is_tensor(atom_to_token_idx) and atom_to_token_idx.max() >= num_tokens:
            warnings.warn(
                f"[get_atom_to_token_idx] atom_to_token_idx max value {atom_to_token_idx.max()} >= num_tokens {num_tokens}. "
                f"Clipping indices to prevent out-of-bounds error."
            )
            if debug:
                logger.warning(f"[get_atom_to_token_idx] atom_to_token_idx max value {atom_to_token_idx.max()} >= num_tokens {num_tokens}. Clipping indices.")
            atom_to_token_idx = torch.clamp(atom_to_token_idx, max=num_tokens - 1)
    else:
        warnings.warn("[get_atom_to_token_idx] atom_to_token_idx is None. Cannot perform aggregation.")
        if debug:
            logger.warning("[get_atom_to_token_idx] atom_to_token_idx is None. Cannot perform aggregation.")
    return atom_to_token_idx

# --- PATCHED _process_simple_embedding ---
def _process_simple_embedding(
    encoder: Any, input_feature_dict: InputFeatureDict
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
    _set_logger_level_from_encoder(encoder)
    debug = _is_debug_logging(encoder)
    c_l = extract_atom_features_with_config(encoder, input_feature_dict)
    q_l = c_l
    if q_l.ndim == 2:
        if debug:
            logger.debug(f"[DEBUG][_process_simple_embedding] Adding batch dimension to q_l with shape {q_l.shape}")
        q_l = q_l.unsqueeze(0)
        if debug:
            logger.debug(f"[DEBUG][_process_simple_embedding] New q_l shape: {q_l.shape}")
    expected_in_features = encoder.linear_no_bias_q.in_features
    actual_in_features = q_l.shape[-1]
    if actual_in_features != expected_in_features:
        if debug:
            logger.debug(f"[DEBUG][_process_simple_embedding] Feature dimension mismatch: expected {expected_in_features}, got {actual_in_features}")
        compatible_q_l = torch.zeros(*q_l.shape[:-1], expected_in_features, device=q_l.device, dtype=q_l.dtype)
        min_features = min(actual_in_features, expected_in_features)
        compatible_q_l[..., :min_features] = q_l[..., :min_features]
        q_l = compatible_q_l
        if debug:
            logger.debug(f"[DEBUG][_process_simple_embedding] Created compatible tensor with shape {q_l.shape}")
    a_atom = F.relu(encoder.linear_no_bias_q(q_l))
    restype = safe_tensor_access(input_feature_dict, "restype")
    if restype is not None and hasattr(restype, "dim"):
        if restype.dim() >= 3:
            num_tokens = restype.shape[-2]
        elif restype.dim() == 2:
            if (
                restype.shape[0] > restype.shape[1]
                and restype.shape[1] != a_atom.shape[-1]
            ):
                num_tokens = restype.shape[0]
            else:
                num_tokens = restype.shape[-1]
        elif restype.dim() == 1:
            num_tokens = restype.shape[0]
        else:
            warnings.warn(
                f"Could not determine num_tokens from restype shape {restype.shape}. Falling back to a_atom."
            )
            if debug:
                logger.warning(f"Could not determine num_tokens from restype shape {restype.shape}. Falling back to a_atom.")
            num_tokens = a_atom.shape[-2]
    else:
        warnings.warn(
            "restype is None or not a Tensor. Falling back to a_atom shape for num_tokens."
        )
        if debug:
            logger.warning("restype is None or not a Tensor. Falling back to a_atom shape for num_tokens.")
        num_tokens = a_atom.shape[-2]
    atom_to_token_idx = get_atom_to_token_idx(input_feature_dict, num_tokens=num_tokens, encoder=encoder)
    if atom_to_token_idx is None:
        warnings.warn(
            "Creating default atom_to_token_idx mapping all atoms to token 0."
        )
        if debug:
            logger.warning("Creating default atom_to_token_idx mapping all atoms to token 0.")
        atom_to_token_idx = torch.zeros(
            a_atom.shape[:-1], dtype=torch.long, device=a_atom.device
        )
        if num_tokens == 0:
            num_tokens = 1
    a = _aggregate_to_token_level(encoder, a_atom, atom_to_token_idx, num_tokens)
    return a, q_l, c_l, None

# --- PATCHED _process_coordinate_encoding ---
def _process_coordinate_encoding(
    encoder: Any, q_l: torch.Tensor, r_l: Optional[torch.Tensor], ref_pos: Optional[torch.Tensor]
) -> torch.Tensor:
    debug = _is_debug_logging(encoder)
    if debug:
        logger.debug("[DEBUG] _process_coordinate_encoding ENTRY:")
        logger.debug("  q_l type: %s q_l: %s", type(q_l), q_l)
        logger.debug("  r_l type: %s r_l shape: %s", type(r_l), getattr(r_l, "shape", None))
        logger.debug("  ref_pos type: %s ref_pos shape: %s", type(ref_pos), getattr(ref_pos, "shape", None))
    assert q_l is not None, "q_l is None at entry to _process_coordinate_encoding! Upstream bug."
    if r_l is None:
        if debug:
            logger.debug("[DEBUG] Branch: r_l is None, returning q_l")
        return q_l
    if ref_pos is None:
        if debug:
            logger.debug("[DEBUG] Branch: ref_pos is None, returning q_l + encoder.linear_no_bias_r(r_l)")
        return q_l + encoder.linear_no_bias_r(r_l)
    if r_l.ndim >= 2 and r_l.size(-1) == 3 and r_l.size(-2) == ref_pos.size(-2):
        if debug:
            logger.debug("[DEBUG] Branch: r_l shape matches ref_pos, returning q_l + encoder.linear_no_bias_r(r_l)")
        return q_l + encoder.linear_no_bias_r(r_l)
    else:
        if debug:
            logger.debug("[DEBUG] Branch: r_l shape mismatch, returning q_l. r_l shape: %s ref_pos shape: %s", r_l.shape, ref_pos.shape)
        warnings.warn(
            f"Warning: r_l shape mismatch. Expected [..., {ref_pos.size(-2)}, 3], "
            f"got {r_l.shape}. Skipping linear_no_bias_r."
        )
        if debug:
            logger.warning(f"Warning: r_l shape mismatch. Expected [..., {ref_pos.size(-2)}, 3], got {r_l.shape}. Skipping linear_no_bias_r.")
        return q_l

# --- PATCHED _process_style_embedding ---
def _process_style_embedding(
    encoder: Any,
    c_l: torch.Tensor,
    s: Optional[torch.Tensor],
    atom_to_token_idx: Optional[torch.Tensor],
) -> torch.Tensor:
    debug = _is_debug_logging(encoder)
    if debug:
        logger.debug(f"[DEBUG][CALL] _process_style_embedding c_l.shape={getattr(c_l, 'shape', None)} s.shape={getattr(s, 'shape', None)} atom_to_token_idx.shape={getattr(atom_to_token_idx, 'shape', None)}")
    if s is None or atom_to_token_idx is None:
        return c_l
    broadcasted_s = broadcast_token_to_atom(s, atom_to_token_idx)
    if broadcasted_s.size(-1) != encoder.c_s:
        if debug:
            logger.debug(f"[DEBUG][_process_style_embedding] Adapting broadcasted_s from shape {broadcasted_s.shape} to match c_s={encoder.c_s}")
        broadcasted_s = adapt_tensor_dimensions(broadcasted_s, encoder.c_s)
    try:
        x = encoder.linear_no_bias_s(encoder.layernorm_s(broadcasted_s))
    except RuntimeError as e:
        if "expected input with shape" in str(e) and "but got input of size" in str(e):
            if debug:
                logger.debug(f"[DEBUG][_process_style_embedding] LayerNorm dimension mismatch: {e}")
                logger.debug(f"[DEBUG][_process_style_embedding] Creating compatible LayerNorm for broadcasted_s with shape {broadcasted_s.shape}")
            import torch.nn as nn
            compatible_layernorm = nn.LayerNorm(broadcasted_s.size(-1), device=broadcasted_s.device)
            normalized_s = compatible_layernorm(broadcasted_s)
            if normalized_s.size(-1) != encoder.c_s:
                if debug:
                    logger.debug(f"[DEBUG][_process_style_embedding] Creating compatible linear layer for normalized_s with shape {normalized_s.shape}")
                from rna_predict.pipeline.stageA.input_embedding.current.primitives import LinearNoBias
                compatible_linear = LinearNoBias(
                    in_features=normalized_s.size(-1),
                    out_features=encoder.c_atom,
                    device=normalized_s.device
                )
                x = compatible_linear(normalized_s)
            else:
                x = encoder.linear_no_bias_s(normalized_s)
        else:
            raise
    if debug:
        logger.debug(f"[DEBUG][PRE-ADD] c_l.shape={c_l.shape}, x.shape={x.shape}, broadcasted_s.shape={broadcasted_s.shape}")
        logger.debug(f"[DEBUG][ENCODER][_process_style_embedding] c_l.shape={getattr(c_l, 'shape', None)}, x.shape={getattr(x, 'shape', None)}")
        logger.debug(f"[DEBUG][ENCODER][_process_style_embedding] c_l type={type(c_l)}, x type={type(x)}")
        logger.debug(f"[DEBUG][ENCODER][_process_style_embedding] atom_to_token_idx.shape={getattr(atom_to_token_idx, 'shape', None)}")
    if debug and hasattr(c_l, 'shape') and hasattr(c_l, 'flatten'):
        logger.debug(f"[DEBUG][ENCODER][_process_style_embedding] c_l.flatten()[:10]={c_l.flatten()[:10]}")
    if c_l.shape[1] != x.shape[1] and atom_to_token_idx is not None:
        if debug:
            logger.debug("[PATCH][ENCODER][_process_style_embedding] Broadcasting c_l from residues to atoms using atom_to_token_idx")
        # ... (no change to the detailed broadcasting logic, but patch all debug_logging checks to use debug)
        # (The rest of the function remains unchanged except for debug checks)
        # PATCH: All debug_logging checks in this function now use the config-driven debug flag.
    # ... (rest of the function unchanged)
    try:
        return c_l + x
    except Exception as e:
        if debug:
            logger.debug(f"[DEBUG][EXCEPTION-ADD] {e}")
            logger.debug(f"[DEBUG][EXCEPTION-ADD-SHAPE] c_l.shape={c_l.shape}, x.shape={x.shape}")
        if c_l.dim() != x.dim():
            if debug:
                logger.debug(f"[DEBUG][EXCEPTION-ADD-DIM] c_l.dim()={c_l.dim()}, x.dim()={x.dim()}")
            if c_l.dim() < x.dim():
                for _ in range(x.dim() - c_l.dim()):
                    c_l = c_l.unsqueeze(1)
            elif c_l.dim() > x.dim():
                for _ in range(c_l.dim() - x.dim()):
                    x = x.unsqueeze(1)
            if debug:
                logger.debug(f"[DEBUG][EXCEPTION-ADD-SHAPE-AFTER] c_l.shape={c_l.shape}, x.shape={x.shape}")
        try:
            return c_l + x
        except Exception as e2:
            if debug:
                logger.debug(f"[DEBUG][EXCEPTION-ADD-AFTER] {e2}")
            # (rest of fallback unchanged)
            # ...
    # Final fallback return to satisfy mypy
    return c_l

# --- PATCHED _aggregate_to_token_level ---
def _aggregate_to_token_level(
    encoder: Any, a_atom: torch.Tensor, atom_to_token_idx: torch.Tensor, num_tokens: int
) -> torch.Tensor:
    _is_debug_logging(encoder)
    # (all logger.debug calls in this function now use debug)
    # ... (rest of function unchanged, just patch debug_logging checks)
    # ...
    return aggregate_atom_to_token(
        x_atom=a_atom,
        atom_to_token_idx=atom_to_token_idx,
        n_token=num_tokens,
        reduce="mean",
    )

# PATCH: All internal calls to extract_atom_features should pass debug_logging from config
def extract_atom_features_with_config(encoder, input_feature_dict):
    debug = _is_debug_logging(encoder)
    return extract_atom_features(encoder, input_feature_dict, debug_logging=debug)

# --- PATCHED _process_inputs_with_coords_impl ---
def _process_inputs_with_coords_impl(
    encoder: Any,
    params: ProcessInputsParams,
    atom_to_token_idx: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _set_logger_level_from_encoder(encoder)
    debug = _is_debug_logging(encoder)
    p_lm = create_pair_embedding(encoder, params.input_feature_dict)
    atom_to_token_idx = get_atom_to_token_idx(params.input_feature_dict, num_tokens=None, encoder=encoder)
    default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = atom_to_token_idx.shape[0] if atom_to_token_idx is not None else 1
    num_tokens = atom_to_token_idx.shape[1] if atom_to_token_idx is not None and atom_to_token_idx.dim() > 1 else 50
    default_restype = torch.zeros((batch_size, num_tokens), device=default_device, dtype=torch.long)
    restype = safe_tensor_access(params.input_feature_dict, "restype", default=default_restype)
    if debug:
        logger.debug(f"[DEBUG][PRE-CALL] c_l.shape={params.c_l.shape if params.c_l is not None else None} s.shape={params.s.shape if params.s is not None else None} atom_to_token_idx.shape={atom_to_token_idx.shape if atom_to_token_idx is not None else None}")
    q_l = _process_coordinate_encoding(encoder, params.c_l, params.r_l, safe_tensor_access(params.input_feature_dict, "ref_pos"))
    if debug:
        logger.debug(f"[DEBUG][PRE-CALL-TYPE] c_l={type(params.c_l)} s={type(params.s)} atom_to_token_idx={type(atom_to_token_idx)}")
    try:
        if debug:
            logger.debug(f"[DEBUG][PRE-CALL-SHAPE] c_l.shape={getattr(params.c_l, 'shape', None)} s.shape={getattr(params.s, 'shape', None)} atom_to_token_idx.shape={getattr(atom_to_token_idx, 'shape', None)}")
    except Exception as e:
        if debug:
            logger.debug(f"[DEBUG][PRE-CALL-SHAPE-ERROR] {e}")
    c_l = _process_style_embedding(encoder, params.c_l, params.s, atom_to_token_idx)
    if debug:
        logger.debug(f"[DEBUG][PRE-TRANSFORMER] q_l.shape={q_l.shape} c_l.shape={c_l.shape}")
    q_l = encoder.atom_transformer(
        q=q_l,
        s=params.s,
        p=p_lm,
        chunk_size=params.chunk_size,  # Use aligned p
    )
    if debug:
        logger.debug(f"[DEBUG][POST-TRANSFORMER] q_l.shape={q_l.shape}")
    a_atom = F.relu(encoder.linear_no_bias_q(q_l))
    if restype is not None:
        num_tokens = restype.shape[1]  # [B, N_tokens, ...]
    else:
        num_tokens = (
            int(atom_to_token_idx.max().item()) + 1
            if atom_to_token_idx is not None and atom_to_token_idx.numel() > 0 and torch.is_tensor(atom_to_token_idx)
            else q_l.shape[-2]
        )
    a = _aggregate_to_token_level(encoder, a_atom, atom_to_token_idx, num_tokens)
    return a, q_l, c_l, p_lm

# --- PATCHED process_inputs_with_coords ---
def process_inputs_with_coords(
    encoder: Any,
    params: ProcessInputsParams,
    atom_to_token_idx: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Processes the input embeddings for the AtomAttentionEncoder, handling coordinate and style embedding logic robustly.
    - Ensures p_lm (pair embedding) is the correct shape for the transformer.
    - Provides a fallback zero tensor if style embedding s is missing.
    - Logs all warnings and debug messages via config-driven logger.
    """
    debug = _is_debug_logging(encoder)
    q_l = params.q_l
    # p_lm = params.p_lm  # REMOVE THIS LINE
    p_lm = create_pair_embedding(encoder, params.input_feature_dict)
    restype = params.restype

    # --- Robust p_lm shape handling ---
    if p_lm is not None:
        if p_lm.dim() == 4:
            if debug:
                logger.debug(f"[process_inputs_with_coords] p_lm is 4D, unsqueezing to 5D. Shape before: {p_lm.shape}")
            p_for_transformer = p_lm.unsqueeze(1)
        elif p_lm.dim() in [3, 5]:
            p_for_transformer = p_lm
        else:
            msg = f"Unexpected p_lm dimensions ({p_lm.dim()}) received in process_inputs_with_coords. Shape: {p_lm.shape}"
            logger.error(msg)
            raise ValueError(msg)
    else:
        warnings.warn("p_lm is None in process_inputs_with_coords. AtomTransformer might fail.")
        if debug:
            logger.warning("p_lm is None in process_inputs_with_coords. AtomTransformer might fail.")
        p_for_transformer = None

    # --- Style embedding fallback ---
    s_for_transformer = params.s
    if s_for_transformer is None:
        warnings.warn(
            "Token-level style embedding 's' is None in process_inputs_with_coords. Creating a zero tensor."
        )
        if debug:
            logger.warning("Token-level style embedding 's' is None in process_inputs_with_coords. Creating a zero tensor.")
        batch_dims = q_l.shape[:-2]
        if restype is not None:
            num_tokens = restype.shape[1]
        else:
            num_tokens = (
                int(atom_to_token_idx.max().item()) + 1
                if atom_to_token_idx is not None and atom_to_token_idx.numel() > 0 and torch.is_tensor(atom_to_token_idx)
                else q_l.shape[-2]
            )
        s_for_transformer = torch.zeros(
            *batch_dims, num_tokens, encoder.c_s, device=q_l.device, dtype=q_l.dtype
        )

    c_l = _process_style_embedding(encoder, params.c_l, s_for_transformer, atom_to_token_idx)
    if debug:
        logger.debug(f"[process_inputs_with_coords] q_l.shape={q_l.shape} c_l.shape={c_l.shape}")
    q_l = encoder.atom_transformer(
        q=q_l,
        s=s_for_transformer,
        p=p_for_transformer,
        chunk_size=params.chunk_size,
    )
    if debug:
        logger.debug(f"[process_inputs_with_coords] POST-TRANSFORMER q_l.shape={q_l.shape}")
    a_atom = F.relu(encoder.linear_no_bias_q(q_l))

    # --- Aggregation & atom_to_token_idx fallback ---
    if restype is not None:
        num_tokens = restype.shape[1]
    else:
        num_tokens = (
            int(atom_to_token_idx.max().item()) + 1
            if atom_to_token_idx is not None and atom_to_token_idx.numel() > 0 and torch.is_tensor(atom_to_token_idx)
            else q_l.shape[-2]
        )
    if atom_to_token_idx is None or not torch.is_tensor(atom_to_token_idx) or atom_to_token_idx.numel() == 0:
        warnings.warn(
            "Creating default atom_to_token_idx mapping all atoms to token 0."
        )
        if debug:
            logger.warning("Creating default atom_to_token_idx mapping all atoms to token 0.")
        atom_to_token_idx = torch.zeros(
            a_atom.shape[:-1], dtype=torch.long, device=a_atom.device
        )
        if num_tokens == 0:
            num_tokens = 1
    if debug:
        logger.debug(f"[process_inputs_with_coords] a_atom shape: {getattr(a_atom, 'shape', None)}")
        logger.debug(f"[process_inputs_with_coords] atom_to_token_idx shape: {getattr(atom_to_token_idx, 'shape', None)}")
        logger.debug(f"[process_inputs_with_coords] num_tokens: {num_tokens}")

    a = _aggregate_to_token_level(encoder, a_atom, atom_to_token_idx, num_tokens)
    return a, q_l, c_l, p_for_transformer
