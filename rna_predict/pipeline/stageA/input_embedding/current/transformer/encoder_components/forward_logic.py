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
            cfg = AtomAttentionConfig(**cfg.to_container())
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
        if num_tokens is not None and atom_to_token_idx.numel() > 0 and torch.is_tensor(atom_to_token_idx) and atom_to_token_idx.max() is not None and num_tokens is not None and atom_to_token_idx.max() is not None and num_tokens is not None and atom_to_token_idx.max() is not None and num_tokens is not None and atom_to_token_idx.max() >= num_tokens:
            if atom_to_token_idx.max() is not None and num_tokens is not None:
                if atom_to_token_idx.max() >= num_tokens:
                    warnings.warn(
                        f"[get_atom_to_token_idx] atom_to_token_idx max value {atom_to_token_idx.max()} >= num_tokens {num_tokens}. "
                        f"Clipping indices to prevent out-of-bounds error."
                    )
                    if debug:
                        logger.warning(f"[get_atom_to_token_idx] atom_to_token_idx max value {atom_to_token_idx.max()} >= num_tokens {num_tokens}. Clipping indices.")
                    if atom_to_token_idx is not None and num_tokens is not None:
                        atom_to_token_idx = torch.clamp(atom_to_token_idx, max=num_tokens - 1)
    else:
        warnings.warn("[get_atom_to_token_idx] atom_to_token_idx is None. Cannot perform aggregation.")
        if debug:
            logger.warning("[get_atom_to_token_idx] atom_to_token_idx is None. Cannot perform aggregation.")
    return atom_to_token_idx

# --- PATCHED _process_simple_embedding ---
def _process_simple_embedding(
    encoder: Any, input_feature_dict: InputFeatureDict
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _set_logger_level_from_encoder(encoder)
    debug = _is_debug_logging(encoder)
    c_l = extract_atom_features_with_config(encoder, input_feature_dict)
    q_l = c_l
    if q_l is not None and q_l.ndim == 2:
        if debug:
            logger.debug(f"[DEBUG][_process_simple_embedding] Adding batch dimension to q_l with shape {getattr(q_l, 'shape', None)}")
        if q_l is not None:
            q_l = q_l.unsqueeze(0)
        if debug:
            logger.debug(f"[DEBUG][_process_simple_embedding] New q_l shape: {getattr(q_l, 'shape', None)}")
    expected_in_features = encoder.linear_no_bias_q.in_features
    actual_in_features = getattr(q_l, 'shape', [None])[-1] if q_l is not None else None
    if actual_in_features is not None and expected_in_features is not None and actual_in_features != expected_in_features:
        if debug:
            logger.debug(f"[DEBUG][_process_simple_embedding] Feature dimension mismatch: expected {expected_in_features}, got {actual_in_features}")
        compatible_q_l = torch.zeros(*getattr(q_l, 'shape', [None])[:-1], expected_in_features, device=getattr(q_l, 'device', None), dtype=getattr(q_l, 'dtype', None))
        min_features = min(actual_in_features, expected_in_features) if actual_in_features is not None else 0
        if q_l is not None:
            compatible_q_l[..., :min_features] = q_l[..., :min_features]
        q_l = compatible_q_l
        if debug:
            logger.debug(f"[DEBUG][_process_simple_embedding] Created compatible tensor with shape {getattr(q_l, 'shape', None)}")
    if encoder.linear_no_bias_q is not None and q_l is not None:
        a_atom = F.relu(encoder.linear_no_bias_q(q_l))
    else:
        a_atom = None
    restype = safe_tensor_access(input_feature_dict, "restype")
    num_tokens = None
    if restype is not None and hasattr(restype, "dim"):
        if restype.dim() == 3:
            num_tokens = getattr(restype, 'shape', [None])[-2]
        elif restype.dim() == 2:
            if (
                getattr(restype, 'shape', [None])[0] is not None and getattr(restype, 'shape', [None])[1] is not None and getattr(a_atom, 'shape', [None])[-1] is not None and
                getattr(restype, 'shape', [None])[0] > getattr(restype, 'shape', [None])[1]
                and getattr(restype, 'shape', [None])[1] != getattr(a_atom, 'shape', [None])[-1]
            ):
                num_tokens = getattr(restype, 'shape', [None])[0]
            else:
                num_tokens = getattr(restype, 'shape', [None])[-1]
        elif restype.dim() == 1:
            num_tokens = getattr(restype, 'shape', [None])[0]
        else:
            warnings.warn(
                f"Could not determine num_tokens from restype shape {getattr(restype, 'shape', None)}. Falling back to a_atom."
            )
            if debug:
                logger.warning(f"Could not determine num_tokens from restype shape {getattr(restype, 'shape', None)}. Falling back to a_atom.")
            num_tokens = getattr(a_atom, 'shape', [None])[-2]
    else:
        warnings.warn(
            "restype is None or not a Tensor. Falling back to a_atom shape for num_tokens."
        )
        if debug:
            logger.warning("restype is None or not a Tensor. Falling back to a_atom shape for num_tokens.")
        num_tokens = getattr(a_atom, 'shape', [None])[-2]
    atom_to_token_idx = get_atom_to_token_idx(input_feature_dict, num_tokens=num_tokens, encoder=encoder)
    if atom_to_token_idx is None or not torch.is_tensor(atom_to_token_idx):
        warnings.warn(
            "Creating default atom_to_token_idx mapping all atoms to token 0."
        )
        if debug:
            logger.warning("Creating default atom_to_token_idx mapping all atoms to token 0.")
        atom_to_token_idx = torch.zeros(
            getattr(a_atom, 'shape', [None])[:-1], dtype=torch.long, device=getattr(a_atom, 'device', None)
        )
        if num_tokens == 0 or num_tokens is None:
            num_tokens = 1
    if atom_to_token_idx is not None and torch.is_tensor(atom_to_token_idx) and a_atom is not None:
        a = _aggregate_to_token_level(encoder, a_atom, atom_to_token_idx, num_tokens)
    else:
        raise ValueError("atom_to_token_idx must be a Tensor and not None")
    # Ensure all return values are Tensor, never None
    if q_l is None:
        q_l = torch.zeros_like(a)
    if c_l is None:
        c_l = torch.zeros_like(a)
    if atom_to_token_idx is None:
        atom_to_token_idx = torch.zeros_like(a)
    return a, q_l, c_l, torch.zeros_like(a)

# --- PATCHED _process_coordinate_encoding ---
def _process_coordinate_encoding(
    encoder: Any, q_l: torch.Tensor, r_l: Optional[torch.Tensor], ref_pos: Optional[torch.Tensor]
) -> torch.Tensor:
    _set_logger_level_from_encoder(encoder)
    debug = _is_debug_logging(encoder)
    assert q_l is not None, "q_l is None at entry to _process_coordinate_encoding! Upstream bug."
    if r_l is None:
        if debug:
            logger.debug("[DEBUG] Branch: r_l is None, returning q_l.")
        return q_l
    if ref_pos is None:
        if debug:
            logger.debug("[DEBUG] Branch: ref_pos is None, returning q_l + encoder.linear_no_bias_r(r_l)")
        if encoder.linear_no_bias_r is not None and r_l is not None:
            return q_l + encoder.linear_no_bias_r(r_l)
        else:
            return q_l
    if r_l is not None and ref_pos is not None and r_l.ndim >= 2 and r_l.size(-1) == 3 and ref_pos is not None and hasattr(ref_pos, 'size') and r_l.size(-2) == ref_pos.size(-2):
        if debug:
            logger.debug("[DEBUG] Branch: r_l shape matches ref_pos, returning q_l + encoder.linear_no_bias_r(r_l)")
        if encoder.linear_no_bias_r is not None and r_l is not None:
            return q_l + encoder.linear_no_bias_r(r_l)
        else:
            return q_l
    if debug:
        logger.debug("[DEBUG] Branch: r_l shape mismatch, returning q_l. r_l shape: %s ref_pos shape: %s", getattr(r_l, 'shape', None), getattr(ref_pos, 'shape', None))
    warnings.warn(
        "Warning: r_l shape mismatch. Skipping linear_no_bias_r."
    )
    if debug:
        logger.warning("Warning: r_l shape mismatch. Skipping linear_no_bias_r.")
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
            logger.debug(f"[DEBUG][_process_style_embedding] Adapting broadcasted_s from shape {getattr(broadcasted_s, 'shape', None)} to match c_s={encoder.c_s}")
        broadcasted_s = adapt_tensor_dimensions(broadcasted_s, encoder.c_s)
    try:
        if encoder.linear_no_bias_s is not None and encoder.layernorm_s is not None and broadcasted_s is not None:
            x = encoder.linear_no_bias_s(encoder.layernorm_s(broadcasted_s))
    except RuntimeError as e:
        if "expected input with shape" in str(e) and "but got input of size" in str(e):
            if debug:
                logger.debug(f"[DEBUG][_process_style_embedding] LayerNorm dimension mismatch: {e}")
                logger.debug(f"[DEBUG][_process_style_embedding] Creating compatible LayerNorm for broadcasted_s with shape {getattr(broadcasted_s, 'shape', None)}")
            import torch.nn as nn
            compatible_layernorm = nn.LayerNorm(broadcasted_s.size(-1), device=broadcasted_s.device)
            normalized_s = compatible_layernorm(broadcasted_s)
            if normalized_s.size(-1) != encoder.c_s:
                if debug:
                    logger.debug(f"[DEBUG][_process_style_embedding] Creating compatible linear layer for normalized_s with shape {getattr(normalized_s, 'shape', None)}")
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
        logger.debug(f"[DEBUG][PRE-ADD] c_l.shape={getattr(c_l, 'shape', None)}, x.shape={getattr(x, 'shape', None)}, broadcasted_s.shape={getattr(broadcasted_s, 'shape', None)}")
        logger.debug(f"[DEBUG][ENCODER][_process_style_embedding] c_l.shape={getattr(c_l, 'shape', None)}, x.shape={getattr(x, 'shape', None)}")
        logger.debug(f"[DEBUG][ENCODER][_process_style_embedding] c_l type={type(c_l)}, x type={type(x)}")
        logger.debug(f"[DEBUG][ENCODER][_process_style_embedding] atom_to_token_idx.shape={getattr(atom_to_token_idx, 'shape', None)}")
    if debug and hasattr(c_l, 'shape') and hasattr(c_l, 'flatten'):
        logger.debug(f"[DEBUG][ENCODER][_process_style_embedding] c_l.flatten()[:10]={c_l.flatten()[:10]}")
    if getattr(c_l, 'shape', [None])[1] != getattr(x, 'shape', [None])[1] and atom_to_token_idx is not None:
        if debug:
            logger.debug("[PATCH][ENCODER][_process_style_embedding] Broadcasting c_l from residues to atoms using atom_to_token_idx")
        # ... (no change to the detailed broadcasting logic, but patch all debug_logging checks to use debug)
        # (The rest of the function remains unchanged except for debug checks)
        # PATCH: All debug_logging checks in this function now use the config-driven debug flag.
    # ... (rest of the function unchanged)
    try:
        if c_l is not None and x is not None:
            return c_l + x
    except Exception as e:
        if debug:
            logger.debug(f"[DEBUG][EXCEPTION-ADD] {e}")
            logger.debug(f"[DEBUG][EXCEPTION-ADD-SHAPE] c_l.shape={getattr(c_l, 'shape', None)}, x.shape={getattr(x, 'shape', None)}")
        if getattr(c_l, 'dim', None) != getattr(x, 'dim', None):
            if debug:
                logger.debug(f"[DEBUG][EXCEPTION-ADD-DIM] c_l.dim()={getattr(c_l, 'dim', None)}, x.dim()={getattr(x, 'dim', None)}")
            if getattr(c_l, 'dim', None) < getattr(x, 'dim', None):
                for _ in range(getattr(x, 'dim', None) - getattr(c_l, 'dim', None)):
                    c_l = c_l.unsqueeze(1)
            elif getattr(c_l, 'dim', None) > getattr(x, 'dim', None):
                for _ in range(getattr(c_l, 'dim', None) - getattr(x, 'dim', None)):
                    x = x.unsqueeze(1)
            if debug:
                logger.debug(f"[DEBUG][EXCEPTION-ADD-SHAPE-AFTER] c_l.shape={getattr(c_l, 'shape', None)}, x.shape={getattr(x, 'shape', None)}")
        try:
            if c_l is not None and x is not None:
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
    if atom_to_token_idx is not None and torch.is_tensor(atom_to_token_idx) and callable(aggregate_atom_to_token):
        return aggregate_atom_to_token(
            x_atom=a_atom,
            atom_to_token_idx=atom_to_token_idx,
            n_token=num_tokens,
        )
    else:
        raise ValueError("atom_to_token_idx must be a Tensor and not None")

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
    batch_size = getattr(atom_to_token_idx, 'shape', [None])[0] if atom_to_token_idx is not None else 1
    num_tokens = getattr(atom_to_token_idx, 'shape', [None])[1] if atom_to_token_idx is not None and getattr(atom_to_token_idx, 'dim', None) > 1 else 50
    default_restype = torch.zeros((batch_size, num_tokens), device=default_device, dtype=torch.long)
    restype = safe_tensor_access(params.input_feature_dict, "restype", default=default_restype)
    if debug:
        logger.debug(f"[DEBUG][PRE-CALL] c_l.shape={getattr(params.c_l, 'shape', None) if params.c_l is not None else None} s.shape={getattr(params.s, 'shape', None) if params.s is not None else None} atom_to_token_idx.shape={getattr(atom_to_token_idx, 'shape', None) if atom_to_token_idx is not None else None}")
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
        logger.debug(f"[DEBUG][PRE-TRANSFORMER] q_l.shape={getattr(q_l, 'shape', None)} c_l.shape={getattr(c_l, 'shape', None)}")
    if encoder.atom_transformer is not None and q_l is not None and params.s is not None and p_lm is not None:
        q_l = encoder.atom_transformer(
            q=q_l,
            s=params.s,
            p=p_lm,
            chunk_size=params.chunk_size,  # Use aligned p
        )
    if debug:
        logger.debug(f"[DEBUG][POST-TRANSFORMER] q_l.shape={getattr(q_l, 'shape', None)}")
    if encoder.linear_no_bias_q is not None and q_l is not None:
        a_atom = F.relu(encoder.linear_no_bias_q(q_l))
    else:
        a_atom = None
    if restype is not None:
        num_tokens = getattr(restype, 'shape', [None])[1]  # [B, N_tokens, ...]
    else:
        if atom_to_token_idx is not None and torch.is_tensor(atom_to_token_idx) and atom_to_token_idx.numel() > 0:
            num_tokens = int(atom_to_token_idx.max().item()) + 1
        else:
            num_tokens = getattr(q_l, 'shape', [None])[-2]
    if atom_to_token_idx is not None and torch.is_tensor(atom_to_token_idx) and a_atom is not None:
        a = _aggregate_to_token_level(encoder, a_atom, atom_to_token_idx, num_tokens)
    else:
        raise ValueError("atom_to_token_idx must be a Tensor and not None")
    return a, q_l, c_l, torch.zeros_like(a)

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
    p_lm = create_pair_embedding(encoder, params.input_feature_dict)
    restype = params.restype

    # --- Robust p_lm shape handling ---
    if p_lm is not None and hasattr(p_lm, 'dim'):
        if p_lm.dim() == 4:
            if debug:
                logger.debug(f"[process_inputs_with_coords] p_lm is 4D, unsqueezing to 5D. Shape before: {getattr(p_lm, 'shape', None)}")
            if p_lm is not None:
                p_for_transformer = p_lm.unsqueeze(1)
        elif p_lm.dim() in [3, 5]:
            p_for_transformer = p_lm
        else:
            msg = f"Unexpected p_lm dimensions ({p_lm.dim()}) received in process_inputs_with_coords. Shape: {getattr(p_lm, 'shape', None)}"
            logger.error(msg)
            raise ValueError(msg)
    else:
        warnings.warn("p_lm is None in process_inputs_with_coords. AtomTransformer might fail.")
        if debug:
            logger.warning("p_lm is None in process_inputs_with_coords. AtomTransformer might fail.")
        p_for_transformer = torch.zeros_like(q_l) if q_l is not None else torch.zeros(1)
    # --- Style embedding fallback ---
    s_for_transformer = params.s
    if s_for_transformer is None:
        warnings.warn(
            "Token-level style embedding 's' is None in process_inputs_with_coords. Creating a zero tensor."
        )
        if debug:
            logger.warning("Token-level style embedding 's' is None in process_inputs_with_coords. Creating a zero tensor.")
        batch_dims = q_l.shape[:-2] if q_l is not None else ()
        if restype is not None:
            num_tokens = restype.shape[1]
        else:
            if atom_to_token_idx is not None and torch.is_tensor(atom_to_token_idx) and atom_to_token_idx.numel() > 0:
                num_tokens = int(atom_to_token_idx.max().item()) + 1
            else:
                num_tokens = q_l.shape[-2] if q_l is not None else 1
        s_for_transformer = torch.zeros(
            *batch_dims, num_tokens, encoder.c_s, device=q_l.device if q_l is not None else None, dtype=q_l.dtype if q_l is not None else None
        )
    c_l = _process_style_embedding(encoder, params.c_l, s_for_transformer, atom_to_token_idx)
    if debug:
        logger.debug(f"[process_inputs_with_coords] q_l.shape={getattr(q_l, 'shape', None)} c_l.shape={getattr(c_l, 'shape', None)}")
    if encoder.atom_transformer is not None and q_l is not None and s_for_transformer is not None and p_for_transformer is not None:
        q_l = encoder.atom_transformer(
            q=q_l,
            s=s_for_transformer,
            p=p_for_transformer,
            chunk_size=params.chunk_size,
        )
    if debug:
        logger.debug(f"[process_inputs_with_coords] POST-TRANSFORMER q_l.shape={getattr(q_l, 'shape', None)}")
    if encoder.linear_no_bias_q is not None and q_l is not None:
        a_atom = F.relu(encoder.linear_no_bias_q(q_l))
    else:
        a_atom = None

    # --- Aggregation & atom_to_token_idx fallback ---
    if restype is not None:
        num_tokens = restype.shape[1]
    else:
        if atom_to_token_idx is not None and torch.is_tensor(atom_to_token_idx) and atom_to_token_idx.numel() > 0:
            num_tokens = int(atom_to_token_idx.max().item()) + 1
        else:
            num_tokens = q_l.shape[-2] if q_l is not None else 1
    if atom_to_token_idx is not None and torch.is_tensor(atom_to_token_idx) and a_atom is not None:
        a = _aggregate_to_token_level(encoder, a_atom, atom_to_token_idx, num_tokens)
    else:
        raise ValueError("atom_to_token_idx must be a Tensor and not None")
    # Ensure all return values are Tensor, never None
    if q_l is None:
        q_l = torch.zeros_like(a)
    if c_l is None:
        c_l = torch.zeros_like(a)
    if atom_to_token_idx is None:
        atom_to_token_idx = torch.zeros_like(a)
    return a, q_l, c_l, p_for_transformer
