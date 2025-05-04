# rna_predict/dataset/collate.py
import torch
import logging

logger = logging.getLogger("rna_predict.dataset.collate")


def rna_collate_fn(batch, cfg=None, debug_logging=None):
    """Collate function for RNA dataset batches, robust to empty and single-item batches.
    Args:
        batch (list): List of samples (dicts) from RNADataset.
        cfg (optional): Hydra config object. If provided, will attempt to source debug_logging from it.
        debug_logging (bool, optional): Explicit debug flag. If None, will use cfg if available, else False.
    Returns:
        dict: Batched tensors and lists.
    """
    # Respect Hydra config hierarchy for debug_logging
    resolved_debug = False
    if debug_logging is not None:
        resolved_debug = debug_logging
    elif cfg is not None:
        # Try to find debug_logging in config, prefer model.stageD.debug_logging > data.debug_logging > False
        try:
            if hasattr(cfg, 'model') and hasattr(cfg.model, 'stageD') and hasattr(cfg.model.stageD, 'debug_logging'):
                resolved_debug = cfg.model.stageD.debug_logging
            elif hasattr(cfg, 'data') and hasattr(cfg.data, 'debug_logging'):
                resolved_debug = cfg.data.debug_logging
        except Exception:
            pass
    debug_logging = resolved_debug

    if debug_logging:
        logger.debug("[collate] Batch size: %d", len(batch))
    # Instrument: Print device info for all tensors in the batch
    for i, sample in enumerate(batch):
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                logger.debug(f"[collate][DEBUG-DEVICE] Batch item {i} key '{k}': device={v.device}, shape={v.shape}, dtype={v.dtype}")
    if len(batch) == 0:
        raise ValueError("Empty batch passed to rna_collate_fn.")
    if len(batch) == 1:
        # Single-item batch: add batch dimension to tensors
        out = {}
        for k, v in batch[0].items():
            if isinstance(v, torch.Tensor):
                out[k] = v.unsqueeze(0)
            elif k in ["atom_names", "residue_indices"]:
                # For target atom metadata, batch as list of lists
                out[k] = [v]
            else:
                out[k] = [v]
        if debug_logging:
            logger.debug("[collate] Single-item batch keys: %s", list(out.keys()))
            for k, v in out.items():
                if isinstance(v, torch.Tensor):
                    logger.debug(f"[collate] Key: {k}, Shape: {v.shape}, Dtype: {v.dtype}, Device: {v.device}, requires_grad: {getattr(v, 'requires_grad', 'N/A')}")
                else:
                    logger.debug(f"[collate] Key: {k}, Type: {type(v)}")
        # Instrument: Print all tensor devices in single-item batch
        print("[DEBUG][collate_fn][single-item] Batch tensor device summary:")
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                print(f"  Key: {k}, shape: {v.shape}, dtype: {v.dtype}, device: {v.device}")
            else:
                print(f"  Key: {k}, type: {type(v)}")
        return out
    # Multi-item batch: stack tensors, listify others
    out = {}
    for k in batch[0].keys():
        vs = [d[k] for d in batch]
        # Instrument: Print devices of all tensors to be stacked
        if isinstance(vs[0], torch.Tensor):
            print(f"[DEBUG][collate_fn][multi-item] Key: {k}")
            for i, v in enumerate(vs):
                print(f"  Sample {i}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
        if isinstance(vs[0], torch.Tensor):
            try:
                stacked = torch.stack(vs, dim=0)
                print(f"  [DEBUG][collate_fn][multi-item] Stacked tensor: shape={stacked.shape}, dtype={stacked.dtype}, device={stacked.device}")
                out[k] = stacked
            except Exception as e:
                print(f"[ERROR][collate_fn][multi-item] Failed to stack key '{k}': {e}")
                raise
        elif isinstance(vs[0], list):
            out[k] = [item for sublist in vs for item in (sublist if isinstance(sublist, list) else [sublist])]
        else:
            out[k] = vs
    if debug_logging:
        logger.debug("[collate] Multi-item batch keys: %s", list(out.keys()))
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                logger.debug(f"[collate] Key: {k}, Shape: {v.shape}, Dtype: {v.dtype}, requires_grad: {getattr(v, 'requires_grad', 'N/A')}")
            else:
                logger.debug(f"[collate] Key: {k}, Type: {type(v)}")
    return out
