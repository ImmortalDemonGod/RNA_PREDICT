# rna_predict/dataset/collate.py
import torch
import logging

logger = logging.getLogger("rna_predict.dataset.collate")

def rna_collate_fn(batch, debug_logging=False):
    """Collate function for RNA dataset batches, robust to empty and single-item batches.
    Args:
        batch (list): List of samples (dicts) from RNADataset.
        debug_logging (bool): If True, emit debug logs.
    Returns:
        dict: Batched tensors and lists.
    """
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
                    logger.debug(f"[collate] Key: {k}, Shape: {v.shape}, Dtype: {v.dtype}, requires_grad: {getattr(v, 'requires_grad', 'N/A')}")
                else:
                    logger.debug(f"[collate] Key: {k}, Type: {type(v)}")
        return out
    # Multi-item batch: stack tensors, listify others
    out = {}
    for k in batch[0]:
        if isinstance(batch[0][k], torch.Tensor):
            stacked = torch.stack([b[k] for b in batch])
            out[k] = stacked
        elif k in ["atom_names", "residue_indices"]:
            # For target atom metadata, batch as list of lists
            out[k] = [b[k] for b in batch]
        else:
            out[k] = [b[k] for b in batch]
    if debug_logging:
        logger.debug("[collate] Multi-item batch keys: %s", list(out.keys()))
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                logger.debug(f"[collate] Key: {k}, Shape: {v.shape}, Dtype: {v.dtype}, requires_grad: {getattr(v, 'requires_grad', 'N/A')}")
            else:
                logger.debug(f"[collate] Key: {k}, Type: {type(v)}")
    return out
