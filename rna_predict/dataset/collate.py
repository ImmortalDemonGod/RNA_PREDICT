# rna_predict/dataset/collate.py
import torch

def rna_collate_fn(batch):
    """Collate function for RNA dataset batches, robust to empty and single-item batches.
    Args:
        batch (list): List of samples (dicts) from RNADataset.
    Returns:
        dict: Batched tensors and lists.
    """
    print("[DEBUG][collate] Batch size:", len(batch))
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
        print("[DEBUG][collate] Single-item batch keys:", list(out.keys()))
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                print(f"[DEBUG][collate] Key: {k}, Shape: {v.shape}, Dtype: {v.dtype}, requires_grad: {getattr(v, 'requires_grad', 'N/A')}")
            else:
                print(f"[DEBUG][collate] Key: {k}, Type: {type(v)}")
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
    print("[DEBUG][collate] Multi-item batch keys:", list(out.keys()))
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"[DEBUG][collate] Key: {k}, Shape: {v.shape}, Dtype: {v.dtype}, requires_grad: {getattr(v, 'requires_grad', 'N/A')}")
        else:
            print(f"[DEBUG][collate] Key: {k}, Type: {type(v)}")
    return out
