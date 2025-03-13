
from datasets import load_dataset

def stream_bprna_dataset(split="train"):
    """
    Stream the bprna-spot dataset from the HF Hub.
    Returns an IterableDataset object for the specified split.
    """
    ds_iter = load_dataset(
        "multimolecule/bprna-spot",
        split=split,
        streaming=True
    )
    return ds_iter
