from datasets import load_dataset
from datasets.iterable_dataset import IterableDataset


def stream_bprna_dataset(split: str = "train") -> IterableDataset:
    """
    Stream the bprna-spot dataset from the HF Hub.

    Args:
        split (str): The dataset split to stream, defaults to "train".

    Returns:
        IterableDataset: An iterable dataset object for the specified split.
    """
    ds_iter = load_dataset("multimolecule/bprna-spot", split=split, streaming=True)
    return ds_iter