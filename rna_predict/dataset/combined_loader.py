"""
Combined loader module for RNA_PREDICT.

This module provides a CombinedLoader class that wraps a DataLoader to provide additional functionality
and error handling for the RNA_PREDICT pipeline.
"""

from torch.utils.data import DataLoader

class CombinedLoader:
    """
    A wrapper around DataLoader that provides additional functionality and error handling.
    
    This class is used to ensure proper initialization and iteration over the data loader,
    with appropriate error messages when used incorrectly.
    """
    
    def __init__(self, dataloader: DataLoader):
        """
        Initialize the CombinedLoader.
        
        Args:
            dataloader: The DataLoader instance to wrap.
        """
        self.dataloader = dataloader
        self._iterator = None
        
    def __iter__(self):
        """
        Get an iterator over the data.
        
        Returns:
            An iterator over the wrapped DataLoader.
        """
        self._iterator = iter(self.dataloader)
        return self
        
    def __next__(self):
        """
        Get the next batch of data.
        
        Returns:
            The next batch from the DataLoader.
            
        Raises:
            RuntimeError: If iter() hasn't been called on the loader first.
        """
        if self._iterator is None:
            raise RuntimeError("Please call `iter(combined_loader)` first")
        return next(self._iterator) 