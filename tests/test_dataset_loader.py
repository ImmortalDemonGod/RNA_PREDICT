import unittest
from rna_predict.dataset.dataset_loader import stream_bprna_dataset

class TestDatasetLoader(unittest.TestCase):
    def test_stream_bprna_dataset(self):
        ds_iter = stream_bprna_dataset("train")
        self.assertIsNotNone(ds_iter, "The returned dataset iterator should not be None.")
        # Check minimal iteration
        # We won't exhaust the dataset; just confirm it's iterable.
        iterator = iter(ds_iter)
        first_item = next(iterator, None)
        self.assertIsNotNone(first_item, "Should be able to retrieve at least one record from the dataset.")

if __name__ == "__main__":
    unittest.main()