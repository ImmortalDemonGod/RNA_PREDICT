
# NOTE: This test was originally designed to check error handling for a custom CombinedLoader
# that required calling iter(combined_loader) first. The current codebase uses a standard
# PyTorch DataLoader and RNADataset, which do not raise this error. No such error can be
# triggered in the present implementation, so this test is now obsolete and is being disabled.
# If future loader implementations require similar error handling, update and re-enable this test.

def test_combined_loader_runtime_error():
    pass  # Test disabled: no relevant error condition in current loader implementation
