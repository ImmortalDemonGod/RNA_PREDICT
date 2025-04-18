"""
Entry point for running all pairformer-related tests.

This file has been split into multiple files for better organization:
    • test_utils.py - Shared test utilities and helper functions
    • test_pairformer_blocks.py - Tests for PairformerBlock and PairformerStack
    • test_msa_components.py - Tests for MSA-related components
    • test_template_and_sampling.py - Tests for TemplateEmbedder and MSA sampling

How to Run
----------
Save this file (e.g. as test_pairformer.py) and run:

    python -m unittest test_pairformer.py

Or integrate it into your continuous integration / test suite as needed.

Dependencies
------------
    • Python 3.7+ recommended
    • PyTorch >= 1.7 (for tensor creation)
    • Hypothesis >= 6.0 (for property-based testing)
    • (Optional) CUDA environment if testing GPU caching code paths
"""

import unittest

# Import all test cases
from tests.stageB.pairwise.test_pairformer_blocks import TestPairformerBlock, TestPairformerStack
from tests.stageB.pairwise.test_msa_components import (
    TestMSAPairWeightedAveraging,
    TestMSAStack,
    TestMSABlock,
    TestMSAModule,
)
from tests.stageB.pairwise.test_template_and_sampling import TestTemplateEmbedder

# Create test suite
def suite():
    suite = unittest.TestSuite()
    
    # Add PairformerBlock and Stack tests
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPairformerBlock))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPairformerStack))
    
    # Add MSA component tests
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestMSAPairWeightedAveraging))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestMSAStack))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestMSABlock))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestMSAModule))
    
    # Add TemplateEmbedder tests
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestTemplateEmbedder))
    
    return suite

if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())