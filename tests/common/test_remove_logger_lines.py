import unittest

from rna_predict.scripts.hypot_test_gen import remove_logger_lines


class TestRemoveLoggerLines(unittest.TestCase):
    def test_remove_logger_lines(self):
        input_text = """[2025-03-27 14:55:48,330] [INFO] [real_accelerator.py:222:get_accelerator] Setting ds_accelerator to mps (auto detect)
2025-03-27 14:55:48,330 - DEBUG - Some debug message that should be removed
# This test code was written by the `hypothesis.extra.ghostwriter` module
import transformer
Normal code line
"""
        expected_output = """# This test code was written by the `hypothesis.extra.ghostwriter` module
import transformer
Normal code line"""
        result = remove_logger_lines(input_text)
        self.assertEqual(result, expected_output.strip())


if __name__ == "__main__":
    unittest.main()
