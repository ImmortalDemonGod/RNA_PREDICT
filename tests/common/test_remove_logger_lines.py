import unittest
import re


def remove_logger_lines(text: str) -> str:
    """
    Remove extraneous logging lines from the generated test content.
    This function filters out:
      - Lines starting with a bracketed or non-bracketed timestamp (e.g. "[2025-3-27 14:55:48,330] ..." or "2025-03-27 14:55:48,330 - ...").
      - Lines containing known noisy substrings such as 'real_accelerator.py:' or 'Setting ds_accelerator to'.
    """
    lines = text.splitlines()
    filtered = []
    timestamp_pattern = re.compile(r"^\[?\d{4}-\d{1,2}-\d{1,2}")
    for line in lines:
        # Skip lines matching a leading timestamp
        if timestamp_pattern.match(line):
            continue
        # Skip lines containing known noisy substrings
        if "real_accelerator.py:" in line or "Setting ds_accelerator to" in line:
            continue
        filtered.append(line)
    return "\n".join(filtered).strip()


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
