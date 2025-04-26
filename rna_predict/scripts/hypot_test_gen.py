"""
hypot_test_gen.py
Module for generating hypothesis tests and related utilities.
"""
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


def fix_leading_zeros(s: str) -> str:
    """
    Replace decimal integers with leading zeros (except a standalone "0") with their corrected form.
    For example, "007" becomes "7" and "-0123" becomes "-123".
    """
    def repl(match):
        sign = match.group(1)
        digits = match.group(2)
        return (sign if sign else '') + str(int(digits))
    # Match optional negative sign, then one or more zeros, then digits
    return re.sub(r'(-?)0+(\d+)', repl, s)
