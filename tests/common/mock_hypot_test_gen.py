"""
Mock implementation of hypot_test_gen.py for testing purposes.
"""

import re


def fix_leading_zeros(test_code: str) -> str:
    """
    Replace decimal integers with leading zeros (except a standalone "0") with their corrected form.
    For example, "007" becomes "7" and "-0123" becomes "-123".
    """
    # Use a regex with negative lookbehind and lookahead to match numbers that start with one or more zeros.
    # The pattern (?<!\d)(-?)0+(\d+)(?!\d) ensures that a minus sign is captured if present,
    # and that only isolated numbers are matched.
    fixed_code = re.sub(
        r"(?<!\d)(-?)0+(\d+)(?!\d)",
        lambda m: m.group(1) + str(int(m.group(2))),
        test_code,
    )
    return fixed_code
