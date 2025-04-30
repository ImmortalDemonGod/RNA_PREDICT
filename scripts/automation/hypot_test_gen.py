"""
hypot_test_gen.py
Stub for fix_leading_zeros used by test_fix_leading_zeros.py.
"""
import re

def fix_leading_zeros(s):
    def repl(match):
        sign = match.group(1)
        digits = match.group(2)
        return (sign if sign else '') + str(int(digits))
    # Match optional negative sign, then one or more zeros, then digits
    # Use a regex with negative lookbehind and lookahead to match numbers that start with one or more zeros.
    # The pattern ensures that only isolated numbers are matched.
    return re.sub(r'(?<!\d)(-?)0+(\d+)(?!\d)', repl, s)

def remove_logger_lines(text):
    import re
    lines = text.splitlines()
    filtered = []
    log_patterns = [
        re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}\] \[[A-Z]+\]"),
        re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - [A-Z]+ -")
    ]
    for line in lines:
        if any(pat.match(line) for pat in log_patterns):
            continue
        filtered.append(line)
    return "\n".join(filtered).strip()
