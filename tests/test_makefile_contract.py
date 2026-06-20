"""
Structural contract tests for the Makefile.

Each test names the bug it catches (see tests/Makefile.bug-catalog.md).
All tests are EXPECTED TO BE RED until the fix is applied to Makefile:47.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
MAKEFILE = REPO_ROOT / "Makefile"


def _prerequisites(target: str) -> list[str]:
    """Return the declared prerequisite list for a make target (comments stripped)."""
    text = MAKEFILE.read_text()
    match = re.search(rf"^{re.escape(target)}\s*:(.*?)$", text, re.MULTILINE)
    if not match:
        return []
    raw = re.sub(r"\s*#+.*$", "", match.group(1))  # strip ## inline comments
    return raw.split()


def _recipe_lines(target: str) -> list[str]:
    """Return the tab-indented recipe lines for a make target."""
    lines = MAKEFILE.read_text().splitlines()
    capturing = False
    result = []
    for line in lines:
        if re.match(rf"^{re.escape(target)}\s*:", line):
            capturing = True
            continue
        if capturing:
            if line.startswith("\t"):
                result.append(line[1:])  # strip the leading tab
            elif line.strip() == "" or line.startswith("#"):
                continue
            else:
                break
    return result


# ---------------------------------------------------------------------------
# B1 — lint-blocks-tests
# ---------------------------------------------------------------------------

def test_test_target_does_not_depend_on_lint_guards_lint_blocks_test_bug():
    """
    Bug: B1 — lint-blocks-tests (Makefile:47 'test: lint').
    A ruff or mypy non-zero exit aborts make before pytest receives control,
    so zero tests run and zero coverage is produced.
    RED until the 'lint' prerequisite is removed from the 'test' target.
    """
    prereqs = _prerequisites("test")
    assert "lint" not in prereqs, (
        f"'test' target must not list 'lint' as a prerequisite; "
        f"found: {prereqs}. "
        "A lint failure must never prevent pytest from running."
    )


# ---------------------------------------------------------------------------
# B2 — unsafe-fixes-mutate-source
# ---------------------------------------------------------------------------

def test_lint_recipe_does_not_use_unsafe_fixes_guards_source_mutation_side_effect_bug():
    """
    Bug: B2 — unsafe-fixes-mutate-source (Makefile:34 '--unsafe-fixes').
    ruff --unsafe-fixes rewrites source files in-place; because 'test: lint',
    every 'make test' invocation silently mutates the working tree.
    RED until '--unsafe-fixes' is removed from the lint recipe.
    """
    recipe = _recipe_lines("lint")
    for line in recipe:
        assert "--unsafe-fixes" not in line, (
            f"lint recipe must not contain '--unsafe-fixes' (mutates source as a "
            f"side effect of running tests): {line!r}"
        )


# ---------------------------------------------------------------------------
# B1/B3 — CI test job aborts on mypy finding (same root cause as B1)
# ---------------------------------------------------------------------------

def test_test_target_prerequisites_contain_only_test_tooling_guards_ci_silent_abort_bug():
    """
    Bug: B3 — ci-test-aborts-silently-on-mypy (Makefile:47 via main.yml:104).
    CI calls 'make test'; if lint is a prerequisite, a mypy type error silently
    aborts the 'Run tests' CI step before pytest runs — CI reports 'test job
    failed' but the failure is a lint issue, not a broken assertion.
    RED until the 'lint' prerequisite is absent from the 'test' target.
    """
    prereqs = _prerequisites("test")
    disallowed = {"lint", "ruff", "mypy", "flake8", "pylint"}
    overlap = disallowed & set(prereqs)
    assert not overlap, (
        f"'test' target must not list linting tools as prerequisites; "
        f"found disallowed prereqs: {sorted(overlap)}. "
        "Linting must run in a separate CI step so test failures are attributable "
        "to broken assertions, not to annotation regressions."
    )
