"""
Tests for rna_predict console-script entry point (finding s2c0l0-003).

Bug catalog: tests/test_entrypoint.bug-catalog.md
Canonical intent: https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/
    1f6481e4d8d7c673f44115c0a5bbaa1703ebe562/audit/02-static-audit.md#L48

Layer contract:
  - test_main_module_importable, test_main_callable: UNIT layer — test Python
    module resolution for the import declared in pyproject.toml:61.
  - test_python_m_rna_predict_help_exits_zero: INTEGRATION layer — tests the
    end-to-end invocation chain (interpreter -> package entry -> __main__.py).
"""

import importlib
import subprocess
import sys


def test_main_module_importable__guards_missing_dunder_main():
    """
    rna_predict.__main__ must be importable.

    Guards Bug 1 (catalog): missing rna_predict/__main__.py causes
    ModuleNotFoundError on every invocation of the 'rna_predict' console
    script or 'python -m rna_predict'.  pyproject.toml:61 declares
    'rna_predict.__main__:main' as the entry target; that module must exist.
    """
    mod = importlib.import_module("rna_predict.__main__")
    assert mod is not None


def test_main_callable__guards_missing_entry_point_symbol():
    """
    rna_predict.__main__.main must be a callable.

    Guards Bug 2 (catalog): if __main__.py existed but defined no top-level
    'main' symbol, the setuptools-generated script wrapper would raise
    ImportError/AttributeError on invocation.  The entry point string
    'rna_predict.__main__:main' (pyproject.toml:61) requires an explicit
    module-level callable named 'main'.
    """
    mod = importlib.import_module("rna_predict.__main__")
    main_fn = getattr(mod, "main", None)
    assert callable(main_fn), (
        "rna_predict.__main__.main must exist and be callable "
        "(required by pyproject.toml:61 console-script entry point)"
    )


def test_python_m_rna_predict_help_exits_zero__guards_cmd_crash():
    """
    `python -m rna_predict --help` must exit 0.

    Guards Bug 3 (catalog): Containerfile:5 CMD ["rna_predict"] and the
    'python -m rna_predict' invocation both require rna_predict/__main__.py.
    Without it, Python raises: "No module named rna_predict.__main__;
    'rna_predict' is a package and cannot be directly executed" (exit 1).
    This is the integration-layer assertion for the same invariant tested at
    unit layer above.
    """
    result = subprocess.run(
        [sys.executable, "-m", "rna_predict", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"python -m rna_predict --help exited {result.returncode}.\n"
        f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
    )
