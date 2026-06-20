# AIV Evidence File (v1.0)

**File:** `Makefile`
**Commit:** `a63f4e6`
**Generated:** 2026-06-20T23:57:35Z
**Protocol:** AIV v2.0 + Addendum 2.7 (Zero-Touch Mandate)

---

## Classification (required)

```yaml
classification:
  risk_tier: R0
  sod_mode: S0
  critical_surfaces: []
  blast_radius: "Makefile"
  classification_rationale: "R0 — single-line Makefile edit removing one word from a prerequisite list; no Python source changes; no logic added; behavioral proof provided by make -n test dry-run (Class A); pre-existing ruff/mypy failures documented"
  classified_by: "Claude"
  classified_at: "2026-06-20T23:57:35Z"
```

## Claim(s)

1. Makefile:47 test: target has no prerequisites — make -n test resolves to pytest and coverage invocations only, with zero ruff/mypy/lint/--unsafe-fixes lines
2. lint target at Makefile:33-35 remains intact with ruff and mypy recipes unchanged
3. --unsafe-fixes flag appears only in lint recipe (Makefile:34) and is not reachable from the test target
4. No existing tests were modified or deleted during this change.

---

## Evidence

### Class E (Intent Alignment)

- **Link:** [https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/1f6481e4d8d7c673f44115c0a5bbaa1703ebe562/audit/02-static-audit.md#L474](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/1f6481e4d8d7c673f44115c0a5bbaa1703ebe562/audit/02-static-audit.md#L474)
- **Requirements Verified:** audit/02-static-audit.md L474 records that test:lint coupling causes make test to abort before pytest runs when mypy/ruff exit non-zero; the fix requires removing lint from the test prerequisite list so make test exit code reflects only test outcomes

### Class B (Referential Evidence)

**Scope Inventory** (SHA: [`a63f4e6`](https://github.com/ImmortalDemonGod/RNA_PREDICT/tree/a63f4e650cb1a317780e99065d0ba1e7806d6ecc))

- [`Makefile#L47`](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/a63f4e650cb1a317780e99065d0ba1e7806d6ecc/Makefile#L47)

### Class A (Execution Evidence)

- Local checks skipped (--skip-checks).
- **Skip reason:** All ruff/mypy failures are pre-existing (8 ruff errors confirmed on base branch before this change; mypy binary absent at .venv/bin/mypy). This change modifies only Makefile:47 — zero Python source lines touched — so the pre-existing tool failures are unrelated to this diff. Class-A behavioral proof is provided by make -n test dry-run (output: pytest+coverage only, zero ruff/mypy/lint/--unsafe-fixes lines), which catches parse/tab/PHONY breakage that grep cannot detect.


---

## Verification Methodology

**R0 (trivial) -- local checks skipped.**
**Reason:** All ruff/mypy failures are pre-existing (8 ruff errors confirmed on base branch before this change; mypy binary absent at .venv/bin/mypy). This change modifies only Makefile:47 — zero Python source lines touched — so the pre-existing tool failures are unrelated to this diff. Class-A behavioral proof is provided by make -n test dry-run (output: pytest+coverage only, zero ruff/mypy/lint/--unsafe-fixes lines), which catches parse/tab/PHONY breakage that grep cannot detect.
Only git diff scope inventory was collected. No execution evidence.

---

## Summary

Remove lint prerequisite from test: target so make test exit code reflects only pytest outcomes
