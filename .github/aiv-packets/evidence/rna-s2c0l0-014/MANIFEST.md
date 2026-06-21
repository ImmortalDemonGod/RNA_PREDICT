# Evidence Manifest — Finding s2c0l0-014

**Finding:** `test: lint` couples test execution to lint, aborting pytest on mypy/ruff errors; `--unsafe-fixes` mutates source as a side effect.
**Baseline SHA (cited):** `1f6481e4d8d7c673f44115c0a5bbaa1703ebe562` (origin/main)
**Fix HEAD SHA:** `e965d1e1c304c53a3cc9baf20b9da40cb6ffc4c7` (fix/rna-s2c0l0-014)
**Independent assessor verdict:** PASS (agent `a0d059933f325fd22`)

## Artifacts

| File | sha256 | Claim proved | Cited baseline ref | AIV class |
|---|---|---|---|---|
| `baseline_red.txt` | `e22f67f9e655c1dff12a4ba1b3caa2cc622d56522742d98a1d99387db3b914f9` | B1+B2+B3 defects EXIST at baseline (3 FAILED, exit code 1) | `1f6481e4` | A, D |
| `head_green.txt` | `893d9f71dae1ba41f3f1f9e35fa184ac390d4fd2696bc1f37f51e9f4e597c4d2` | B1+B2+B3 defects ABSENT at HEAD (3 PASSED, exit code 0) | `1f6481e4` → `e965d1e1` | A, D |
| `makefile_diff.txt` | `bf03b650f3ff5515218b22813a40f48a0b2133a0021670a23e215edc1844f365` | Both structural changes (B1 prereq removal, B2 --unsafe-fixes removal) | `1f6481e4` → `e965d1e1` | B, D |
| `makefile_baseline.txt` | `df4f236819c99c8e54a7332b64602e575246ee14dc3b8743d297d1ef9286fb56` | Baseline state of Makefile (defect present) | `1f6481e4` | D |
| `makefile_head.txt` | `bf5325dcbd1c56277f3db3a1d0bfe0a379a5085ac1cc0409b9899609c715e2a2` | Fixed state of Makefile (defect absent) | `e965d1e1` | D |
| `class_c_negative_search.txt` | `f689ca565c613531c8879067ccd9975f55193a23d93fe307ea4ef321a171cea9` | Neither `test: lint` pattern nor `--unsafe-fixes` appears in HEAD Makefile | `e965d1e1` | C |

## Notes

- Baseline worktree `/tmp/rna-s2c0l0-014_base` (detached at `1f6481e4`) was used to produce the RED run.
- The test file `tests/test_makefile_contract.py` was added by this fix branch and copied to the baseline worktree for the before-run (the task specifies using the NEW RED tests against baseline).
- Independent assessor flagged: tests verify Makefile *structure* (regex parse), not runtime `make` invocation — an acceptable tradeoff for a pure-config change with no infra boundary.
- Independent assessor flagged: tests 1 and 3 share the same root assertion (both check `lint` not in test prereqs). This is intentional — B3 is a superset of B1 (CI-specific framing). Both are counted.
