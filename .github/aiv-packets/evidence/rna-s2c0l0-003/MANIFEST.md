# Evidence Manifest — rna-s2c0l0-003

Finding: Containerfile:5 `CMD ["rna_predict"]` crashes with `ModuleNotFoundError: No module named 'rna_predict.__main__'`
Baseline ref: `1f6481e4d8d7c673f44115c0a5bbaa1703ebe562` (origin/main)
HEAD ref: `5ffd5327c6b212ea4233e5caa8b84e4713dac4cc` (fix/rna-s2c0l0-003)

| Artifact | sha256 | Claim proved | Cited baseline ref | AIV class |
|---|---|---|---|---|
| `baseline_red.txt` | `ba572ba01080e32df09e80f4f471f189d58bb6855e483445d96e91af6f75d764` | Defect EXISTS at baseline: `ModuleNotFoundError: No module named 'rna_predict.__main__'`; `python -m rna_predict --help` exits 1 | `1f6481e4` | A (execution/behavioral) + D (differential before) |
| `head_green.txt` | `dedf1ad302f566193b1ca0ea34dd88ec4b1e7fc2c2f2b9baca747466fe201ee8` | Fix PRESENT at HEAD: 3/3 tests pass; `find rna_predict -name __main__.py` returns `rna_predict/__main__.py`; `python -m rna_predict --help` exits 0 and prints Hydra help | `1f6481e4` (diffed against) | A (execution/behavioral) + D (differential after) |

## Claim-to-artifact map

| Claim | Artifact | Verdict |
|---|---|---|
| `ModuleNotFoundError` exists at baseline | `baseline_red.txt` line 38 | PASS — directly shown |
| `find rna_predict -name __main__.py` returns empty at baseline | `baseline_red.txt` bottom section | PASS — `EXIT:0` with no path output |
| `python -m rna_predict --help` exits 1 at baseline | `baseline_red.txt` bottom section | PASS — `EXIT:1` shown |
| `rna_predict/__main__.py` exists at HEAD | `head_green.txt` find section | PASS — path printed, `EXIT:0` |
| `python -m rna_predict --help` exits 0 at HEAD | `head_green.txt` bottom section | PASS — Hydra help printed, `EXIT:0` |
| All 3 RED tests pass at HEAD | `head_green.txt` pytest output | PASS — `3 passed in 12.87s` |
