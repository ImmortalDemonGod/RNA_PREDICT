# Evidence Manifest — rna-s2c3l0-020

Finding: s2c3l0-020 (critical) — InputFeatureEmbedder ModuleNotFoundError (rna_predict.models)
Baseline SHA (origin/main): 1f6481e4d8d7c673f44115c0a5bbaa1703ebe562
Head SHA: fa1d25b6fdc311b9a9986094fc3dd7b401ee390c
Fix commit: a8120fef fix(s2c3l0-020): repair broken models.encoder import path in legacy InputFeatureEmbedder
Captured: 2026-06-20

| Artifact | SHA-256 | Claim proved | Cited baseline | AIV class |
|---|---|---|---|---|
| baseline_red.txt | d8269bc7dc3a6f3d72b286a04e54313a6f60525a6da7f2ddb7223ee0edddbf75 | Defect EXISTS on origin/main: import raises ModuleNotFoundError | 1f6481e4 | A, D |
| head_green.txt | 898a76c59efd174dd9a6c88897382ed0b8905ff432faa25d3e33e53c62e6e1b4 | Defect ABSENT at HEAD: import exits 0 | 1f6481e4 (vs HEAD) | A, D |
| baseline_pytest_red.txt | f9909ff4eae67ca1edcfc8cf1752b0a7c330788cf43e78d6846caf5918de14bd | All 4 regression tests FAIL on baseline (same ModuleNotFoundError) | 1f6481e4 | A, D |
| head_pytest_green.txt | 29d57e3d4932ab4dedf12f3eacb43aa6d3e90266181feea728f18975208c7556 | 3/4 tests PASS at HEAD; 1 pre-existing deepspeed failure unrelated to fix | fa1d25b6 | A, D |
| file_diff.txt | ae0eb549b7c7585de8a70042d973b1d4a7a4288ba4458616cac69dff44068bb3 | Before/after diff bound to baseline: 2 import lines corrected | 1f6481e4 vs fa1d25b6 | D |
| negative_check.txt | 0b8d21be74fb266955ceae5c056c2b5876189686c22aa4467f7650cac82bf43d | Zero live imports of rna_predict.models in rna_predict/ Python source at HEAD | fa1d25b6 | C |
