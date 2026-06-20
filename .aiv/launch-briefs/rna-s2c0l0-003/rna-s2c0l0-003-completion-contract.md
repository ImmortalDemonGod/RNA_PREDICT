===== PR-rna-s2c0l0-003 COMPLETION CONTRACT — add missing __main__ entry-point module =====

GOAL: Resolve finding s2c0l0-003 (audit/02-static-audit.md L48). Create rna_predict/__main__.py
defining main() so the console script rna_predict.__main__:main (pyproject.toml:61) resolves
without error. Verified when: uv run rna_predict --help exits 0 and prints Hydra help text;
find rna_predict -name __main__.py returns exactly one path.

---

VERIFY (binary green/red):

[1] MODULE EXISTS ON DISK
    cmd: find rna_predict -name __main__.py
    pass: exactly one line of output — rna_predict/__main__.py

[2] IMPORT RESOLVES AT RUNTIME
    cmd: python -c "from rna_predict.__main__ import main; print('ok')"
    pass: prints "ok", exits 0; no ModuleNotFoundError or ImportError

[3] PYTHON -m ENTRY POINT WORKS
    cmd: python -m rna_predict --help
    pass: exits 0; stdout contains Hydra help text (e.g. "Override any config" or "app.cfg"
          or "--help" usage block from Hydra's default output)

[4] CONSOLE SCRIPT LIVE-FIRE (canonical verification from finding)
    cmd: uv run rna_predict --help
    pass: exits 0; stdout contains Hydra help text matching [3] output pattern

[5] DELEGATION PATH (path-conditional — implement whichever path was chosen)
    Option A — delegate to rna_predict.interface:main:
        check: grep -n "from rna_predict.interface import\|rna_predict\.interface" rna_predict/__main__.py
        Option A pass: at least one matching line returned

    Option B — delegate to rna_predict.main:main:
        check: grep -n "from rna_predict\.main import\|rna_predict\.main" rna_predict/__main__.py
        Option B pass: at least one matching line returned

    (Exactly one option must be GREEN; the other is N/A for this PR.)

[6] NO DOUBLE-HYDRA-MAIN DECORATION
    cmd: uv run rna_predict --help
    pass: exits 0 with no HydraException or "HydraConfig is already set" error; the call chain
          from __main__.main() reaches exactly one @hydra.main decorator
    drill: if [4] fails with a Hydra exception, trace the import chain;
           rna_predict/__main__.py must not define its own @hydra.main wrapping an
           already-decorated delegate

[7] LOCAL-CI PASSES (floor — regression gate)
    cmd: make test
    pass: exits 0; pytest reports 0 failures and 0 errors attributable to this patch;
          ruff lint/format step within make test also exits 0

[8] PACKET VALIDATES (floor — substrate gate)
    cmd: aiv check
    pass: exits 0; the AIV packet for this commit is well-formed and passes all checks;
          packet Class E (intent alignment) URL resolves to the canonical audit source:
          https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/1f6481e4d8d7c673f44115c0a5bbaa1703ebe562/audit/02-static-audit.md#L48
          (never a local taskmaster task or pipeline launch-brief)

> Floor slots dropped with notes:
>   NO-ATTRIBUTION: dropped — this is the AI-driven track; agent commits are expected;
>                   human acts are H1 (finding) and H2 (judge+merge) only.
>   PROGRESS-TRACKER CLOSURE: dropped — review.spec_sections.progress_tracker not configured.

---

PRE-MERGE:
- [ ] All VERIFY items [1]–[8] are GREEN (zero red, zero skipped without documented rationale)
- [ ] Review body read in full — not just the review status check; no load-bearing unresolved
      comments remain
- [ ] Quiet-window: no new commits pushed after the final review approval (rebase only if
      the reviewer explicitly requests it; a force-push resets the quiet window)

---

POST-MERGE:
- Bookkeeping: mark finding s2c0l0-003 resolved in the audit tracking record; record the merge
  commit SHA against the finding entry in audit/02-static-audit.md or the project's finding log.
- Unblock: s2c0l0-002 (Containerfile:1 Python version fix) is independent of this PR and may
  merge in either order; notify whoever holds that branch that the console script is now
  functional so a full container smoke-test (`CMD ["rna_predict"]`) can be attempted once both
  land.
- Triggers: N/A — no downstream artifact or deployment pipeline depends on __main__.py at
  deploy time; the console script is the sole consumer and it is not part of a build artifact.
- Retro-verify: from a clean install (`pip install -e . && rna_predict --help` OR
  `uv run rna_predict --help` from a fresh shell), confirm the console script works outside
  the dev worktree; record pass/fail in the PR retro comment.

---

OUT-OF-SCOPE REMINDERS:
- Containerfile:1 Python 3.7→3.10+ fix — s2c0l0-002; do not include in this PR.
- CI requirements.txt corruption (main.yml:39) — s2c0l0-004; do not include in this PR.
- setup.py / release.yml version drift — s2c0l2-0006; do not include in this PR.
- DimensionsConfig production-defaults restoration — s2c1l2-dimsconfig-reduced-defaults; do not
  include in this PR.
- Smoke-test addition for rna_predict --help in tests/ — nice-to-have; defer to test-debt round.

===== END CONTRACT =====
