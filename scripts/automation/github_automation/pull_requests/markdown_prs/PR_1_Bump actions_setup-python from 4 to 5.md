# Pull Request #1: Bump actions/setup-python from 4 to 5

## Status
- State: MERGED
- Created: 2025-03-12
- Updated: 2025-03-22
- Closed: 2025-03-22
- Merged: 2025-03-22

## Changes
- Additions: 7
- Deletions: 7
- Changed Files: 3

## Author
- Name: N/A
- Login: app/dependabot
- Bot: Yes

## Assignees
- None

## Description
Bumps [actions/setup-python](https://github.com/actions/setup-python) from 4 to 5.

Release notes
Sourced from actions/setup-python's releases.

v5.0.0
What's Changed
In scope of this release, we update node version runtime from node16 to node20 (actions/setup-python#772). Besides, we update dependencies to the latest versions.
Full Changelog: https://github.com/actions/setup-python/compare/v4.8.0...v5.0.0
v4.8.0
What's Changed
In scope of this release we added support for GraalPy (actions/setup-python#694). You can use this snippet to set up GraalPy:
steps:
- uses: actions/checkout@v4
- uses: actions/setup-python@v4 
  with:
    python-version: 'graalpy-22.3' 
- run: python my_script.py

Besides, the release contains such changes as:

Trim python version when reading from file by @​FerranPares in actions/setup-python#628
Use non-deprecated versions in examples by @​jeffwidman in actions/setup-python#724
Change deprecation comment to past tense by @​jeffwidman in actions/setup-python#723
Bump @​babel/traverse from 7.9.0 to 7.23.2 by @​dependabot in actions/setup-python#743
advanced-usage.md: Encourage the use actions/checkout@v4 by @​cclauss in actions/setup-python#729
Examples now use checkout@v4 by @​simonw in actions/setup-python#738
Update actions/checkout to v4 by @​dmitry-shibanov in actions/setup-python#761

New Contributors

@​FerranPares made their first contribution in actions/setup-python#628
@​timfel made their first contribution in actions/setup-python#694
@​jeffwidman made their first contribution in actions/setup-python#724

Full Changelog: https://github.com/actions/setup-python/compare/v4...v4.8.0
v4.7.1
What's Changed

Bump word-wrap from 1.2.3 to 1.2.4 by @​dependabot in actions/setup-python#702
Add range validation for toml files by @​dmitry-shibanov in actions/setup-python#726

Full Changelog: https://github.com/actions/setup-python/compare/v4...v4.7.1
v4.7.0
In scope of this release, the support for reading python version from pyproject.toml was added (actions/setup-python#669).
      - name: Setup Python
        uses: actions/setup-python@v4
&lt;/tr&gt;&lt;/table&gt; 


... (truncated)


Commits

4237552 Improve Advanced Usage examples (#645)
709bfa5 Bump requests from 2.24.0 to 2.32.2 in /tests/data (#1019)
ceb20b2 Bump @​actions/http-client from 2.2.1 to 2.2.3 (#1020)
0dc2d2c Bump actions/publish-immutable-action from 0.0.3 to 0.0.4 (#1014)
feb9c6e Bump urllib3 from 1.25.9 to 1.26.19 in /tests/data (#895)
d0b4fc4 Bump undici from 5.28.4 to 5.28.5 (#1012)
e3dfaac Configure Dependabot settings (#1008)
b8cf3eb Use the new cache service: upgrade @actions/cache to ^4.0.0 (#1007)
1928ae6 Update README.md (#1009)
3fddbee Enhance Workflows: Add Ubuntu-24, Remove Python 3.8  (#985)
Additional commits viewable in compare view




[![Dependabot compatibility score](https://dependabot-badges.githubapp.com/badges/compatibility_score?dependency-name=actions/setup-python&package-manager=github_actions&previous-version=4&new-version=5)](https://docs.github.com/en/github/managing-security-vulnerabilities/about-dependabot-security-updates#about-compatibility-scores)

Dependabot will resolve any conflicts with this PR as long as you don't alter it yourself. You can also trigger a rebase manually by commenting `@dependabot rebase`.

[//]: # (dependabot-automerge-start)
[//]: # (dependabot-automerge-end)

---


Dependabot commands and options


You can trigger Dependabot actions by commenting on this PR:
- `@dependabot rebase` will rebase this PR
- `@dependabot recreate` will recreate this PR, overwriting any edits that have been made to it
- `@dependabot merge` will merge this PR after your CI passes on it
- `@dependabot squash and merge` will squash and merge this PR after your CI passes on it
- `@dependabot cancel merge` will cancel a previously requested merge and block automerging
- `@dependabot reopen` will reopen this PR if it is closed
- `@dependabot close` will close this PR and stop Dependabot recreating it. You can achieve the same result by closing it manually
- `@dependabot show  ignore conditions` will show all of the ignore conditions of the specified dependency
- `@dependabot ignore this major version` will close this PR and stop Dependabot creating any more for this major version (unless you reopen the PR or upgrade to it yourself)
- `@dependabot ignore this minor version` will close this PR and stop Dependabot creating any more for this minor version (unless you reopen the PR or upgrade to it yourself)
- `@dependabot ignore this dependency` will close this PR and stop Dependabot creating any more for this dependency (unless you reopen the PR or upgrade to it yourself)

## Comments
