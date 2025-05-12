# Pull Request #5: Bump actions/checkout from 3 to 4

## Status
- State: MERGED
- Created: 2025-03-12
- Updated: 2025-03-22
- Closed: 2025-03-22
- Merged: 2025-03-22

## Changes
- Additions: 8
- Deletions: 8
- Changed Files: 4

## Author
- Name: N/A
- Login: app/dependabot
- Bot: Yes

## Assignees
- None

## Description
Bumps [actions/checkout](https://github.com/actions/checkout) from 3 to 4.

Release notes
Sourced from actions/checkout's releases.

v4.0.0
What's Changed

Update default runtime to node20 by @​takost in actions/checkout#1436
Support fetching without the --progress option by @​simonbaird in actions/checkout#1067
Release 4.0.0 by @​takost in actions/checkout#1447

New Contributors

@​takost made their first contribution in actions/checkout#1436
@​simonbaird made their first contribution in actions/checkout#1067

Full Changelog: https://github.com/actions/checkout/compare/v3...v4.0.0
v3.6.0
What's Changed

Mark test scripts with Bash'isms to be run via Bash by @​dscho in actions/checkout#1377
Add option to fetch tags even if fetch-depth &gt; 0 by @​RobertWieczoreck in actions/checkout#579
Release 3.6.0 by @​luketomlinson in actions/checkout#1437

New Contributors

@​RobertWieczoreck made their first contribution in actions/checkout#579
@​luketomlinson made their first contribution in actions/checkout#1437

Full Changelog: https://github.com/actions/checkout/compare/v3.5.3...v3.6.0
v3.5.3
What's Changed

Fix: Checkout Issue in self hosted runner due to faulty submodule check-ins by @​megamanics in actions/checkout#1196
Fix typos found by codespell by @​DimitriPapadopoulos in actions/checkout#1287
Add support for sparse checkouts by @​dscho and @​dfdez in actions/checkout#1369
Release v3.5.3 by @​TingluoHuang in actions/checkout#1376

New Contributors

@​megamanics made their first contribution in actions/checkout#1196
@​DimitriPapadopoulos made their first contribution in actions/checkout#1287
@​dfdez made their first contribution in actions/checkout#1369

Full Changelog: https://github.com/actions/checkout/compare/v3...v3.5.3
v3.5.2
What's Changed

Fix: Use correct API url / endpoint in GHES by @​fhammerl in actions/checkout#1289 based on #1286 by @​1newsr

Full Changelog: https://github.com/actions/checkout/compare/v3.5.1...v3.5.2
v3.5.1
What's Changed

Improve checkout performance on Windows runners by upgrading @​actions/github dependency by @​BrettDong in actions/checkout#1246

New Contributors

@​BrettDong made their first contribution in actions/checkout#1246



... (truncated)


Changelog
Sourced from actions/checkout's changelog.

Changelog
v4.2.2

url-helper.ts now leverages well-known environment variables by @​jww3 in actions/checkout#1941
Expand unit test coverage for isGhes by @​jww3 in actions/checkout#1946

v4.2.1

Check out other refs/* by commit if provided, fall back to ref by @​orhantoy in actions/checkout#1924

v4.2.0

Add Ref and Commit outputs by @​lucacome in actions/checkout#1180
Dependency updates by @​dependabot- actions/checkout#1777, actions/checkout#1872

v4.1.7

Bump the minor-npm-dependencies group across 1 directory with 4 updates by @​dependabot in actions/checkout#1739
Bump actions/checkout from 3 to 4 by @​dependabot in actions/checkout#1697
Check out other refs/* by commit by @​orhantoy in actions/checkout#1774
Pin actions/checkout's own workflows to a known, good, stable version. by @​jww3 in actions/checkout#1776

v4.1.6

Check platform to set archive extension appropriately by @​cory-miller in actions/checkout#1732

v4.1.5

Update NPM dependencies by @​cory-miller in actions/checkout#1703
Bump github/codeql-action from 2 to 3 by @​dependabot in actions/checkout#1694
Bump actions/setup-node from 1 to 4 by @​dependabot in actions/checkout#1696
Bump actions/upload-artifact from 2 to 4 by @​dependabot in actions/checkout#1695
README: Suggest user.email to be 41898282+github-actions[bot]@users.noreply.github.com by @​cory-miller in actions/checkout#1707

v4.1.4

Disable extensions.worktreeConfig when disabling sparse-checkout by @​jww3 in actions/checkout#1692
Add dependabot config by @​cory-miller in actions/checkout#1688
Bump the minor-actions-dependencies group with 2 updates by @​dependabot in actions/checkout#1693
Bump word-wrap from 1.2.3 to 1.2.5 by @​dependabot in actions/checkout#1643

v4.1.3

Check git version before attempting to disable sparse-checkout by @​jww3 in actions/checkout#1656
Add SSH user parameter by @​cory-miller in actions/checkout#1685
Update actions/checkout version in update-main-version.yml by @​jww3 in actions/checkout#1650

v4.1.2

Fix: Disable sparse checkout whenever sparse-checkout option is not present @​dscho in actions/checkout#1598

v4.1.1

Correct link to GitHub Docs by @​peterbe in actions/checkout#1511
Link to release page from what's new section by @​cory-miller in actions/checkout#1514

v4.1.0

Add support for partial checkout filters



... (truncated)


Commits

11bd719 Prepare 4.2.2 Release (#1953)
e3d2460 Expand unit test coverage (#1946)
163217d url-helper.ts now leverages well-known environment variables. (#1941)
eef6144 Prepare 4.2.1 release (#1925)
6b42224 Add workflow file for publishing releases to immutable action package (#1919)
de5a000 Check out other refs/* by commit if provided, fall back to ref (#1924)
d632683 Prepare 4.2.0 release (#1878)
6d193bf Bump braces from 3.0.2 to 3.0.3 (#1777)
db0cee9 Bump the minor-npm-dependencies group across 1 directory with 4 updates (#1872)
b684943 Add Ref and Commit outputs (#1180)
Additional commits viewable in compare view




[![Dependabot compatibility score](https://dependabot-badges.githubapp.com/badges/compatibility_score?dependency-name=actions/checkout&package-manager=github_actions&previous-version=3&new-version=4)](https://docs.github.com/en/github/managing-security-vulnerabilities/about-dependabot-security-updates#about-compatibility-scores)

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
