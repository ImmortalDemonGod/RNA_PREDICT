# Pull Request #3: Bump stefanzweifel/git-auto-commit-action from 4 to 5

## Status
- State: MERGED
- Created: 2025-03-12
- Updated: 2025-03-22
- Closed: 2025-03-22
- Merged: 2025-03-22

## Changes
- Additions: 1
- Deletions: 1
- Changed Files: 1

## Author
- Name: N/A
- Login: app/dependabot
- Bot: Yes

## Assignees
- None

## Description
Bumps [stefanzweifel/git-auto-commit-action](https://github.com/stefanzweifel/git-auto-commit-action) from 4 to 5.

Release notes
Sourced from stefanzweifel/git-auto-commit-action's releases.

v5.0.0
New major release that bumps the default runtime to Node 20. There are no other breaking changes.
Changed

Update node version to node20 (#300) @​ryudaitakai
Add _log and _set_github_output functions (#273) @​stefanzweifel

Fixed

Seems like there is an extra space (#288) @​pedroamador
Fix git-auto-commit.yml (#277) @​zcong1993

Dependency Updates

Bump actions/checkout from 3 to 4 (#302) @​dependabot
Bump bats from 1.9.0 to 1.10.0 (#293) @​dependabot
Bump github/super-linter from 4 to 5 (#289) @​dependabot
Bump bats from 1.8.2 to 1.9.0 (#282) @​dependabot

v4.16.0
Changed

Don't commit files when only LF/CRLF changes (#265) @​ZeroRin
Update default email address of github-actions[bot] (#264) @​Teko012

Fixed

Fix link and text for workflow limitation (#263) @​Teko012

v4.15.4
Fixed

Let Action fail if git binary can't be located (#261) @​stefanzweifel

Dependency Updates

Bump github/super-linter from 3 to 4 (#258) @​dependabot
Bump bats from 1.7.0 to 1.8.2 (#259) @​dependabot
Bump actions/checkout from 2 to 3 (#257) @​dependabot

v4.15.3
Changed

Use deprecated set-output syntax if GITHUB_OUTPUT environment is not available (#255) @​stefanzweifel

v4.15.2
Changed

Replace set-output usage with GITHUB_OUTPUT (#252) @​amonshiz



... (truncated)


Changelog
Sourced from stefanzweifel/git-auto-commit-action's changelog.

v4.15.4 - 2022-11-05
Fixed

Let Action fail if git binary can't be located (#261) @​stefanzweifel

Dependency Updates

Bump github/super-linter from 3 to 4 (#258) @​dependabot
Bump bats from 1.7.0 to 1.8.2 (#259) @​dependabot
Bump actions/checkout from 2 to 3 (#257) @​dependabot

v4.15.3 - 2022-10-26
Changed

Use deprecated set-output syntax if GITHUB_OUTPUT environment is not available (#255) @​stefanzweifel

v4.15.2 - 2022-10-22
Changed

Replace set-output usage with GITHUB_OUTPUT (#252) @​amonshiz

v4.15.1 - 2022-10-10
Fixed

Run Action on Node16 (#247) @​stefanzweifel




Commits

e348103 Merge pull request #354 from parkerbxyz/patch-1
032ffbe Include github.actor_id in default commit_author
0b492c0 Bump bats from 1.11.0 to 1.11.1 (#353)
050015d Add Scope/Permissions documentation for PATs
573710f docs(README): fix broken protected branch docs link (#346)
e961da7 Update README.md (#343)
ac88237 Bump github/super-linter from 6 to 7 (#342)
be823a7 Bump github/super-linter from 5 to 6 (#335)
55a82ca Add Section on preventing infinite loops to README
18157e6 Update bug.yaml
Additional commits viewable in compare view




[![Dependabot compatibility score](https://dependabot-badges.githubapp.com/badges/compatibility_score?dependency-name=stefanzweifel/git-auto-commit-action&package-manager=github_actions&previous-version=4&new-version=5)](https://docs.github.com/en/github/managing-security-vulnerabilities/about-dependabot-security-updates#about-compatibility-scores)

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
