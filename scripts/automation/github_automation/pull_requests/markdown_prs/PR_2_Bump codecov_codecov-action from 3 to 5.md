# Pull Request #2: Bump codecov/codecov-action from 3 to 5

## Status
- State: CLOSED
- Created: 2025-03-12
- Updated: 2025-03-18
- Closed: 2025-03-18
- Merged: N/A

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
Bumps [codecov/codecov-action](https://github.com/codecov/codecov-action) from 3 to 5.

Release notes
Sourced from codecov/codecov-action's releases.

v5.0.0
v5 Release
v5 of the Codecov GitHub Action will use the Codecov Wrapper to encapsulate the CLI. This will help ensure that the Action gets updates quicker.
Migration Guide
The v5 release also coincides with the opt-out feature for tokens for public repositories. In the Global Upload Token section of the settings page of an organization in codecov.io, you can set the ability for Codecov to receive a coverage reports from any source. This will allow contributors or other members of a repository to upload without needing access to the Codecov token. For more details see how to upload without a token.

[!WARNING]
The following arguments have been changed

file (this has been deprecated in favor of files)
plugin (this has been deprecated in favor of plugins)


The following arguments have been added:

binary
gcov_args
gcov_executable
gcov_ignore
gcov_include
report_type
skip_validation
swift_project

You can see their usage in the action.yml file.
What's Changed

chore(deps): bump to eslint9+ and remove eslint-config-google by @​thomasrockhu-codecov in codecov/codecov-action#1591
build(deps-dev): bump @​octokit/webhooks-types from 7.5.1 to 7.6.1 by @​dependabot in codecov/codecov-action#1595
build(deps-dev): bump typescript from 5.6.2 to 5.6.3 by @​dependabot in codecov/codecov-action#1604
build(deps-dev): bump @​typescript-eslint/parser from 8.8.0 to 8.8.1 by @​dependabot in codecov/codecov-action#1601
build(deps): bump @​actions/core from 1.11.0 to 1.11.1 by @​dependabot in codecov/codecov-action#1597
build(deps): bump github/codeql-action from 3.26.9 to 3.26.11 by @​dependabot in codecov/codecov-action#1596
build(deps-dev): bump @​typescript-eslint/eslint-plugin from 8.8.0 to 8.8.1 by @​dependabot in codecov/codecov-action#1600
build(deps-dev): bump eslint from 9.11.1 to 9.12.0 by @​dependabot in codecov/codecov-action#1598
build(deps): bump github/codeql-action from 3.26.11 to 3.26.12 by @​dependabot in codecov/codecov-action#1609
build(deps): bump actions/checkout from 4.2.0 to 4.2.1 by @​dependabot in codecov/codecov-action#1608
build(deps): bump actions/upload-artifact from 4.4.0 to 4.4.3 by @​dependabot in codecov/codecov-action#1607
build(deps-dev): bump @​typescript-eslint/parser from 8.8.1 to 8.9.0 by @​dependabot in codecov/codecov-action#1612
build(deps-dev): bump @​typescript-eslint/eslint-plugin from 8.8.1 to 8.9.0 by @​dependabot in codecov/codecov-action#1611
build(deps-dev): bump @​typescript-eslint/eslint-plugin from 8.9.0 to 8.10.0 by @​dependabot in codecov/codecov-action#1615
build(deps-dev): bump eslint from 9.12.0 to 9.13.0 by @​dependabot in codecov/codecov-action#1618
build(deps): bump github/codeql-action from 3.26.12 to 3.26.13 by @​dependabot in codecov/codecov-action#1617
build(deps-dev): bump @​typescript-eslint/parser from 8.9.0 to 8.10.0 by @​dependabot in codecov/codecov-action#1614
build(deps-dev): bump @​typescript-eslint/eslint-plugin from 8.10.0 to 8.11.0 by @​dependabot in codecov/codecov-action#1620
build(deps-dev): bump @​typescript-eslint/parser from 8.10.0 to 8.11.0 by @​dependabot in codecov/codecov-action#1619
build(deps-dev): bump @​types/jest from 29.5.13 to 29.5.14 by @​dependabot in codecov/codecov-action#1622
build(deps): bump actions/checkout from 4.2.1 to 4.2.2 by @​dependabot in codecov/codecov-action#1625
build(deps): bump github/codeql-action from 3.26.13 to 3.27.0 by @​dependabot in codecov/codecov-action#1624
build(deps-dev): bump @​typescript-eslint/eslint-plugin from 8.11.0 to 8.12.1 by @​dependabot in codecov/codecov-action#1626
build(deps-dev): bump @​typescript-eslint/eslint-plugin from 8.12.1 to 8.12.2 by @​dependabot in codecov/codecov-action#1629



... (truncated)


Changelog
Sourced from codecov/codecov-action's changelog.

v5 Release
v5 of the Codecov GitHub Action will use the Codecov Wrapper to encapsulate the CLI. This will help ensure that the Action gets updates quicker.
Migration Guide
The v5 release also coincides with the opt-out feature for tokens for public repositories. In the Global Upload Token section of the settings page of an organization in codecov.io, you can set the ability for Codecov to receive a coverage reports from any source. This will allow contributors or other members of a repository to upload without needing access to the Codecov token. For more details see how to upload without a token.

[!WARNING]
The following arguments have been changed

file (this has been deprecated in favor of files)
plugin (this has been deprecated in favor of plugins)


The following arguments have been added:

binary
gcov_args
gcov_executable
gcov_ignore
gcov_include
report_type
skip_validation
swift_project

You can see their usage in the action.yml file.
What's Changed

chore(deps): bump to eslint9+ and remove eslint-config-google by @​thomasrockhu-codecov in codecov/codecov-action#1591
build(deps-dev): bump @​octokit/webhooks-types from 7.5.1 to 7.6.1 by @​dependabot in codecov/codecov-action#1595
build(deps-dev): bump typescript from 5.6.2 to 5.6.3 by @​dependabot in codecov/codecov-action#1604
build(deps-dev): bump @​typescript-eslint/parser from 8.8.0 to 8.8.1 by @​dependabot in codecov/codecov-action#1601
build(deps): bump @​actions/core from 1.11.0 to 1.11.1 by @​dependabot in codecov/codecov-action#1597
build(deps): bump github/codeql-action from 3.26.9 to 3.26.11 by @​dependabot in codecov/codecov-action#1596
build(deps-dev): bump @​typescript-eslint/eslint-plugin from 8.8.0 to 8.8.1 by @​dependabot in codecov/codecov-action#1600
build(deps-dev): bump eslint from 9.11.1 to 9.12.0 by @​dependabot in codecov/codecov-action#1598
build(deps): bump github/codeql-action from 3.26.11 to 3.26.12 by @​dependabot in codecov/codecov-action#1609
build(deps): bump actions/checkout from 4.2.0 to 4.2.1 by @​dependabot in codecov/codecov-action#1608
build(deps): bump actions/upload-artifact from 4.4.0 to 4.4.3 by @​dependabot in codecov/codecov-action#1607
build(deps-dev): bump @​typescript-eslint/parser from 8.8.1 to 8.9.0 by @​dependabot in codecov/codecov-action#1612
build(deps-dev): bump @​typescript-eslint/eslint-plugin from 8.8.1 to 8.9.0 by @​dependabot in codecov/codecov-action#1611
build(deps-dev): bump @​typescript-eslint/eslint-plugin from 8.9.0 to 8.10.0 by @​dependabot in codecov/codecov-action#1615
build(deps-dev): bump eslint from 9.12.0 to 9.13.0 by @​dependabot in codecov/codecov-action#1618
build(deps): bump github/codeql-action from 3.26.12 to 3.26.13 by @​dependabot in codecov/codecov-action#1617
build(deps-dev): bump @​typescript-eslint/parser from 8.9.0 to 8.10.0 by @​dependabot in codecov/codecov-action#1614
build(deps-dev): bump @​typescript-eslint/eslint-plugin from 8.10.0 to 8.11.0 by @​dependabot in codecov/codecov-action#1620
build(deps-dev): bump @​typescript-eslint/parser from 8.10.0 to 8.11.0 by @​dependabot in codecov/codecov-action#1619
build(deps-dev): bump @​types/jest from 29.5.13 to 29.5.14 by @​dependabot in codecov/codecov-action#1622
build(deps): bump actions/checkout from 4.2.1 to 4.2.2 by @​dependabot in codecov/codecov-action#1625
build(deps): bump github/codeql-action from 3.26.13 to 3.27.0 by @​dependabot in codecov/codecov-action#1624
build(deps-dev): bump @​typescript-eslint/eslint-plugin from 8.11.0 to 8.12.1 by @​dependabot in codecov/codecov-action#1626
build(deps-dev): bump @​typescript-eslint/eslint-plugin from 8.12.1 to 8.12.2 by @​dependabot in codecov/codecov-action#1629
build(deps-dev): bump @​typescript-eslint/parser from 8.11.0 to 8.12.2 by @​dependabot in codecov/codecov-action#1628



... (truncated)


Commits

0565863 chore(release): 5.4.0 (#1781)
c545d7b update wrapper submodule to 0.2.0, add recurse_submodules arg (#1780)
2488e99 build(deps): bump actions/upload-artifact from 4.6.0 to 4.6.1 (#1775)
a46c158 build(deps): bump ossf/scorecard-action from 2.4.0 to 2.4.1 (#1776)
062ee7e build(deps): bump github/codeql-action from 3.28.9 to 3.28.10 (#1777)
1fecca8 Clarify in README that use_pypi bypasses integrity checks too (#1773)
2e6e9c5 Fix use of safe.directory inside containers (#1768)
a5dc5a5 Fix description for report_type input (#1770)
4898080 build(deps): bump github/codeql-action from 3.28.8 to 3.28.9 (#1765)
5efa07b Fix a typo in the example (#1758)
Additional commits viewable in compare view




[![Dependabot compatibility score](https://dependabot-badges.githubapp.com/badges/compatibility_score?dependency-name=codecov/codecov-action&package-manager=github_actions&previous-version=3&new-version=5)](https://docs.github.com/en/github/managing-security-vulnerabilities/about-dependabot-security-updates#about-compatibility-scores)

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

### Comment by dependabot
- Created: 2025-03-18
- Author Association: NONE

Looks like codecov/codecov-action is up-to-date now, so this is no longer needed.

---
