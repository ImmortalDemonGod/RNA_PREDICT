# Pull Request #64: build(deps): bump actions/upload-artifact from 3 to 4

## Status
- State: MERGED
- Created: 2025-04-21
- Updated: 2025-05-04
- Closed: 2025-05-04
- Merged: 2025-05-04

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
Bumps [actions/upload-artifact](https://github.com/actions/upload-artifact) from 3 to 4.

Release notes
Sourced from actions/upload-artifact's releases.

v4.0.0
What's Changed
The release of upload-artifact@v4 and download-artifact@v4 are major changes to the backend architecture of Artifacts. They have numerous performance and behavioral improvements.
ℹ️ However, this is a major update that includes breaking changes. Artifacts created with versions v3 and below are not compatible with the v4 actions. Uploads and downloads must use the same major actions versions. There are also key differences from previous versions that may require updates to your workflows.
For more information, please see:

The changelog post.
The README.
The migration documentation.
As well as the underlying npm package, @​actions/artifact documentation.

New Contributors

@​vmjoseph made their first contribution in actions/upload-artifact#464

Full Changelog: https://github.com/actions/upload-artifact/compare/v3...v4.0.0
v3.2.1
What's Changed
This fixes the include-hidden-files input introduced in https://github.com/actions/upload-artifact/releases/tag/v3.2.0

Ensure hidden files input is used by @​joshmgross in actions/upload-artifact#609

Full Changelog: https://github.com/actions/upload-artifact/compare/v3.2.0...v3.2.1
v3.2.1-node20
What's Changed
This fixes the include-hidden-files input introduced in https://github.com/actions/upload-artifact/releases/tag/v3.2.0-node20

Ensure hidden files input is used by @​joshmgross in actions/upload-artifact#608

Full Changelog: https://github.com/actions/upload-artifact/compare/v3.2.0-node20...v3.2.1-node20
v3.2.0
Notice: Breaking Changes :warning:
We will no longer include hidden files and folders by default in the upload-artifact action of this version. This reduces the risk that credentials are accidentally uploaded into artifacts. Customers who need to continue to upload these files can use a new option, include-hidden-files, to continue to do so.
See &quot;Notice of upcoming deprecations and breaking changes in GitHub Actions runners&quot; changelog and this issue for more details.
What's Changed

V3 backport: Exclude hidden files by default by @​SrRyan in actions/upload-artifact#604



... (truncated)


Commits

ea165f8 Merge pull request #685 from salmanmkc/salmanmkc/3-new-upload-artifacts-release
0839620 Prepare for new release of actions/upload-artifact with new toolkit cache ver...
4cec3d8 Merge pull request #673 from actions/yacaovsnc/artifact_2.2.2
e9fad96 license cache update for artifact
b26fd06 Update to use artifact 2.2.2 package
65c4c4a Merge pull request #662 from actions/yacaovsnc/add_variable_for_concurrency_a...
0207619 move files back to satisfy licensed ci
1ecca81 licensed cache updates
9742269 Expose env vars to controll concurrency and timeout
6f51ac0 Merge pull request #656 from bdehamer/bdehamer/artifact-digest
Additional commits viewable in compare view




[![Dependabot compatibility score](https://dependabot-badges.githubapp.com/badges/compatibility_score?dependency-name=actions/upload-artifact&package-manager=github_actions&previous-version=3&new-version=4)](https://docs.github.com/en/github/managing-security-vulnerabilities/about-dependabot-security-updates#about-compatibility-scores)

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

### Comment by coderabbitai
- Created: 2025-04-21
- Author Association: NONE

> [!IMPORTANT]
> ## Review skipped
> 
> Bot user detected.
> 
> To trigger a single review, invoke the `@coderabbitai review` command.
> 
> You can disable this status message by setting the `reviews.review_status` to `false` in the CodeRabbit configuration file.

---


🪧 Tips

### Chat

There are 3 ways to chat with [CodeRabbit](https://coderabbit.ai?utm_source=oss&utm_medium=github&utm_campaign=ImmortalDemonGod/RNA_PREDICT&utm_content=64):

- Review comments: Directly reply to a review comment made by CodeRabbit. Example:
  - `I pushed a fix in commit , please review it.`
  - `Generate unit testing code for this file.`
  - `Open a follow-up GitHub issue for this discussion.`
- Files and specific lines of code (under the "Files changed" tab): Tag `@coderabbitai` in a new review comment at the desired location with your query. Examples:
  - `@coderabbitai generate unit testing code for this file.`
  -	`@coderabbitai modularize this function.`
- PR comments: Tag `@coderabbitai` in a new PR comment to ask questions about the PR branch. For the best results, please provide a very specific query, as very limited context is provided in this mode. Examples:
  - `@coderabbitai gather interesting stats about this repository and render them as a table. Additionally, render a pie chart showing the language distribution in the codebase.`
  - `@coderabbitai read src/utils.ts and generate unit testing code.`
  - `@coderabbitai read the files in the src/scheduler package and generate a class diagram using mermaid and a README in the markdown format.`
  - `@coderabbitai help me debug CodeRabbit configuration file.`

Note: Be mindful of the bot's finite context window. It's strongly recommended to break down tasks such as reading entire modules into smaller chunks. For a focused discussion, use review comments to chat about specific files and their changes, instead of using the PR comments.

### CodeRabbit Commands (Invoked using PR comments)

- `@coderabbitai pause` to pause the reviews on a PR.
- `@coderabbitai resume` to resume the paused reviews.
- `@coderabbitai review` to trigger an incremental review. This is useful when automatic reviews are disabled for the repository.
- `@coderabbitai full review` to do a full review from scratch and review all the files again.
- `@coderabbitai summary` to regenerate the summary of the PR.
- `@coderabbitai resolve` resolve all the CodeRabbit review comments.
- `@coderabbitai configuration` to show the current CodeRabbit configuration for the repository.
- `@coderabbitai help` to get help.

### Other keywords and placeholders

- Add `@coderabbitai ignore` anywhere in the PR description to prevent this PR from being reviewed.
- Add `@coderabbitai summary` to generate the high-level summary at a specific location in the PR description.
- Add `@coderabbitai` anywhere in the PR title to generate the title automatically.

### CodeRabbit Configuration File (`.coderabbit.yaml`)

- You can programmatically configure CodeRabbit by adding a `.coderabbit.yaml` file to the root of your repository.
- Please see the [configuration documentation](https://docs.coderabbit.ai/guides/configure-coderabbit) for more information.
- If your editor has YAML language server enabled, you can add the path at the top of this file to enable auto-completion and validation: `# yaml-language-server: $schema=https://coderabbit.ai/integrations/schema.v2.json`

### Documentation and Community

- Visit our [Documentation](https://docs.coderabbit.ai) for detailed information on how to use CodeRabbit.
- Join our [Discord Community](http://discord.gg/coderabbit) to get help, request features, and share feedback.
- Follow us on [X/Twitter](https://twitter.com/coderabbitai) for updates and announcements.

---
