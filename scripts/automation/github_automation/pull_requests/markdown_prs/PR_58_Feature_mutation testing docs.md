# Pull Request #58: Feature/mutation testing docs

## Status
- State: MERGED
- Created: 2025-04-14
- Updated: 2025-04-16
- Closed: 2025-04-16
- Merged: 2025-04-16

## Changes
- Additions: 6474
- Deletions: 258
- Changed Files: 52

## Author
- Name: ImmortalDemonGod
- Login: ImmortalDemonGod
- Bot: No

## Assignees
- None

## Description
### Summary :memo:
_Write an overview about it._

### Details
_Describe more what you did on changes._
1. (...)
2. (...)

### Bugfixes :bug: (delete if dind't have any)
-

### Checks
- [ ] Closed #798
- [ ] Tested Changes
- [ ] Stakeholder Approval

## Summary by CodeRabbit

- **New Features**
  - Added configuration files for Cosmic Ray and Mutatest to enable mutation testing.
  - Introduced scripts and documentation for running and configuring mutation testing tools.
  - Added a new CLI entry script for Task Master and a Bash wrapper for easier task management.
  - Created a script to reorganize project scripts into categorized subdirectories with README summaries.
  - Added a configurable mutation testing runner script with error handling and reporting.

- **Documentation**
  - Added comprehensive guides and references for mutation testing using both Cosmic Ray and Mutatest, including setup, usage, reporting, and best practices.
  - Provided step-by-step tutorials and troubleshooting sections to assist users in integrating mutation testing into their workflow.
  - Added README files describing the purpose of new script subdirectories and overall script organization.

- **Chores**
  - Updated development dependencies to include Cosmic Ray and Mutatest.
  - Enhanced .gitignore to exclude mutation test reports, specific test output files, and selectively track scripts.

- **Bug Fixes**
  - Improved test cleanup logic to correctly handle symbolic links during integration test teardown.

- **Refactor**
  - Removed an old test reorganization script replaced by a new, more comprehensive script for organizing test and utility scripts.

## Comments

### Comment by coderabbitai
- Created: 2025-04-14
- Author Association: NONE

> [!CAUTION]
> ## Review failed
> 
> The pull request is closed.

## Walkthrough

This update introduces comprehensive support and documentation for mutation testing in the project. New configuration files for both Cosmic Ray and Mutatest mutation testing tools are added, alongside detailed user guides and reference documentation. The development dependencies are updated to include these tools. The `.gitignore` is expanded to exclude mutation test reports and specific test output files. Additionally, a test script is improved to handle symbolic links more robustly during cleanup operations. A new shell script reorganizes the `scripts` directory into categorized subfolders with README files, and an obsolete test reorganization script is removed. New CLI and automation scripts are added, including a Task Master CLI entry point and mutation test runner with error handling. No changes are made to public code interfaces or exported entities.

## Changes

| File(s)                                                                                       | Change Summary                                                                                                                                                                                                                                                                                                                                                          |
|-----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `cosmic-ray.dev.toml`, `mutatest.ini`                                                         | Added configuration files for Cosmic Ray and Mutatest mutation testing tools, specifying modules, test commands, exclusion patterns, timeouts, and parallel execution settings.                                                                                                                                                |
| `pyproject.toml`                                                                              | Added "cosmic-ray" and "mutatest" to the development dependencies.                                                                                                                                                                                                                                                             |
| `docs/testing/cosmic-ray.md`, `docs/testing/mutation_testing_guide.md`, `docs/testing/mutation_testing.md` | Added extensive documentation and guides for mutation testing using Cosmic Ray and Mutatest, including concepts, setup, configuration, advanced usage, troubleshooting, and best practices.                                                                                                                                    |
| `docs/source/...`                                                                             | Introduced a full set of reStructuredText documentation files for Cosmic Ray, including how-tos, references, concepts, theory, tutorials, and continuous integration instructions.                                                                                                                                            |
| `tests/stageA/integration/test_run_stageA.py`                                                 | Improved symbolic link handling in test cleanup: now checks for and unlinks symlinks instead of recursively deleting them.                                                                                                                                                                                                    |
| `.gitignore`                                                                                  | Updated to stop ignoring the entire `scripts/` directory, added script-specific ignores for cache and metadata files, added `predict_test.ct` to ignore, and ignore all `.rst` files in `reports/mutation_tests/` except `.gitkeep`.                                                                                                                                             |
| `reorganize_scripts.sh`                                                                       | Added a new shell script to reorganize the `scripts` directory into categorized subdirectories with README files describing their purposes.                                                                                                                                                                                  |
| `rna_predict/scripts/reorg_tests.sh`                                                         | Deleted the old test reorganization script that moved and structured test files into stage and category folders.                                                                                                                                                                                                                |
| `scripts/README.md`, `scripts/analysis/README.md`, `scripts/automation/README.md`, `scripts/coverage/README.md`, `scripts/test_utils/README.md` | Added README files to `scripts` and its new subdirectories describing their contents and purposes.                                                                                                                                                                                                                              |
| `scripts/dev.js`                                                                             | Added a new executable JavaScript CLI entry point for the Task Master CLI, delegating command processing to a modular command handler with optional debug logging.                                                                                                                                                              |
| `scripts/example_prd.txt`                                                                    | Added a new product requirements document template with structured sections for context and detailed PRD elements.                                                                                                                                                                                                              |
| `scripts/run_mutation_tests.sh`                                                             | Added a new Bash script to run mutation tests with Mutatest, supporting configurable parameters, error detection, and logging.                                                                                                                                                                                                  |
| `scripts/task-complexity-report.json`                                                       | Added a JSON report detailing complexity analysis and subtasks for 31 development tasks related to the project.                                                                                                                                                                                                                  |
| `scripts/task-master`                                                                        | Added a Bash wrapper script to run the Node.js `task-master` CLI from the project root, forwarding arguments and ensuring output capture.                                                                                                                                                                                      |

## Poem

> In the warren, code is hopping bright,  
> With mutants lurking out of sight.  
> Tools and guides now lead the way,  
> To test, mutate, and save the day!  
> Reports are hidden, links unspun—  
> This bunny’s work is never done.  
> 🐇✨

> [!TIP]
> 
> ⚡💬 Agentic Chat (Pro Plan, General Availability)
> 
> - We're introducing multi-step agentic chat in review comments and issue comments, within and outside of PR's. This feature enhances review and issue discussions with the CodeRabbit agentic chat by enabling advanced interactions, including the ability to create pull requests directly from comments and add commits to existing pull requests.
> 
> 

---


📜 Recent review details

**Configuration used: CodeRabbit UI**
**Review profile: CHILL**
**Plan: Pro**


📥 Commits

Reviewing files that changed from the base of the PR and between 7251df81910eb8e4059110be6a36964a3eb7cab2 and 6408d82e07249d0b1e87ad95a30a910710d0ea89.




⛔ Files ignored due to path filters (2)

* `scripts/screen_finder_app/templates/roo_question.png` is excluded by `!**/*.png`
* `scripts/screen_finder_app/templates/template.png` is excluded by `!**/*.png`




📒 Files selected for processing (13)

* `.gitignore` (1 hunks)
* `reorganize_scripts.sh` (1 hunks)
* `rna_predict/scripts/reorg_tests.sh` (0 hunks)
* `scripts/README.md` (1 hunks)
* `scripts/analysis/README.md` (1 hunks)
* `scripts/automation/README.md` (1 hunks)
* `scripts/coverage/README.md` (1 hunks)
* `scripts/dev.js` (1 hunks)
* `scripts/example_prd.txt` (1 hunks)
* `scripts/run_mutation_tests.sh` (1 hunks)
* `scripts/task-complexity-report.json` (1 hunks)
* `scripts/task-master` (1 hunks)
* `scripts/test_utils/README.md` (1 hunks)






✨ Finishing Touches

- [ ]  📝 Generate Docstrings



---


🪧 Tips

### Chat

There are 3 ways to chat with [CodeRabbit](https://coderabbit.ai?utm_source=oss&utm_medium=github&utm_campaign=ImmortalDemonGod/RNA_PREDICT&utm_content=58):

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
- `@coderabbitai generate docstrings` to [generate docstrings](https://docs.coderabbit.ai/finishing-touches/docstrings) for this PR.
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
