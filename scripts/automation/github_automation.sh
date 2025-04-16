#!/bin/bash

# github_automation.sh
# A script to automate common GitHub and Git operations for multiple branches,
# including robust commit history logging and now improved naming + pretty-printing of issues.

set -euo pipefail

# ----------------------------
# Configuration and Defaults
# ----------------------------

# Default base branch
DEFAULT_BASE_BRANCH="master"

# Default branch prefix to filter feature branches
DEFAULT_BRANCH_PREFIX="feature/"

# ----------------------------------------------
# CHANGED THIS PART to always use $(pwd):
# ----------------------------------------------
DEFAULT_OUTPUT_DIR="$(pwd)/github_automation_output"
# ----------------------------------------------

# ----------------------------
# Functions
# ----------------------------

# Function to display usage information
usage() {
  echo "Usage: $0 [-b base_branch] [-p branch_prefix] [-o output_dir]"
  echo ""
  echo "  -b    Specify the base branch (default: $DEFAULT_BASE_BRANCH)"
  echo "  -p    Specify the branch prefix to filter feature branches (default: $DEFAULT_BRANCH_PREFIX)"
  echo "  -o    Specify the output directory (default: $DEFAULT_OUTPUT_DIR)"
  echo "  -h    Show this help message"
  exit 1
}

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Function to extract repository in "owner/repo" format
get_repo() {
  local remote_url
  remote_url=$(git config --get remote.origin.url)

  # SSH pattern
  if [[ "$remote_url" =~ ^git@github\.com:(.+)/(.+)\.git$ ]]; then
    echo "${BASH_REMATCH[1]}/${BASH_REMATCH[2]}"
  # HTTPS pattern (including optional .git suffix)
  elif [[ "$remote_url" =~ ^https://github\.com/(.+)/(.+)(\.git)?$ ]]; then
    # Just remove the trailing .git from group #2, if present:
    repo_only="$(echo "${BASH_REMATCH[2]}" | sed 's/\.git$//')"
    echo "${BASH_REMATCH[1]}/$repo_only"
  else
    echo "Error: Unable to parse repository from remote URL '$remote_url'." >&2
    exit 1
  fi
}

# Function to list pull requests
list_pull_requests() {
  local repo="$1"
  local output_file="$2"

  echo "Fetching pull requests for repository: $repo"
  gh pr list \
    --repo "$repo" \
    --state all \
    --json number,title,author,state,createdAt,updatedAt,closedAt,mergedAt,url,body,labels,assignees,reviewDecision,comments,additions,deletions,changedFiles \
    --limit 100 | jq . > "$output_file"
  echo "Pull requests saved to $output_file"
}

# Function to list issues (PRETTY-PRINT + RENAME)
list_issues() {
  local repo="$1"
  local output_file="$2"

  echo "Fetching issues for repository: $repo"
  gh issue list \
    --repo "$repo" \
    --state all \
    --json number,title,state,body,labels,assignees,createdAt,updatedAt \
  | jq --indent 2 . > "$output_file"

  echo "Issues saved to $output_file"
}

# Function to generate git diff
generate_diff() {
  local base="$1"
  local feature="$2"
  local output_file="$3"

  echo "Generating diff between $base and $feature"
  git diff "$base..$feature" > "$output_file"
  echo "Diff saved to $output_file"
}

# Function to generate a robust git log
robust_generate_log() {
  local base="$1"
  local branch="$2"
  local output_file="$3"

  echo "Generating robust log for '$branch' compared to '$base'..."

  # 1. Find merge base
  local merge_base
  merge_base=$(git merge-base "$base" "$branch" || true)

  if [[ -z "$merge_base" ]]; then
    # If there's no valid merge base, just log the entire branch history
    echo "No valid merge-base found. Logging entire commit history for '$branch'."
    git log "$branch" --pretty=format:"%h - %an, %ar : %s" > "$output_file"
  else
    # Count commits between merge base and the branch
    local commit_count
    commit_count=$(git rev-list --count "$merge_base..$branch")

    if [[ "$commit_count" -eq 0 ]]; then
      echo "No unique commits found from $merge_base to '$branch'. Logging entire branch history."
      git log "$branch" --pretty=format:"%h - %an, %ar : %s" > "$output_file"
    else
      echo "Found $commit_count commits from $merge_base to '$branch'. Logging them."
      git log "$merge_base..$branch" --pretty=format:"%h - %an, %ar : %s" > "$output_file"
    fi
  fi

  # Check if the resulting log file is empty or not
  if [[ ! -s "$output_file" ]]; then
    echo "Warning: Log file for '$branch' is empty." >&2
  else
    echo "Log saved to $output_file"
    echo "Sample commits in '$branch':"
    head -n 5 "$output_file"
  fi
}

# Function to determine the primary branch (master or main)
determine_primary_branch() {
  if git show-ref --verify --quiet refs/heads/master; then
    echo "master"
  elif git show-ref --verify --quiet refs/heads/main; then
    echo "main"
  else
    echo "Error: Neither 'master' nor 'main' branch exists. Please specify the base branch using -b option." >&2
    exit 1
  fi
}

# ----------------------------
# Main Script Execution
# ----------------------------

# Parse command-line options
BASE_BRANCH="$DEFAULT_BASE_BRANCH"
BRANCH_PREFIX="$DEFAULT_BRANCH_PREFIX"
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"

while getopts ":b:p:o:h" opt; do
  case $opt in
    b)
      BASE_BRANCH="$OPTARG"
      ;;
    p)
      BRANCH_PREFIX="$OPTARG"
      ;;
    o)
      OUTPUT_DIR="$OPTARG"
      ;;
    h)
      usage
      ;;
    \?)
      echo "Error: Invalid option -$OPTARG" >&2
      usage
      ;;
    :)
      echo "Error: Option -$OPTARG requires an argument." >&2
      usage
      ;;
  esac
done

# Check for required commands
for cmd in gh jq git; do
  if ! command_exists "$cmd"; then
    echo "Error: Required command '$cmd' is not installed." >&2
    exit 1
  fi
done

# Make sure we have all branches up-to-date
echo "Fetching all remote branches to ensure we're up-to-date..."
git fetch --all --prune

# Get repository
REPO=$(get_repo)
echo "Repository detected: $REPO"

# Determine primary branch if base_branch is still set to the default
if [[ "$BASE_BRANCH" == "$DEFAULT_BASE_BRANCH" ]]; then
  BASE_BRANCH=$(determine_primary_branch)
  echo "Using primary branch as base: $BASE_BRANCH"
else
  # Verify that the specified base branch exists locally
  if ! git show-ref --verify --quiet refs/heads/"$BASE_BRANCH"; then
    echo "Error: Specified base branch '$BASE_BRANCH' does not exist locally." >&2
    exit 1
  fi
  echo "Using specified base branch: $BASE_BRANCH"
fi

# Create output directory and subdirectories
mkdir -p "$OUTPUT_DIR/pull_requests"
mkdir -p "$OUTPUT_DIR/issues"
mkdir -p "$OUTPUT_DIR/diffs"
mkdir -p "$OUTPUT_DIR/logs"

# Define output files
PR_OUTPUT="$OUTPUT_DIR/pull_requests/pull_requests_full_pretty.json"
ISSUE_OUTPUT="$OUTPUT_DIR/issues/all_issues_pretty.json"  # renamed from 'issues.json'

# Execute functions for pull requests and issues
list_pull_requests "$REPO" "$PR_OUTPUT"
list_issues "$REPO" "$ISSUE_OUTPUT"

# Fetch all local branches with the specified prefix
echo "Fetching local branches with prefix '$BRANCH_PREFIX'"
BRANCHES=$(git for-each-ref --format='%(refname:short)' refs/heads/ | grep "^$BRANCH_PREFIX" || true)

if [[ -z "$BRANCHES" ]]; then
  echo "No branches found with prefix '$BRANCH_PREFIX'."
else
  echo "Found branches:"
  echo "$BRANCHES"

  for BRANCH in $BRANCHES; do
    # Define output files for each branch
    DIFF_OUTPUT="$OUTPUT_DIR/diffs/${BRANCH//\//-}.diff"
    LOG_OUTPUT="$OUTPUT_DIR/logs/${BRANCH//\//-}.log"

    # Generate diff
    generate_diff "$BASE_BRANCH" "$BRANCH" "$DIFF_OUTPUT"
    # Generate robust log
    robust_generate_log "$BASE_BRANCH" "$BRANCH" "$LOG_OUTPUT"
  done
fi

echo "All operations completed successfully."
