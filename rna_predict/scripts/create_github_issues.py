#!/usr/bin/env python3
"""
create_github_issues.py

Creates GitHub issues in the specified repository from a JSON file.

Usage:
  1) Explicit owner/repo:
     python create_github_issues.py <owner> <repo> <issues.json>

     python quick-fixes/automation-scripts/create_github_issues.py ImmortalDemonGod ProjectEquiSurv /Users/tomriddle1/ProjectEquiSurv/issue.json

  2) Auto-detect owner/repo from your local Git remote:
     python create_github_issues.py <issues.json>

Requirements:
  - Have 'requests' installed: `pip install requests`
  - Have 'git' installed and a valid remote origin in your local repo
  - An environment variable GITHUB_TOKEN with your GitHub Personal Access Token
"""

import json
import os
import re
import subprocess
import sys
from typing import Tuple

import requests


def load_issues(json_path: str):
    """
    Load an array of issues from a JSON file.
    """
    if not os.path.exists(json_path):
        print(f"Error: File '{json_path}' does not exist.")
        sys.exit(1)
    with open(json_path, "r", encoding="utf-8") as f:
        issues = json.load(f)
    return issues


def get_repo_from_git() -> Tuple[str, str]:
    """
    Parse 'owner' and 'repo' from the local git remote URL.
    E.g.:
      git@github.com:OWNER/REPO.git
      https://github.com/OWNER/REPO.git
    Returns (owner, repo).
    Exits if it cannot parse the remote.
    """
    try:
        remote_url = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"], text=True
        ).strip()
    except subprocess.CalledProcessError:
        print("Error: Unable to retrieve remote.origin.url from git config.")
        sys.exit(1)

    # SSH Format: git@github.com:OWNER/REPO(.git)
    match_ssh = re.match(r"^git@github\.com:(.+)/(.+)(\.git)?$", remote_url)
    if match_ssh:
        return match_ssh.group(1), match_ssh.group(2)

    # HTTPS Format: https://github.com/OWNER/REPO(.git)
    match_https = re.match(r"^https://github\.com/(.+)/(.+)(\.git)?$", remote_url)
    if match_https:
        return match_https.group(1), match_https.group(2)

    print(f"Error: Unable to parse owner/repo from remote URL: {remote_url}")
    sys.exit(1)


def create_issue(owner: str, repo: str, issue_data: dict, headers: dict):
    """
    Make a POST request to GitHub's API to create an issue.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"

    # Construct minimal body
    data = {"title": issue_data["title"], "body": issue_data["body"]}

    # Labels, if present
    if "labels" in issue_data and issue_data["labels"]:
        data["labels"] = issue_data["labels"]

    # Assignees, if present
    if "assignees" in issue_data and issue_data["assignees"]:
        data["assignees"] = issue_data["assignees"]

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 201:
        print(f"Issue '{issue_data['title']}' created successfully.")
    else:
        print(
            f"Failed to create issue '{issue_data['title']}'. "
            f"Status Code: {response.status_code}"
        )
        print(f"Response: {response.json()}")


def main():
    """
    Entrypoint for creating GitHub issues from JSON.
    Two usage patterns:
      1) python create_github_issues.py <owner> <repo> <issues.json>
      2) python create_github_issues.py <issues.json>  # auto-detect
    """
    # Check environment variable
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print(
            "Error: GitHub PAT not found. Please set the 'GITHUB_TOKEN' environment variable."
        )
        sys.exit(1)

    # Parse command-line arguments
    # Cases:
    #   1) 3 arguments => <owner> <repo> <issues.json>
    #   2) 1 argument  => <issues.json> (use git to parse <owner>/<repo>)
    # Otherwise, show usage.
    if len(sys.argv) == 4:
        owner = sys.argv[1]
        repo = sys.argv[2]
        json_path = sys.argv[3]
    elif len(sys.argv) == 2:
        # Single argument => issues.json
        json_path = sys.argv[1]
        owner, repo = get_repo_from_git()
    else:
        print("Usage:")
        print("  1) python create_github_issues.py <owner> <repo> <issues.json>")
        print("  2) python create_github_issues.py <issues.json>  (auto-detect repo)")
        sys.exit(1)

    # Load JSON issues
    issues = load_issues(json_path)

    # Prepare request headers
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    print(f"Creating {len(issues)} issue(s) in '{owner}/{repo}'...")
    for issue in issues:
        create_issue(owner, repo, issue, headers)

    print("Done.")


if __name__ == "__main__":
    main()
