import re
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

LOG_FILE = 'all_commits.log'

# Regex to parse commit lines
COMMIT_RE = re.compile(r'^(?P<hash>[a-f0-9]+) - (?P<author>.*?), (?P<time>.+?) : (?P<msg>.+)$')

# Regex to extract type (conventional commit)
TYPE_RE = re.compile(r'(?:\S+\s+)?(?P<type>\w+)(?:\([^)]+\))?:', re.UNICODE)

def parse_commit_line(line):
    m = COMMIT_RE.match(line)
    if not m:
        return None
    d = m.groupdict()
    msg = d['msg']
    
    # Handle escaped newlines and split message into lines
    messages = [m.strip() for m in msg.replace('\\n', '\n').split('\n') if m.strip()]
    types = []
    
    for submsg in messages:
        type_match = TYPE_RE.match(submsg)
        if type_match:
            ctype = type_match.group('type').lower()
        elif submsg.lower().startswith('merge'):
            ctype = 'merge'
        else:
            ctype = 'other'
        types.append(ctype)
    
    # If no types found in any message, use 'other'
    if not types:
        types = ['other']
    
    return {
        'hash': d['hash'],
        'author': d['author'],
        'time_ago': d['time'],
        'types': types,  # Now returns a list of types
        'messages': messages  # Store all messages
    }

def analyze_log(log_path):
    commits = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            commit = parse_commit_line(line)
            if commit:
                commits.append(commit)
    
    # Flatten the types for analysis
    all_types = []
    for commit in commits:
        all_types.extend(commit['types'])
    
    # Create DataFrame with flattened types
    df = pd.DataFrame({
        'type': all_types,
        'author': [commit['author'] for commit in commits for _ in commit['types']]
    })
    
    # Save analysis output to a file
    with open('commit_analysis_output.txt', 'w') as f:
        f.write(f"Total commits: {len(commits)}\n")
        f.write(f"Total commit messages: {len(all_types)}\n\n")
        f.write("=== Commit Types (including multi-message commits) ===\n")
        f.write(df['type'].value_counts().to_string())
        f.write("\n\n=== Top Authors ===\n")
        f.write(df['author'].value_counts().to_string())
    
    print(f"Total commits: {len(commits)}")
    print(f"Total commit messages: {len(all_types)}")
    print("\n=== Commit Types (including multi-message commits) ===")
    print(df['type'].value_counts())
    print("\n=== Top Authors ===")
    print(df['author'].value_counts())
    
    # Plot commit type distribution
    plt.figure(figsize=(10,5))
    sns.countplot(data=df, x='type', order=df['type'].value_counts().index)
    plt.title('Commit Type Distribution (Including Multi-Message Commits)')
    plt.xlabel('Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('commit_type_distribution.png')
    print("\nPlot saved as commit_type_distribution.png")
    
    # Additional analysis: commits with multiple types
    multi_type_commits = [c for c in commits if len(c['types']) > 1]
    if multi_type_commits:
        print(f"\n=== Commits with Multiple Types: {len(multi_type_commits)} ===")
        for commit in multi_type_commits[:5]:  # Show first 5 examples
            print(f"\nCommit {commit['hash']}:")
            print(f"Types: {', '.join(commit['types'])}")
            print("Messages:")
            for msg in commit['messages']:
                print(f"  - {msg}")
    
    # Time-based distribution plot
    df['time_ago'] = [commit['time_ago'] for commit in commits for _ in commit['types']]
    # Use the exact ISO format for parsing
    df['time_ago'] = pd.to_datetime(df['time_ago'], format='%Y-%m-%d %H:%M:%S %z', errors='coerce')
    df = df.dropna(subset=['time_ago'])
    df['month'] = df['time_ago'].dt.to_period('M')
    
    plt.figure(figsize=(12,6))
    sns.countplot(data=df, x='month', hue='type', order=df['month'].value_counts().index)
    plt.title('Commit Type Distribution Over Time')
    plt.xlabel('Month')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('commit_type_time_distribution.png')
    print("\nTime-based distribution plot saved as commit_type_time_distribution.png")

if __name__ == "__main__":
    analyze_log(LOG_FILE) 