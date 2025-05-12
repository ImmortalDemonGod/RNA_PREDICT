import json
import os
from datetime import datetime
from pathlib import Path
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def parse_pr_file(file_path):
    """Parse a PR markdown file and extract relevant information."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract basic information
    pr_number = re.search(r'# Pull Request #(\d+)', content).group(1)
    title = re.search(r'# Pull Request #\d+: (.*?)\n', content).group(1)
    
    # Extract dates
    created = re.search(r'Created: (.*?)\n', content).group(1)
    merged = re.search(r'Merged: (.*?)\n', content).group(1)
    
    # Extract changes
    additions = int(re.search(r'Additions: (\d+)', content).group(1))
    deletions = int(re.search(r'Deletions: (\d+)', content).group(1))
    changed_files = int(re.search(r'Changed Files: (\d+)', content).group(1))
    
    # Calculate development time
    if created != 'N/A' and merged != 'N/A':
        created_date = datetime.strptime(created, '%Y-%m-%d')
        merged_date = datetime.strptime(merged, '%Y-%m-%d')
        dev_time_days = (merged_date - created_date).days
    else:
        dev_time_days = None
    
    # Categorize PR
    categories = []
    title_lower = title.lower()
    description_lower = content.lower()
    
    # Feature detection
    feature_keywords = ['feat', 'feature', 'add', 'implement', 'new']
    if any(keyword in title_lower for keyword in feature_keywords):
        categories.append('feature')
    
    # Bugfix detection
    bugfix_keywords = ['fix', 'bug', 'issue', 'resolve', 'correct']
    if any(keyword in title_lower for keyword in bugfix_keywords):
        categories.append('bugfix')
    
    # Refactor detection
    refactor_keywords = ['refactor', 'restructure', 'reorganize', 'cleanup', 'optimize']
    if any(keyword in title_lower for keyword in refactor_keywords):
        categories.append('refactor')
    
    # Documentation detection
    doc_keywords = ['doc', 'documentation', 'readme', 'guide']
    if any(keyword in title_lower for keyword in doc_keywords):
        categories.append('documentation')
    
    # If no category found, mark as 'other'
    if not categories:
        categories.append('other')
    
    return {
        'pr_number': int(pr_number),
        'title': title,
        'created': created,
        'merged': merged,
        'dev_time_days': dev_time_days,
        'additions': additions,
        'deletions': deletions,
        'changed_files': changed_files,
        'categories': categories
    }

def analyze_prs():
    """Analyze all PR markdown files and generate metrics."""
    pr_dir = Path("markdown_prs")
    if not pr_dir.exists():
        print("Error: markdown_prs directory not found!")
        return
    
    # Parse all PR files
    pr_data = []
    for file in pr_dir.glob("PR_*.md"):
        try:
            pr_info = parse_pr_file(file)
            pr_data.append(pr_info)
        except Exception as e:
            print(f"Error parsing {file}: {e}")
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(pr_data)
    
    # Basic statistics
    print("\n=== Development Time Statistics ===")
    print(f"Average development time: {df['dev_time_days'].mean():.1f} days")
    print(f"Median development time: {df['dev_time_days'].median():.1f} days")
    print(f"Min development time: {df['dev_time_days'].min():.1f} days")
    print(f"Max development time: {df['dev_time_days'].max():.1f} days")
    
    # Category distribution
    category_counts = defaultdict(int)
    for pr in pr_data:
        for category in pr['categories']:
            category_counts[category] += 1
    
    print("\n=== PR Categories ===")
    for category, count in sorted(category_counts.items()):
        print(f"{category}: {count} PRs")
    
    # Generate visualizations
    plt.figure(figsize=(15, 10))
    
    # Development time distribution
    plt.subplot(2, 2, 1)
    sns.histplot(data=df, x='dev_time_days', bins=20)
    plt.title('Development Time Distribution')
    plt.xlabel('Days')
    plt.ylabel('Number of PRs')
    
    # Category distribution
    plt.subplot(2, 2, 2)
    categories = list(category_counts.keys())
    counts = list(category_counts.values())
    plt.pie(counts, labels=categories, autopct='%1.1f%%')
    plt.title('PR Categories Distribution')
    
    # Changes over time
    plt.subplot(2, 2, 3)
    df['total_changes'] = df['additions'] + df['deletions']
    plt.scatter(df['pr_number'], df['total_changes'])
    plt.title('Total Changes per PR')
    plt.xlabel('PR Number')
    plt.ylabel('Total Changes')
    
    # Changed files distribution
    plt.subplot(2, 2, 4)
    sns.histplot(data=df, x='changed_files', bins=20)
    plt.title('Changed Files Distribution')
    plt.xlabel('Number of Files')
    plt.ylabel('Number of PRs')
    
    plt.tight_layout()
    plt.savefig('pr_analysis.png')
    print("\nAnalysis visualization saved as 'pr_analysis.png'")
    
    # Save detailed metrics to JSON
    metrics = {
        'development_time': {
            'mean': float(df['dev_time_days'].mean()),
            'median': float(df['dev_time_days'].median()),
            'min': float(df['dev_time_days'].min()),
            'max': float(df['dev_time_days'].max())
        },
        'categories': dict(category_counts),
        'changes': {
            'total_additions': int(df['additions'].sum()),
            'total_deletions': int(df['deletions'].sum()),
            'total_files_changed': int(df['changed_files'].sum()),
            'average_changes_per_pr': float(df['total_changes'].mean())
        }
    }
    
    with open('pr_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Detailed metrics saved to 'pr_metrics.json'")

if __name__ == "__main__":
    analyze_prs() 