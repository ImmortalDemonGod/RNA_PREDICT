import json
import os
import re
from datetime import datetime
from pathlib import Path

def format_date(date_str):
    """Convert ISO date string to a more readable format."""
    if not date_str:
        return "N/A"
    dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    return dt.strftime('%Y-%m-%d')  # Only keep the date, remove time

def clean_encoded_content(text):
    """Remove encoded content and unnecessary HTML comments from text."""
    if not text:
        return text
        
    # Remove base64-like encoded content in HTML comments
    text = re.sub(r'<!--\s*[A-Za-z0-9+/]{100,}\s*-->', '', text)
    
    # Remove CodeRabbit auto-generated comments
    text = re.sub(r'<!--\s*This is an auto-generated comment:.*?-->', '', text)
    text = re.sub(r'<!--\s*end of auto-generated comment:.*?-->', '', text)
    
    # Remove other unnecessary HTML comments
    text = re.sub(r'<!--\s*raw HTML omitted\s*-->', '', text)
    text = re.sub(r'<!--\s*walkthrough_start\s*-->', '', text)
    text = re.sub(r'<!--\s*walkthrough_end\s*-->', '', text)
    text = re.sub(r'<!--\s*internal state start\s*-->', '', text)
    text = re.sub(r'<!--\s*internal state end\s*-->', '', text)
    text = re.sub(r'<!--\s*finishing_touch_checkbox_start\s*-->', '', text)
    text = re.sub(r'<!--\s*finishing_touch_checkbox_end\s*-->', '', text)
    text = re.sub(r'<!--\s*tips_start\s*-->', '', text)
    text = re.sub(r'<!--\s*tips_end\s*-->', '', text)
    text = re.sub(r'<!--\s*announcements_start\s*-->', '', text)
    text = re.sub(r'<!--\s*announcements_end\s*-->', '', text)
    
    # Remove any remaining HTML comments
    text = re.sub(r'<!--.*?-->', '', text)
    
    # Remove any base64-like strings that might not be in comments
    text = re.sub(r'[A-Za-z0-9+/]{100,}', '', text)
    
    # Remove any remaining encoded content patterns
    text = re.sub(r'data:.*?;base64,.*?(?=\s|$)', '', text)
    text = re.sub(r'[A-Za-z0-9+/]{50,}={0,2}', '', text)
    
    # Clean up multiple newlines that might result from comment removal
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove any remaining HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    return text.strip()

def create_markdown_content(pr):
    """Create markdown content for a single PR."""
    # Basic PR information
    content = f"""# Pull Request #{pr['number']}: {pr['title']}

## Status
- State: {pr['state']}
- Created: {format_date(pr['createdAt'])}
- Updated: {format_date(pr['updatedAt'])}
- Closed: {format_date(pr['closedAt'])}
- Merged: {format_date(pr['mergedAt'])}

## Changes
- Additions: {pr['additions']}
- Deletions: {pr['deletions']}
- Changed Files: {pr['changedFiles']}

## Author
- Name: {pr['author'].get('name', 'N/A')}
- Login: {pr['author'].get('login', 'N/A')}
- Bot: {'Yes' if pr['author'].get('is_bot', False) else 'No'}

## Assignees
{chr(10).join(f'- {assignee["login"]}' for assignee in pr['assignees']) if pr['assignees'] else '- None'}

## Description
{clean_encoded_content(pr['body']) if pr['body'] else 'No description provided'}

## Comments
"""
    
    # Add comments
    for comment in pr['comments']:
        content += f"""
### Comment by {comment['author']['login']}
- Created: {format_date(comment['createdAt'])}
- Author Association: {comment['authorAssociation']}

{clean_encoded_content(comment['body'])}

---
"""
    
    return content

def main():
    # Create output directory
    output_dir = Path("markdown_prs")
    output_dir.mkdir(exist_ok=True)
    
    # Read JSON file
    json_path = Path("pull_requests_full_pretty.json")
    with open(json_path, 'r') as f:
        prs = json.load(f)
    
    # Convert each PR to markdown
    for pr in prs:
        # Create filename from PR number and title
        safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in pr['title'])
        filename = f"PR_{pr['number']}_{safe_title}.md"
        filepath = output_dir / filename
        
        # Create markdown content
        content = create_markdown_content(pr)
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Created: {filepath}")

if __name__ == "__main__":
    main() 