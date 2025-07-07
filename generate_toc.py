#!/usr/bin/env python3
"""
Script to automatically generate Table of Contents for Markdown files
based on heading markers (#, ##, ###, etc.)
"""

import re
import sys
from pathlib import Path

def extract_headings(content):
    """Extract all headings from markdown content"""
    headings = []
    lines = content.split('\n')
    
    for line_num, line in enumerate(lines, 1):
        # Match markdown headings (# ## ### etc.)
        heading_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
        if heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            
            # Skip the main title (# level) and Table of Contents heading
            if level == 1 or title.lower() == "table of contents":
                continue
                
            headings.append({
                'level': level,
                'title': title,
                'line': line_num,
                'anchor': create_anchor(title)
            })
    
    return headings

def create_anchor(title):
    """Create GitHub-style anchor link from heading title"""
    # Convert to lowercase
    anchor = title.lower()
    
    # Remove special characters and replace spaces with hyphens
    anchor = re.sub(r'[^\w\s-]', '', anchor)
    anchor = re.sub(r'\s+', '-', anchor)
    anchor = re.sub(r'-+', '-', anchor)
    anchor = anchor.strip('-')
    
    return anchor

def generate_toc(headings):
    """Generate table of contents from headings list"""
    toc_lines = []
    
    for heading in headings:
        # Calculate indentation based on heading level
        # Level 2 (##) = no indent, Level 3 (###) = 2 spaces, etc.
        indent = '  ' * (heading['level'] - 2) if heading['level'] > 2 else ''
        
        # Create the TOC entry
        toc_entry = f"{indent}- [{heading['title']}](#{heading['anchor']})"
        toc_lines.append(toc_entry)
    
    return '\n'.join(toc_lines)

def find_toc_section(content):
    """Find the Table of Contents section in the content"""
    lines = content.split('\n')
    toc_start = None
    toc_end = None
    
    for i, line in enumerate(lines):
        if re.match(r'^##\s+Table of Contents\s*$', line.strip()):
            toc_start = i
            # Find the end of TOC section (next ## heading or end of file)
            for j in range(i + 1, len(lines)):
                if re.match(r'^##\s+', lines[j].strip()):
                    toc_end = j
                    break
            if toc_end is None:
                toc_end = len(lines)
            break
    
    return toc_start, toc_end

def update_toc_in_content(content, new_toc):
    """Update the Table of Contents section in the content"""
    lines = content.split('\n')
    toc_start, toc_end = find_toc_section(content)
    
    if toc_start is None:
        print("Warning: Table of Contents section not found!")
        return content
    
    # Replace the TOC section
    new_lines = (
        lines[:toc_start + 1] +  # Keep everything up to and including "## Table of Contents"
        [''] +  # Empty line after heading
        new_toc.split('\n') +  # New TOC content
        [''] +  # Empty line before next section
        lines[toc_end:]  # Keep everything after TOC
    )
    
    return '\n'.join(new_lines)

def main():
    """Main function to generate and update TOC"""
    # File path
    file_path = Path("Transformers_Library_Development_Guide.md")
    
    if not file_path.exists():
        print(f"Error: File {file_path} not found!")
        sys.exit(1)
    
    # Read the file
    print(f"Reading {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract headings
    print("Extracting headings...")
    headings = extract_headings(content)
    
    print(f"Found {len(headings)} headings:")
    for heading in headings:
        indent = '  ' * (heading['level'] - 2)
        print(f"  {indent}- {heading['title']} (Level {heading['level']})")
    
    # Generate new TOC
    print("\nGenerating Table of Contents...")
    new_toc = generate_toc(headings)
    
    print("Generated TOC:")
    print("=" * 50)
    print(new_toc)
    print("=" * 50)
    
    # Update content
    print("\nUpdating document...")
    updated_content = update_toc_in_content(content, new_toc)
    
    # Write back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print(f"✅ Successfully updated Table of Contents in {file_path}")
    print(f"📊 TOC contains {len(headings)} entries")

if __name__ == "__main__":
    main()
