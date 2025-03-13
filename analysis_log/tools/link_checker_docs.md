# Link Checker Documentation

**Tags:** #tools #maintenance

## Overview
The Link Checker is a PowerShell script designed to validate internal links between markdown files in the CS2 reinforcement learning agent analysis documentation. It helps maintain the integrity of the backlinked document structure by identifying broken links.

## Features
- Scans all markdown files recursively
- Identifies broken internal links
- Handles relative and absolute paths within the repository
- Excludes external web links (http/https)
- Generates a comprehensive report of all links and their status

## Usage

### Prerequisites
- Windows PowerShell 5.1 or higher
- Execute permission for PowerShell scripts

### Running the Script
1. Open PowerShell in the `analysis_log` directory
2. Execute the script:
   ```powershell
   .\tools\link_checker.ps1
   ```

### Understanding the Results
The script outputs:
- Real-time feedback on broken links with their source and target
- A summary of total links checked and broken links found
- A CSV file (`link_check_results.csv`) with detailed results

## How It Works

### Link Extraction
The script extracts links from markdown files using regular expressions, looking for the standard markdown link pattern: `[text](link)`.

### Link Validation
For each extracted link, the script:
1. Determines the absolute file path based on the link type (relative/absolute)
2. Verifies if the target file exists
3. Records the result

### Handling Different Link Types
- **Same directory links**: `[Link](file.md)`
- **Subdirectory links**: `[Link](subdirectory/file.md)`
- **Parent directory links**: `[Link](../directory/file.md)`
- **Root-relative links**: `[Link](/path/from/root.md)`

## Maintenance
To improve or expand the Link Checker:

1. Add support for additional link types
   ```powershell
   # Add new link pattern detection in Extract-Links function
   ```

2. Enhance reporting capabilities
   ```powershell
   # Modify the export section at the end of the script
   ```

3. Add automatic fixing of simple issues
   ```powershell
   # Create a new function for fixing common link problems
   ```

## Integration with Documentation Workflow
Consider running this link checker:
- Before committing major document changes
- As part of regular documentation maintenance
- When reorganizing the document structure

## Related Tools
- [Document Tagging System](../document_tags.md) - For tag-based organization
- [Component Relationship Visualization](../visualization/component_relationships.md) - For structural visualization 

Examples of link formats:
```markdown
[Link Text](file.md)
[Link Text](subdirectory/file.md)
[Link Text](../parent-directory/file.md)
[Link Text](/absolute/path/file.md)
``` 