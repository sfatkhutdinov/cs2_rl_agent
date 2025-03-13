# Link Checker for Markdown Files
# This script scans all markdown files in the analysis_log directory,
# identifies links, and verifies if they point to existing files.

# Function to extract links from markdown content
function Extract-Links {
    param (
        [string]$Content
    )
    
    # Match markdown links in the format [text](link)
    $linkPattern = '\[(?<text>[^\]]*)\]\((?<link>[^)]+)\)'
    $matches = [regex]::Matches($Content, $linkPattern)
    
    $links = @()
    foreach ($match in $matches) {
        $link = $match.Groups['link'].Value
        # Exclude web links (http, https)
        if (-not ($link -match '^https?://')) {
            $links += $link
        }
    }
    
    return $links
}

# Function to validate a link
function Validate-Link {
    param (
        [string]$BaseDir,
        [string]$FilePath,
        [string]$Link
    )
    
    # Get directory of the current file
    $fileDir = Split-Path -Parent $FilePath
    
    # Handle relative paths
    if ($Link.StartsWith("../")) {
        # Going up one directory
        $targetPath = Join-Path -Path (Split-Path -Parent $fileDir) -ChildPath $Link.Substring(3)
    }
    elseif (-not $Link.StartsWith("/")) {
        # Same directory
        $targetPath = Join-Path -Path $fileDir -ChildPath $Link
    }
    else {
        # Absolute path (within the repository)
        $targetPath = Join-Path -Path $BaseDir -ChildPath $Link.TrimStart("/")
    }
    
    # Normalize path
    $targetPath = $targetPath -replace "/", "\"
    
    # Check if file exists
    $exists = Test-Path -Path $targetPath
    
    return @{
        SourceFile = $FilePath
        Link = $Link
        TargetPath = $targetPath
        Exists = $exists
    }
}

# Main script
$analysisLogDir = (Get-Location).Path
$markdownFiles = Get-ChildItem -Path $analysisLogDir -Filter "*.md" -Recurse

$results = @()

Write-Host "Scanning markdown files for links..."
foreach ($file in $markdownFiles) {
    $content = Get-Content -Path $file.FullName -Raw
    $links = Extract-Links -Content $content
    
    foreach ($link in $links) {
        $validation = Validate-Link -BaseDir $analysisLogDir -FilePath $file.FullName -Link $link
        $results += $validation
        
        if (-not $validation.Exists) {
            Write-Host "Broken link found:"
            Write-Host "  Source: $($validation.SourceFile)"
            Write-Host "  Link: $($validation.Link)"
            Write-Host "  Target: $($validation.TargetPath)"
            Write-Host ""
        }
    }
}

# Summary
$totalLinks = $results.Count
$brokenLinks = ($results | Where-Object { -not $_.Exists }).Count

Write-Host "Link checking complete!"
Write-Host "Total links checked: $totalLinks"
Write-Host "Broken links found: $brokenLinks"

# Export results to CSV
$results | Export-Csv -Path "link_check_results.csv" -NoTypeInformation

Write-Host "Results exported to link_check_results.csv" 