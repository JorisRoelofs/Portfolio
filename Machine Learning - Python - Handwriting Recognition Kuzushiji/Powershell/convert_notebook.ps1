# Converts Jupyter Notebook to README.md

# Set notebook and output file
Write-Host "Converting $notebook to markdown..."
$notebook = "Handwriting Recognition Kuzushiji.ipynb"
$output   = "README.md"

# Stop script on any error
$ErrorActionPreference = "Stop"

# Convert
jupyter nbconvert --to markdown "$notebook" --output "$output"
Write-Host "Conversion complete."