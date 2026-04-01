# ./scripts/validate-readonly-query.ps1
# Blocks SQL write operations, allows SELECT queries

# Read JSON input from stdin
$inputJson = $Input | Out-String | ConvertFrom-Json

# Extract the command field from tool_input
$command = $inputJson.tool_input.command

if (-not $command) {
    exit 0
}

# Block write operations (case-insensitive)
$blockingPattern = "\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|REPLACE|MERGE)\b"
if ($command -match $blockingPattern) {
    [Console]::Error.WriteLine("Blocked: Write operations not allowed. Use SELECT queries only.")
    exit 2
}

exit 0
