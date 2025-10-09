# add, commit, & push to git in 1 line

# Grab commit message from arguments
param (
    [string]$CommitMessage
)

# Check if commit message was provided
if (-not $CommitMessage) {
    $CommitMessage = Read-Host "Enter commit message"
}

# Stage changes
git add .

# Commit with the provided message
git commit -m "$CommitMessage"

# Optional: push to current branch
git push
