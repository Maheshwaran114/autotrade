#!/bin/bash
# Script to safely push changes to GitHub
# This will exclude sensitive files

# Configuration
REPO_DIR="/Users/dharanyamahesh/Documents/GitHub/autotrade"
COMMIT_MESSAGE="Improve GitHub Actions workflow and SSH connectivity"

# Check prerequisites
if ! command -v git &> /dev/null; then
  echo "Error: git is not installed or not in PATH"
  exit 1
fi

# Go to repository directory
cd "$REPO_DIR" || { echo "Error: Could not change to repository directory"; exit 1; }

# Check if this is a git repository
if ! git rev-parse --is-inside-work-tree &> /dev/null; then
  echo "Error: $REPO_DIR is not a git repository"
  exit 1
fi

# Check if there are sensitive files we want to exclude
echo "Checking for sensitive files to exclude..."
SENSITIVE_FILES=(
  "github_secrets.txt"
  "ssh_key_id.txt"
  "test_config.sh"
  "*.pem"
  "*.key"
  "*.env"
  "config/credentials.yml"
  "terraform.tfvars"
)

# Create/update .gitignore to exclude sensitive files
for file in "${SENSITIVE_FILES[@]}"; do
  if ! grep -q "^$file$" "$REPO_DIR/.gitignore" 2>/dev/null; then
    echo "$file" >> "$REPO_DIR/.gitignore"
    echo "Added $file to .gitignore"
  fi
done

# Check for sensitive patterns in all files that would be added
echo "Checking for potential sensitive data in files..."
git ls-files --others --exclude-standard | xargs grep -l -E "(password|secret|token|key|credential|api_key|access_token|dop_v1_)" 2>/dev/null > /tmp/sensitive_checks.txt
git ls-files --modified | xargs grep -l -E "(password|secret|token|key|credential|api_key|access_token|dop_v1_)" 2>/dev/null >> /tmp/sensitive_checks.txt

if [ -s /tmp/sensitive_checks.txt ]; then
  echo "⚠️  WARNING: The following files may contain sensitive information:"
  cat /tmp/sensitive_checks.txt
  echo
  echo "Please review these files carefully before committing."
  echo "Would you like to continue? (y/n)"
  read -r SENSITIVE_CONFIRM
  
  if [ "$SENSITIVE_CONFIRM" != "y" ]; then
    echo "Operation cancelled. Please review your files."
    rm /tmp/sensitive_checks.txt
    exit 0
  fi
fi

rm -f /tmp/sensitive_checks.txt

# List files that will be committed
echo
echo "Files that will be added to the commit:"
git add --dry-run .

echo
echo "Would you like to see a detailed diff of changes? (y/n)"
read -r SHOW_DIFF

if [ "$SHOW_DIFF" == "y" ]; then
  git diff --cached
fi

echo
echo "Would you like to continue with these changes? (y/n)"
read -r CONFIRM

if [ "$CONFIRM" != "y" ]; then
  echo "Operation cancelled."
  exit 0
fi

# Add and commit changes
git add .
git status

echo
echo "Ready to commit with message: \"$COMMIT_MESSAGE\""
echo "Would you like to change the commit message? (y/n)"
read -r CHANGE_MSG

if [ "$CHANGE_MSG" == "y" ]; then
  echo "Enter new commit message:"
  read -r NEW_MESSAGE
  COMMIT_MESSAGE="$NEW_MESSAGE"
fi

git commit -m "$COMMIT_MESSAGE"

# Check branch status
echo "Checking branch status..."
git fetch
LOCAL=$(git rev-parse @)
REMOTE=$(git rev-parse @{u} 2>/dev/null)
BASE=$(git merge-base @ @{u} 2>/dev/null)

if [ -z "$REMOTE" ]; then
  echo "No upstream branch found. This might be a new branch."
elif [ "$LOCAL" = "$REMOTE" ]; then
  echo "Branch is up to date with remote."
elif [ "$LOCAL" = "$BASE" ]; then
  echo "⚠️  WARNING: Your branch is behind the remote branch. Consider pulling changes first."
  echo "Would you like to pull changes before pushing? (y/n)"
  read -r PULL_FIRST
  
  if [ "$PULL_FIRST" == "y" ]; then
    git pull
  fi
elif [ "$REMOTE" = "$BASE" ]; then
  echo "Your branch is ahead of the remote branch."
else
  echo "⚠️  WARNING: Your branch has diverged from the remote branch."
  echo "You might want to merge or rebase before pushing."
  echo "Would you like to continue anyway? (y/n)"
  read -r DIVERGED_CONFIRM
  
  if [ "$DIVERGED_CONFIRM" != "y" ]; then
    echo "Push cancelled. Your changes are committed locally but not pushed."
    exit 0
  fi
fi

# Push changes
echo
echo "Ready to push changes to GitHub. Continue? (y/n)"
read -r PUSH_CONFIRM

if [ "$PUSH_CONFIRM" == "y" ]; then
  git push
  echo "Changes pushed to GitHub!"
else
  echo "Push cancelled. Your changes are committed locally but not pushed."
fi