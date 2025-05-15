#!/bin/bash
# Script to commit and push SSH connectivity fixes

set -e

echo "===== COMMITTING SSH CONNECTIVITY FIXES ====="
echo "Date/Time: $(date)"

# Check if we're in a git repository
if [ ! -d "./.git" ]; then
  echo "Error: Not in a git repository root directory."
  exit 1
fi

# Add all the new and modified files
echo "Adding files to git..."
git add .github/workflows/deploy-infra-and-app.yml
git add *.sh
git add docs/SSH_*.md
git add changes_summary.txt

# Create commit message
echo "Creating commit message..."
COMMIT_MESSAGE="Fix SSH connectivity issues in GitHub Actions workflow

This commit includes:
- Fixed SSH configuration with proper UserKnownHostsFile paths
- Enhanced retry logic with progressive connection options
- Droplet power state verification and automatic power-on capability
- Server-side SSH service validation scripts
- Comprehensive diagnostic and repair tools
- Workflow YAML syntax fixes

Key files:
- Fixed GitHub Actions workflow with proper SSH settings
- complete_ssh_fix.sh: All-in-one SSH repair utility
- github_actions_ssh_enhancer.sh: Enhanced SSH connectivity
- validate_ssh_service.sh: Server-side SSH diagnostics
- Documentation in docs/ directory"

# Commit the changes
echo "Committing changes..."
git commit -m "$COMMIT_MESSAGE"

# Ask for confirmation before pushing
echo ""
echo "Changes committed successfully."
echo "Do you want to push these changes to GitHub? (y/n)"
read CONFIRM

if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
  echo "Pushing changes to GitHub..."
  git push origin main
  echo "✅ Changes pushed successfully!"
else
  echo "Changes were committed but not pushed."
  echo "To push later, run: git push origin main"
fi

echo "===== SSH CONNECTIVITY FIX COMMIT COMPLETE ====="
