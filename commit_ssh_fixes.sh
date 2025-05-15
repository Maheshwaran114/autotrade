#!/bin/bash
# Script to commit SSH key path fixes

echo "Committing SSH configuration fixes"

# Check if we're in the right directory
if [ ! -d ".git" ]; then
  echo "Error: Not in a git repository. Please run this script from the repository root."
  exit 1
fi

# Add the modified files
git add .github/workflows/deploy-infra-and-app.yml
git add ssh_diagnostics.sh
git add test_ssh_key_format.sh
git add test_ssh_connection.sh
git add SSH_TESTING_INSTRUCTIONS.md
git add commit_ssh_fixes.sh

# Commit the changes
git commit -m "Fix SSH key path references from id_rsa to id_bn_trading"

echo "✅ Changes committed successfully!"
echo
echo "To push these changes to the repository, run:"
echo "git push origin <your-branch-name>"
echo
echo "If you want to create a new branch first, run:"
echo "git checkout -b fix/ssh-key-path"
echo "git push origin fix/ssh-key-path"
