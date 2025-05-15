#!/bin/bash
# Script to commit SSH connection fixes

echo "Committing SSH connection fixes..."
git add .github/workflows/deploy-infra-and-app.yml
git add enhanced_ssh_diagnostics.sh
git add fix_ssh_config.sh
git add check_server_ssh.sh
git add ssh_diagnostics.sh
git add docs/SSH_TROUBLESHOOTING.md

git commit -m "Fix SSH connection issues in GitHub Actions workflow

- Fixed UserKnownHostsFile path from /null to /dev/null
- Added enhanced SSH diagnostics and fix scripts
- Improved SSH connection parameters with better timeouts and retries
- Added server-side SSH configuration check script
- Created comprehensive SSH troubleshooting documentation
- Added network connectivity checks before SSH connection
- Enhanced error reporting and debugging"

echo "✅ Changes committed successfully!"
echo "To push these changes to the repository, run:"
echo "git push origin <branch-name>"
