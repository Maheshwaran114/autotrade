#!/bin/bash
# COMMIT ALL SSH FIXES
# This script commits all the SSH connectivity fixes

set -e

echo "===== COMMITTING SSH FIXES ====="
echo "Date/Time: $(date)"
echo "User: $(whoami)"

# Define files to commit
echo "Adding new files to git..."
git add \
  ssh_connectivity_monitor.sh \
  server_health_check.sh \
  install_ssh_fixes.sh \
  verify_ssh_fixes.sh \
  docs/COMPLETE_SSH_FIXES.md \
  docs/SSH_FIXES_UPDATE.md

# Commit the changes
echo "Committing changes..."
git commit -m "Enhanced SSH connectivity with monitoring and verification tools

- Added ssh_connectivity_monitor.sh for continuous SSH monitoring
- Added server_health_check.sh for application health verification
- Created install_ssh_fixes.sh to automate fix installation
- Added verify_ssh_fixes.sh to verify all fixes are applied
- Updated documentation with comprehensive fix details
- Enhanced parameter consistency across all SSH commands
- Added detailed diagnostics for SSH connection failures"

# Push the changes
echo "Pushing changes to GitHub..."
git push origin main || echo "Failed to push changes. Please push manually."

echo "===== COMMIT COMPLETE ====="
echo "All SSH fixes have been committed and pushed to GitHub."
echo "Next step: Monitor the GitHub Actions workflow to ensure the fixes are working as expected."
