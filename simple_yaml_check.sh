#!/bin/bash
# Simple YAML syntax checker for GitHub Actions workflows

WORKFLOW_FILE="$1"

if [ -z "$WORKFLOW_FILE" ]; then
  echo "Error: No workflow file specified."
  echo "Usage: $0 <path-to-workflow-file.yml>"
  exit 1
fi

if [ ! -f "$WORKFLOW_FILE" ]; then
  echo "Error: File '$WORKFLOW_FILE' does not exist."
  exit 1
fi

echo "Checking for common SSH heredoc issues in $WORKFLOW_FILE..."

# Check for proper indentation in heredocs
HEREDOC_ISSUES=$(grep -n "<<EOF" "$WORKFLOW_FILE" | grep -v "<<-EOF")
if [ -n "$HEREDOC_ISSUES" ]; then
  echo "WARNING: Found standard heredocs (<<EOF) which might cause indentation issues in YAML:"
  echo "$HEREDOC_ISSUES"
  echo "Consider using <<-EOF for indented heredocs in YAML."
fi

# Check for UserKnownHostsFile with /null path
NULL_PATH_ISSUES=$(grep -n "UserKnownHostsFile /null" "$WORKFLOW_FILE")
if [ -n "$NULL_PATH_ISSUES" ]; then
  echo "ERROR: Found incorrect UserKnownHostsFile paths:"
  echo "$NULL_PATH_ISSUES"
  echo "Should be /dev/null instead of /null."
fi

# Check for proper SSH connection options
if grep -q "ssh " "$WORKFLOW_FILE" && ! grep -q "StrictHostKeyChecking=no" "$WORKFLOW_FILE"; then
  echo "WARNING: SSH commands found, but no StrictHostKeyChecking=no option."
  echo "This may cause the SSH connection to hang while waiting for confirmation."
fi

if grep -q "ssh " "$WORKFLOW_FILE" && ! grep -q "ConnectTimeout" "$WORKFLOW_FILE"; then
  echo "WARNING: SSH commands found, but no ConnectTimeout option."
  echo "This may cause the SSH connection to hang indefinitely on network issues."
fi

echo "Check completed!"
exit 0
