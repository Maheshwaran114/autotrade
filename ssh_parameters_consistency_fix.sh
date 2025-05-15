#!/bin/bash
# SSH_PARAMETERS_CONSISTENCY_FIX
# This script ensures consistent SSH parameters usage across all workflow files

set -e

echo "===== SSH PARAMETERS CONSISTENCY FIX ====="
echo "Date/Time: $(date)"
echo "User: $(whoami)"

WORKFLOW_FILE=".github/workflows/deploy-infra-and-app.yml"

if [ ! -f "$WORKFLOW_FILE" ]; then
    echo "❌ Workflow file not found at $WORKFLOW_FILE"
    exit 1
fi

echo "Checking workflow file for SSH parameter consistency..."

# Create backup
cp "$WORKFLOW_FILE" "${WORKFLOW_FILE}.bak"
echo "✅ Created backup at ${WORKFLOW_FILE}.bak"

# Check for inconsistent UserKnownHostsFile
if grep -q "UserKnownHostsFile /null" "$WORKFLOW_FILE"; then
    echo "❌ Found inconsistent 'UserKnownHostsFile /null' in workflow file"
    echo "Fixing the issue..."
    sed -i.fix1 's|UserKnownHostsFile /null|UserKnownHostsFile /dev/null|g' "$WORKFLOW_FILE"
    echo "✅ Fixed UserKnownHostsFile setting"
fi

# Ensure consistent SSH parameters in all commands
STANDARD_PARAMS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=30 -o ConnectTimeout=60"

# Check for simple SSH commands without parameters
if grep -q "ssh -i" "$WORKFLOW_FILE" | grep -v "StrictHostKeyChecking"; then
    echo "❌ Found SSH commands without proper parameters"
    # This is complex to fix with just sed, so we'll just report it
    echo "MANUAL FIX REQUIRED: Ensure all SSH commands use consistent parameters"
    grep -n "ssh -i" "$WORKFLOW_FILE" | grep -v "StrictHostKeyChecking"
fi

# Check for consistent heredoc syntax
if grep -q "<< EOF" "$WORKFLOW_FILE"; then
    echo "❌ Found outdated heredoc syntax << EOF in workflow"
    echo "Fixing to use <<-EOF for proper indentation..."
    sed -i.fix2 's|<< EOF|<<-EOF|g' "$WORKFLOW_FILE"
    echo "✅ Fixed heredoc syntax"
fi

echo "Review complete!"
echo "Checking for remaining issues..."

# Verify no /null paths remain
if grep -q "/null" "$WORKFLOW_FILE"; then
    echo "⚠️ WARNING: Still found /null references in the workflow:"
    grep -n "/null" "$WORKFLOW_FILE"
else
    echo "✅ No incorrect /null paths found"
fi

echo "===== FIX COMPLETE ====="
echo "Please review the workflow file manually for any remaining inconsistencies."
