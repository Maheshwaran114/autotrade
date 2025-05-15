#!/bin/bash
# Ensure consistent SSH parameters in the verification section
# This script specifically targets the verification section in the workflow

set -e

echo "===== SSH VERIFICATION PARAMETERS FIX ====="
echo "Date/Time: $(date)"
echo "User: $(whoami)"

WORKFLOW_FILE=".github/workflows/deploy-infra-and-app.yml"

if [ ! -f "$WORKFLOW_FILE" ]; then
    echo "❌ Workflow file not found at $WORKFLOW_FILE"
    exit 1
fi

# Create backup
cp "$WORKFLOW_FILE" "${WORKFLOW_FILE}.verifyfix.bak"
echo "✅ Created backup at ${WORKFLOW_FILE}.verifyfix.bak"

# Find and fix verification SSH command
echo "Searching for verification SSH command..."

# Extract the verification section and check if it uses consistent parameters
VERIFY_SECTION=$(grep -A 20 "Method 3: Checking container status via SSH" "$WORKFLOW_FILE")

echo "Current verification SSH command:"
echo "$VERIFY_SECTION" | grep "ssh" | head -1

# Create optimized SSH parameters for verification
BEST_PARAMS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=30 -o ConnectTimeout=60 -o ServerAliveCountMax=10 -o TCPKeepAlive=yes"

# Use the optimal pattern for verbosity level
if [[ "$VERIFY_SECTION" == *"-vvv"* ]]; then
    # Keep verbose logging if it's already present
    echo "Keeping verbose logging in verification command"
    BEST_PARAMS="-vvv $BEST_PARAMS"
fi

# First, find lines with ssh command for verification
LINE_NUM=$(grep -n "Method 3: Checking container status via SSH" "$WORKFLOW_FILE" | cut -d: -f1)
if [ -n "$LINE_NUM" ]; then
    # Look for the ssh command in the next 5 lines
    SSH_LINE_NUM=$(tail -n +$LINE_NUM "$WORKFLOW_FILE" | grep -n "ssh" | head -1 | cut -d: -f1)
    if [ -n "$SSH_LINE_NUM" ]; then
        # Calculate actual line number
        ACTUAL_LINE_NUM=$((LINE_NUM + SSH_LINE_NUM - 1))
        
        # Extract current line
        CURRENT_LINE=$(sed -n "${ACTUAL_LINE_NUM}p" "$WORKFLOW_FILE")
        echo "Found SSH verification command at line $ACTUAL_LINE_NUM:"
        echo "$CURRENT_LINE"
        
        # Check if it needs fixing
        if [[ "$CURRENT_LINE" == *"UserKnownHostsFile /null"* ]] || [[ "$CURRENT_LINE" != *"ServerAliveInterval"* ]]; then
            echo "SSH verification command needs fixing"
            
            # Create the fixed line, preserving indentation and structure
            INDENT=$(echo "$CURRENT_LINE" | sed 's/^\(\s*\).*/\1/')
            
            # Extract the key file part
            KEY_PART=$(echo "$CURRENT_LINE" | grep -o "\-i [^ ]*")
            
            # Extract the user@host part
            USER_HOST=$(echo "$CURRENT_LINE" | grep -o "root@[^ ]*")
            
            # Build the fixed line
            FIXED_LINE="${INDENT}ssh ${BEST_PARAMS} ${KEY_PART} ${USER_HOST}"
            
            # Replace the line in the file
            sed -i.verifyfix "${ACTUAL_LINE_NUM}s|.*|${FIXED_LINE}|" "$WORKFLOW_FILE"
            
            echo "✅ Fixed SSH verification command:"
            echo "$FIXED_LINE"
        else
            echo "✅ SSH verification command already using proper parameters"
        fi
    else
        echo "❌ Couldn't find SSH command in verification section"
    fi
else
    echo "❌ Couldn't find verification section in workflow file"
fi

echo "===== VERIFICATION PARAMETERS FIX COMPLETE ====="
