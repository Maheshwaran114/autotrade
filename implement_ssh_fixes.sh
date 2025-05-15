#!/bin/bash
# Complete SSH Connection Fix Implementation Script
# This script applies all the necessary fixes and creates a complete implementation

set -e

echo "===== IMPLEMENTING COMPLETE SSH CONNECTION FIX ====="
echo "Date/Time: $(date)"

# Check if we're in the right repository
if [ ! -d "./.git" ]; then
  echo "Error: Not in a git repository root directory."
  echo "Please run this script from the root of the autotrade repository."
  exit 1
fi

# Check if the .github/workflows directory exists
if [ ! -d "./.github/workflows" ]; then
  echo "Error: .github/workflows directory not found."
  echo "This script must be run in the autotrade repository with GitHub Actions workflows."
  exit 1
fi

# Make all scripts executable
echo "Making all scripts executable..."
find . -name "*.sh" -not -path "*/\.*" -type f -exec chmod +x {} \; || echo "Warning: Could not make some scripts executable"

# Copy the fixed workflow file to the actual workflow file
if [ -f "./.github/workflows/deploy-infra-and-app.yml.fixed" ]; then
  echo "Applying fixed workflow file..."
  cp ./.github/workflows/deploy-infra-and-app.yml.fixed ./.github/workflows/deploy-infra-and-app.yml
  echo "✅ Applied fixed workflow file"
else
  echo "❌ Fixed workflow file not found. Please run the workflow fixer script first."
  exit 1
fi

# Verify the workflow file
if [ -f "./simple_yaml_check.sh" ]; then
  echo "Verifying workflow file..."
  ./simple_yaml_check.sh ./.github/workflows/deploy-infra-and-app.yml
  echo "✅ Workflow verification completed"
else
  echo "Warning: simple_yaml_check.sh not found. Skipping workflow verification."
fi

# Check if all required scripts exist
REQUIRED_SCRIPTS=(
  "./complete_ssh_fix.sh"
  "./github_actions_ssh_enhancer.sh"
  "./validate_ssh_service.sh"
  "./fix_ssh_config.sh"
)

MISSING_SCRIPTS=0
for script in "${REQUIRED_SCRIPTS[@]}"; do
  if [ ! -f "$script" ]; then
    echo "❌ Required script not found: $script"
    MISSING_SCRIPTS=1
  else
    echo "✅ Found required script: $script"
  fi
done

if [ $MISSING_SCRIPTS -eq 1 ]; then
  echo "Warning: Some required scripts are missing. The fix may not be complete."
else
  echo "✅ All required scripts found"
fi

# Create documentation if it doesn't exist
if [ ! -f "./docs/SSH_CONNECTION_FIX.md" ]; then
  echo "Creating SSH connection fix documentation..."
  mkdir -p ./docs
  cat > ./docs/SSH_CONNECTION_FIX.md << 'EOF'
# SSH Connection Fix for GitHub Actions to DigitalOcean

## Overview

This documentation explains the fixes implemented to resolve SSH connectivity issues between GitHub Actions and DigitalOcean droplets in the BN Trading application workflow.

## Root Cause Analysis

After extensive debugging, we identified several issues:

1. **SSH Configuration Errors**: Incorrect path specification (`/null` instead of `/dev/null`) in the SSH configuration
2. **Connection Timeouts**: GitHub Actions runners experiencing network connectivity issues or timeout limits too short
3. **Droplet Power State**: The droplet might have been powered off when the workflow attempted to connect
4. **YAML Syntax Errors**: Problems with heredoc blocks in the workflow file causing SSH commands to fail
5. **SSH Service Issues**: Potential configuration issues with the SSH service on the droplet

## Fix Summary

We've implemented a comprehensive set of solutions:

1. **Fixed SSH Configuration**: Corrected UserKnownHostsFile from `/null` to `/dev/null`
2. **Enhanced Connection Parameters**: Added multiple retry strategies with increasing timeouts and progressive options
3. **Added Droplet Power Management**: Implemented automatic power-on capability if the droplet is off
4. **Created Validation Tools**: Built scripts to validate SSH service on the droplet and YAML syntax in workflows
5. **Improved Error Handling**: Added better diagnostic outputs and recovery mechanisms

## New Tools Created

1. `complete_ssh_fix.sh`: Comprehensive SSH connectivity repair tool
2. `validate_workflow_ssh.sh`: GitHub Actions workflow validator specifically for SSH commands
3. `validate_ssh_service.sh`: Server-side SSH service validator
4. `github_actions_ssh_enhancer.sh`: GitHub Actions specialized SSH connection enhancer
5. `fix_ssh_config.sh`: Tool to detect and fix SSH config issues
6. `fix_network_connectivity.sh`: Network connectivity repair tool

## Using the Complete SSH Fix

The `complete_ssh_fix.sh` script provides an all-in-one solution for diagnosing and fixing SSH connectivity issues:

```bash
./complete_ssh_fix.sh <droplet-ip> [digitalocean-token]
```

The script performs:

1. SSH configuration validation and fixes
2. Network connectivity tests
3. Progressive SSH connection attempts with different parameters
4. DigitalOcean droplet status verification and power-on if needed
5. Server-side SSH service validation
6. Comprehensive diagnostic output

## Workflow Changes

The main workflow file has been updated with:

1. **Fixed SSH Configuration**: Corrected UserKnownHostsFile paths
2. **Progressive Connection Strategy**: Added multiple connection attempts with increasing timeouts
3. **Server-side Validation**: Added the ability to deploy and run SSH service validation on the droplet
4. **Improved Diagnostics**: Enhanced error reporting and diagnostic tools

## Best Practices for SSH in GitHub Actions

1. **Always use `/dev/null` for UserKnownHostsFile**: Never use `/null` which is an invalid path
2. **Set proper timeout values**: Use ConnectTimeout, ServerAliveInterval, and ServerAliveCountMax
3. **Use proper indentation in YAML heredocs**: Use `<<-EOF` for indented heredocs in YAML files
4. **Check power state**: Always verify the droplet is powered on before attempting SSH connections
5. **Implement retry logic**: Use multiple connection attempts with increasing timeouts
6. **Verify server-side SSH configuration**: Deploy validation scripts to check SSH service status
EOF
  echo "✅ Created SSH connection fix documentation"
fi

# Create a summary of changes
echo "Creating changes summary..."
cat > ./changes_summary.txt << 'EOF'
# SSH Connection Fix Implementation Summary

## Issues Fixed

1. Fixed incorrect UserKnownHostsFile path (/null → /dev/null)
2. Added progressive retry logic with increasing timeouts and connection options
3. Added droplet power state checking and automatic power-on capability
4. Fixed YAML syntax issues with heredocs in GitHub Actions workflow
5. Added comprehensive SSH connectivity diagnostics and repair tools
6. Added server-side SSH service validation

## Files Modified

1. .github/workflows/deploy-infra-and-app.yml - Fixed workflow file with enhanced SSH connection handling
2. Various utility scripts (*.sh) - Made executable and updated with fixes

## New Files Created

1. complete_ssh_fix.sh - Comprehensive SSH connectivity repair tool
2. github_actions_ssh_enhancer.sh - GitHub Actions specialized SSH enhancer
3. validate_ssh_service.sh - Server-side SSH service validator
4. fix_ssh_config.sh - SSH configuration validator and fixer
5. simple_yaml_check.sh - Simple workflow file syntax checker
6. docs/SSH_CONNECTION_FIX.md - Documentation of the implemented fixes

## Testing Steps

1. Run the workflow to verify SSH connectivity works
2. If issues persist, use the complete_ssh_fix.sh script to diagnose and fix issues
3. Check the server-side SSH configuration with validate_ssh_service.sh
EOF

echo "✅ Created changes summary"
echo "Implementation completed successfully!"
echo "You can now commit these changes to the repository."
echo "Run the workflow to verify that the SSH connectivity issues are resolved."
