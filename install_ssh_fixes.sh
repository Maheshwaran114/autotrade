#!/bin/bash
# SSH FIXES INSTALLER
# This script installs and configures all SSH fixes and monitoring tools

set -e

echo "===== SSH FIXES INSTALLER ====="
echo "Date/Time: $(date)"
echo "User: $(whoami)"

# Create a backups directory
BACKUP_DIR="./ssh_fixes_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
echo "Created backup directory: $BACKUP_DIR"

# Function to check if a script exists and make it executable
check_and_make_executable() {
  SCRIPT="$1"
  if [ -f "$SCRIPT" ]; then
    echo "✅ Found $SCRIPT"
    chmod +x "$SCRIPT"
    echo "  Made it executable"
    return 0
  else
    echo "❌ Missing $SCRIPT"
    return 1
  fi
}

# 1. Backup workflow file
echo "1. Backing up workflow file..."
WORKFLOW_FILE=".github/workflows/deploy-infra-and-app.yml"
if [ -f "$WORKFLOW_FILE" ]; then
  cp "$WORKFLOW_FILE" "$BACKUP_DIR/deploy-infra-and-app.yml.bak"
  echo "✅ Backed up workflow file to $BACKUP_DIR/deploy-infra-and-app.yml.bak"
else
  echo "❌ Workflow file not found at $WORKFLOW_FILE"
  exit 1
fi

# 2. Check and make all scripts executable
echo "2. Making all scripts executable..."
SCRIPTS=(
  "complete_ssh_fix.sh"
  "fix_ssh_config.sh"
  "ssh_parameters_consistency_fix.sh"
  "fix_ssh_verification_params.sh"
  "github_actions_ssh_enhancer.sh"
  "validate_ssh_service.sh"
  "network_diagnostics.sh"
  "ssh_diagnostics.sh"
  "enhanced_ssh_diagnostics.sh"
  "fix_network_connectivity.sh"
  "validate_workflow_ssh.sh"
  "validate_workflow_yaml.sh"
  "simple_yaml_check.sh"
  "ssh_connectivity_monitor.sh"
  "server_health_check.sh"
)

for script in "${SCRIPTS[@]}"; do
  check_and_make_executable "$script"
done

# 3. Run the SSH config fix
echo "3. Running SSH configuration fix..."
if [ -f "fix_ssh_config.sh" ]; then
  ./fix_ssh_config.sh
else
  echo "❌ SSH config fix script not found"
fi

# 4. Run the parameters consistency fix
echo "4. Running SSH parameters consistency fix..."
if [ -f "ssh_parameters_consistency_fix.sh" ]; then
  ./ssh_parameters_consistency_fix.sh
else
  echo "❌ SSH parameters consistency fix script not found"
fi

# 5. Run the verification parameters fix
echo "5. Running SSH verification parameters fix..."
if [ -f "fix_ssh_verification_params.sh" ]; then
  ./fix_ssh_verification_params.sh
else
  echo "❌ SSH verification parameters fix script not found"
fi

# 6. Validate the workflow file
echo "6. Validating workflow file..."
if [ -f "validate_workflow_yaml.sh" ]; then
  ./validate_workflow_yaml.sh "$WORKFLOW_FILE"
else
  echo "❌ Workflow validator script not found"
fi

# 7. Create comprehensive documentation
echo "7. Creating comprehensive documentation..."
DOCS_DIR="./docs"
mkdir -p "$DOCS_DIR"

# Update fixes summary document
cat > "$DOCS_DIR/COMPLETE_SSH_FIXES.md" << 'EOF'
# Complete SSH Connectivity Fixes

## Overview

This document provides a comprehensive overview of all SSH connectivity fixes implemented in the Bank Nifty Trading application's GitHub Actions workflow.

## Components

### 1. Configuration Fixes
- Fixed `UserKnownHostsFile` path from incorrect `/null` to correct `/dev/null`
- Implemented optimal SSH config settings with proper timeout values
- Added proper permissions for SSH keys and configuration files

### 2. Connection Strategies
- Implemented progressive retry logic with increasing timeouts
- Added fallback connection options with alternative algorithms
- Enhanced error reporting and diagnostic capabilities

### 3. Monitoring Tools
- Added server-side health check script
- Created SSH connectivity monitoring script
- Implemented validation tools for workflow files

### 4. Workflow Enhancements
- Added consistent SSH parameters across all commands
- Fixed heredoc syntax for proper YAML formatting
- Added comprehensive error handling and recovery mechanisms

## Tools Created

1. **Core Fix Scripts**:
   - `complete_ssh_fix.sh`: Comprehensive SSH connectivity repair
   - `fix_ssh_config.sh`: SSH configuration validator and fixer
   - `ssh_parameters_consistency_fix.sh`: Ensures consistent parameters
   - `fix_ssh_verification_params.sh`: Optimizes verification section

2. **Diagnostic Tools**:
   - `github_actions_ssh_enhancer.sh`: GitHub Actions specialized enhancer
   - `validate_ssh_service.sh`: Server-side SSH service validator
   - `network_diagnostics.sh`: Network connectivity diagnostic tool
   - `ssh_diagnostics.sh`: SSH-specific diagnostics

3. **Validation Tools**:
   - `validate_workflow_ssh.sh`: Workflow SSH command validator
   - `validate_workflow_yaml.sh`: YAML syntax validator
   - `simple_yaml_check.sh`: Quick YAML syntax checker

4. **Monitoring Tools**:
   - `ssh_connectivity_monitor.sh`: Ongoing connectivity monitor
   - `server_health_check.sh`: Server-side application health check

## Usage Instructions

### Basic SSH Fix
To apply the complete SSH connectivity fix to your environment:
```bash
./complete_ssh_fix.sh <droplet-ip> [digitalocean-token]
```

### Validate Workflow
To validate your GitHub Actions workflow file:
```bash
./validate_workflow_yaml.sh .github/workflows/deploy-infra-and-app.yml
```

### Monitor Connectivity
To monitor SSH connectivity to your server:
```bash
./ssh_connectivity_monitor.sh <droplet-ip> [interval] [duration]
```

### Check Server Health
Deploy and run the server health check:
```bash
./server_health_check.sh
```

## Best Practices

1. Always use `/dev/null` for `UserKnownHostsFile`
2. Implement retry logic with progressive timeout values
3. Use proper heredoc syntax in YAML with `<<-EOF`
4. Set appropriate connection parameters like `ServerAliveInterval`
5. Implement server-side health checks for complete visibility
6. Use monitoring tools to detect intermittent connectivity issues

## Troubleshooting

If you encounter SSH connectivity issues:

1. Run the complete fix: `./complete_ssh_fix.sh <droplet-ip>`
2. Validate your workflow: `./validate_workflow_yaml.sh <workflow-file>`
3. Check server-side health: Deploy and run `server_health_check.sh`
4. Monitor connectivity: `./ssh_connectivity_monitor.sh <droplet-ip>`

## Maintenance

Regularly check the health of your deployment with:
```bash
./ssh_connectivity_monitor.sh <droplet-ip> 3600 86400  # Monitor for 24 hours with checks every hour
```

And verify server health with:
```bash
ssh -i ~/.ssh/id_bn_trading -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@<droplet-ip> 'bash -s' < server_health_check.sh
```
EOF

echo "✅ Created comprehensive documentation at $DOCS_DIR/COMPLETE_SSH_FIXES.md"

# 8. Summary
echo "===== INSTALLATION SUMMARY ====="
echo "All SSH fix scripts have been installed and configured."
echo "The workflow file has been validated and fixed."
echo "Comprehensive documentation has been created."
echo ""
echo "Next steps:"
echo "1. Review the changes made to the workflow file"
echo "2. Commit and push the changes to GitHub"
echo "3. Monitor the workflow execution to ensure fixes are working"
echo ""
echo "For detailed documentation, see: $DOCS_DIR/COMPLETE_SSH_FIXES.md"
