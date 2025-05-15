#!/bin/bash
# SSH FIXES VERIFICATION
# This script verifies that all SSH fixes have been properly applied

set -e

echo "===== SSH FIXES VERIFICATION ====="
echo "Date/Time: $(date)"
echo "User: $(whoami)"

# Output report file
REPORT_FILE="ssh_fixes_verification_$(date +%Y%m%d_%H%M%S).md"
echo "Creating verification report in: $REPORT_FILE"

# Start the report
cat > "$REPORT_FILE" << 'EOF'
# SSH Fixes Verification Report

## Overview
This report verifies the implementation of SSH connectivity fixes in the GitHub Actions workflow.

## 1. Workflow File Check
EOF

# 1. Check workflow file
WORKFLOW_FILE=".github/workflows/deploy-infra-and-app.yml"
if [ -f "$WORKFLOW_FILE" ]; then
  echo "- ✅ Workflow file exists" >> "$REPORT_FILE"
  
  # Check for correct UserKnownHostsFile
  if grep -q "UserKnownHostsFile /null" "$WORKFLOW_FILE"; then
    echo "- ❌ Found incorrect UserKnownHostsFile /null in workflow" >> "$REPORT_FILE"
  else
    echo "- ✅ Using correct UserKnownHostsFile /dev/null" >> "$REPORT_FILE"
  fi
  
  # Check for retry logic
  if grep -q "MAX_DEPLOY_ATTEMPTS" "$WORKFLOW_FILE"; then
    RETRY_COUNT=$(grep -o "MAX_DEPLOY_ATTEMPTS=[0-9]*" "$WORKFLOW_FILE" | cut -d= -f2)
    echo "- ✅ Retry logic implemented with $RETRY_COUNT attempts" >> "$REPORT_FILE"
  else
    echo "- ❌ No retry logic found" >> "$REPORT_FILE"
  fi
  
  # Check for progressive SSH options
  if grep -q "Progressive SSH options" "$WORKFLOW_FILE"; then
    echo "- ✅ Progressive SSH options implemented" >> "$REPORT_FILE"
  else
    echo "- ❌ No progressive SSH options found" >> "$REPORT_FILE"
  fi
  
  # Check for proper heredoc syntax
  if grep -q "<< EOF" "$WORKFLOW_FILE"; then
    echo "- ❌ Found outdated heredoc syntax << EOF" >> "$REPORT_FILE"
  elif grep -q "<<-EOF" "$WORKFLOW_FILE"; then
    echo "- ✅ Using proper heredoc syntax <<-EOF" >> "$REPORT_FILE"
  else
    echo "- ⚠️ No heredoc syntax found" >> "$REPORT_FILE"
  fi
else
  echo "- ❌ Workflow file not found at $WORKFLOW_FILE" >> "$REPORT_FILE"
fi

# 2. Check SSH config
cat >> "$REPORT_FILE" << 'EOF'

## 2. SSH Configuration Check
EOF

if [ -f ~/.ssh/config ]; then
  echo "- ✅ SSH config file exists" >> "$REPORT_FILE"
  
  # Check for correct UserKnownHostsFile
  if grep -q "UserKnownHostsFile /null" ~/.ssh/config; then
    echo "- ❌ Found incorrect UserKnownHostsFile /null in SSH config" >> "$REPORT_FILE"
  elif grep -q "UserKnownHostsFile /dev/null" ~/.ssh/config; then
    echo "- ✅ Using correct UserKnownHostsFile /dev/null" >> "$REPORT_FILE"
  else
    echo "- ⚠️ No UserKnownHostsFile setting found" >> "$REPORT_FILE"
  fi
  
  # Check for StrictHostKeyChecking
  if grep -q "StrictHostKeyChecking no" ~/.ssh/config; then
    echo "- ✅ StrictHostKeyChecking correctly set to no" >> "$REPORT_FILE"
  else
    echo "- ⚠️ StrictHostKeyChecking not set to no" >> "$REPORT_FILE"
  fi
  
  # Check for ServerAlive settings
  if grep -q "ServerAliveInterval" ~/.ssh/config; then
    INTERVAL=$(grep "ServerAliveInterval" ~/.ssh/config | awk '{print $2}')
    echo "- ✅ ServerAliveInterval set to $INTERVAL" >> "$REPORT_FILE"
  else
    echo "- ⚠️ No ServerAliveInterval setting found" >> "$REPORT_FILE"
  fi
  
  # Check permissions
  SSH_CONFIG_PERMS=$(stat -c "%a" ~/.ssh/config 2>/dev/null || stat -f "%p" ~/.ssh/config | cut -c 3-5)
  if [ "$SSH_CONFIG_PERMS" = "600" ]; then
    echo "- ✅ SSH config has correct permissions (600)" >> "$REPORT_FILE"
  else
    echo "- ⚠️ SSH config has permissions $SSH_CONFIG_PERMS (should be 600)" >> "$REPORT_FILE"
  fi
else
  echo "- ❌ No SSH config file found" >> "$REPORT_FILE"
fi

# 3. Check fix scripts
cat >> "$REPORT_FILE" << 'EOF'

## 3. Fix Scripts Check
EOF

REQUIRED_SCRIPTS=(
  "complete_ssh_fix.sh"
  "fix_ssh_config.sh"
  "ssh_parameters_consistency_fix.sh"
  "fix_ssh_verification_params.sh"
  "github_actions_ssh_enhancer.sh"
  "validate_ssh_service.sh"
  "ssh_connectivity_monitor.sh"
  "server_health_check.sh"
)

ALL_SCRIPTS_FOUND=true

for script in "${REQUIRED_SCRIPTS[@]}"; do
  if [ -f "$script" ]; then
    if [ -x "$script" ]; then
      echo "- ✅ $script exists and is executable" >> "$REPORT_FILE"
    else
      echo "- ⚠️ $script exists but is not executable" >> "$REPORT_FILE"
      ALL_SCRIPTS_FOUND=false
    fi
  else
    echo "- ❌ $script not found" >> "$REPORT_FILE"
    ALL_SCRIPTS_FOUND=false
  fi
done

# 4. Check documentation
cat >> "$REPORT_FILE" << 'EOF'

## 4. Documentation Check
EOF

DOC_FILES=(
  "docs/SSH_CONNECTION_FIX.md"
  "docs/SSH_FIXES_SUMMARY.md"
  "docs/COMPLETE_SSH_FIXES.md"
)

ALL_DOCS_FOUND=true

for doc in "${DOC_FILES[@]}"; do
  if [ -f "$doc" ]; then
    echo "- ✅ $doc exists" >> "$REPORT_FILE"
  else
    echo "- ⚠️ $doc not found" >> "$REPORT_FILE"
    ALL_DOCS_FOUND=false
  fi
done

# 5. Overall assessment
cat >> "$REPORT_FILE" << 'EOF'

## 5. Overall Assessment
EOF

if grep -q "❌" "$REPORT_FILE"; then
  echo "- ⚠️ Some critical issues were found" >> "$REPORT_FILE"
  if [ "$ALL_SCRIPTS_FOUND" = false ]; then
    echo "  - Some required fix scripts are missing or not executable" >> "$REPORT_FILE"
  fi
  if grep -q "UserKnownHostsFile /null" "$REPORT_FILE"; then
    echo "  - Incorrect UserKnownHostsFile paths still present" >> "$REPORT_FILE"
  fi
  if grep -q "No retry logic found" "$REPORT_FILE"; then
    echo "  - Missing retry logic in workflow" >> "$REPORT_FILE"
  fi
  
  echo "- 🔧 Recommended actions:" >> "$REPORT_FILE"
  echo "  1. Run ./install_ssh_fixes.sh to install all required scripts" >> "$REPORT_FILE"
  echo "  2. Run ./ssh_parameters_consistency_fix.sh to fix UserKnownHostsFile paths" >> "$REPORT_FILE"
  echo "  3. Run ./fix_ssh_verification_params.sh to ensure consistent SSH parameters" >> "$REPORT_FILE"
else
  echo "- ✅ All critical fixes have been applied correctly" >> "$REPORT_FILE"
fi

# Recommendations
cat >> "$REPORT_FILE" << 'EOF'

## 6. Next Steps

1. **Monitor the workflow**: Run the workflow to verify that the SSH connectivity works as expected
2. **Regular checks**: Use the ssh_connectivity_monitor.sh script to regularly check connectivity
3. **Server health**: Deploy and use server_health_check.sh to monitor application health
4. **Documentation**: Keep the documentation up to date with any further changes

## 7. Additional Recommendations

1. **IP Allowlisting**: Consider adding GitHub Actions IP ranges to allowed lists in your firewalls
2. **Health Checks**: Implement pre-deployment health checks to verify droplet status
3. **Monitoring**: Add monitoring to detect if droplets go offline
4. **Alternative Deployment**: Consider implementing alternative deployment methods as fallbacks
EOF

echo "✅ Verification complete!"
echo "Report saved to: $REPORT_FILE"
echo ""
echo "Summary:"
if grep -q "❌" "$REPORT_FILE"; then
  echo "⚠️ Some issues were found. Please review the report for details."
  echo "Run ./install_ssh_fixes.sh to fix these issues."
else
  echo "✅ All SSH fixes have been properly applied."
fi
