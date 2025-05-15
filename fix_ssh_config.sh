#!/bin/bash
# SSH Config Verification and Fix Script for GitHub Actions
# Use this script to detect and fix issues with SSH configuration

set -e

echo "===== SSH CONFIG VALIDATION AND FIX ====="
echo "Date/Time: $(date)"
echo "User: $(whoami)"

# Create SSH directory if it doesn't exist
if [ ! -d ~/.ssh ]; then
  echo "Creating SSH directory..."
  mkdir -p ~/.ssh
  chmod 700 ~/.ssh
  echo "✅ Created SSH directory with proper permissions"
else
  echo "✅ SSH directory exists"
  ls -la ~/.ssh || echo "Failed to list SSH directory contents"
fi

# Check for SSH config file
if [ -f ~/.ssh/config ]; then
  echo "SSH config exists. Checking for issues..."
  
  # Backup current config
  cp ~/.ssh/config ~/.ssh/config.bak
  echo "✅ Backed up existing SSH config to ~/.ssh/config.bak"
  
  # Check for incorrect UserKnownHostsFile setting
  if grep -q "UserKnownHostsFile /null" ~/.ssh/config; then
    echo "❌ Found incorrect 'UserKnownHostsFile /null' in SSH config"
    echo "Fixing the issue..."
    sed -i.bak 's|UserKnownHostsFile /null|UserKnownHostsFile /dev/null|g' ~/.ssh/config
    echo "✅ Fixed UserKnownHostsFile setting"
  else
    echo "✅ UserKnownHostsFile setting is correct or not present"
  fi
  
  # Check for other common issues
  if ! grep -q "StrictHostKeyChecking" ~/.ssh/config; then
    echo "⚠️ StrictHostKeyChecking is not set in SSH config"
    echo "Adding StrictHostKeyChecking no to config..."
    echo "StrictHostKeyChecking no" >> ~/.ssh/config
  fi
  
  # Check if LogLevel is too verbose
  if grep -q "LogLevel DEBUG" ~/.ssh/config; then
    echo "⚠️ SSH LogLevel is set to DEBUG which can be too verbose"
    echo "Changing to VERBOSE level..."
    sed -i.bak2 's|LogLevel DEBUG|LogLevel VERBOSE|g' ~/.ssh/config
  fi
  
  # Ensure proper permissions on config file
  chmod 600 ~/.ssh/config
  echo "✅ Set SSH config permissions to 600"
  
else
  echo "No SSH config file found. Creating a new one with optimal settings..."
  
  cat > ~/.ssh/config << 'EOF'
Host *
  StrictHostKeyChecking no
  UserKnownHostsFile /dev/null
  ServerAliveInterval 60
  ServerAliveCountMax 10
  ConnectTimeout 180
  IdentitiesOnly yes
  LogLevel VERBOSE
  BatchMode yes
  IdentityFile ~/.ssh/id_bn_trading
EOF
  chmod 600 ~/.ssh/config
  echo "✅ Created new SSH config with proper settings"
fi

# Show current SSH config
echo "=== CURRENT SSH CONFIG ==="
cat ~/.ssh/config
echo "=========================="

# Check SSH key
if [ -f ~/.ssh/id_bn_trading ]; then
  echo "✅ SSH key exists at ~/.ssh/id_bn_trading"
  
  # Check permissions
  chmod 600 ~/.ssh/id_bn_trading
  echo "✅ Set key permissions to 600"
  
  # Check key format
  if head -1 ~/.ssh/id_bn_trading | grep -q "BEGIN"; then
    echo "✅ Key appears to be in the correct format"
    echo "Key type: $(ssh-keygen -l -f ~/.ssh/id_bn_trading | awk '{print $2}' || echo "Could not determine key type")"
  else
    echo "❌ Key does not appear to be in the correct format!"
    echo "First line of key file (redacted):"
    head -1 ~/.ssh/id_bn_trading | sed 's/.*/----- REDACTED -----/'
  fi
else
  echo "❌ SSH key not found at ~/.ssh/id_bn_trading"
fi

# Test SSH with enhanced options
if [ -n "$1" ]; then
  TARGET_IP="$1"
  echo "=== TESTING SSH CONNECTION TO $TARGET_IP ==="
  echo "Using enhanced connection parameters..."
  
  # Try to connect with robust parameters
  set +e  # Don't exit on error
  timeout 30 ssh -vvv \
    -o BatchMode=yes \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o ServerAliveInterval=30 \
    -o ConnectTimeout=30 \
    -o ConnectionAttempts=3 \
    -i ~/.ssh/id_bn_trading \
    root@$TARGET_IP exit 2>&1
  
  SSH_EXIT=$?
  echo "SSH exit code: $SSH_EXIT"
  if [ $SSH_EXIT -eq 0 ]; then
    echo "✅ SSH connection successful!"
  else
    echo "❌ SSH connection failed with exit code $SSH_EXIT"
  fi
else
  echo "No IP address provided. Skip connection test."
  echo "To test connection, run: $0 <ip_address>"
fi

echo "===== SSH CONFIG VALIDATION COMPLETE ====="
