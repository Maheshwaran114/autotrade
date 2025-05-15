#!/bin/bash
# Enhanced SSH Connection Diagnostic Script
# This script helps diagnose SSH connection issues in GitHub Actions

set -e

# Get the target IP address
TARGET_IP=$1
if [ -z "$TARGET_IP" ]; then
  echo "Error: No target IP provided"
  echo "Usage: $0 <target_ip>"
  exit 1
fi

echo "===== ENHANCED SSH CONNECTION DIAGNOSTICS ====="
echo "Target IP: $TARGET_IP"
echo

# 1. Basic network connectivity tests
echo "=== NETWORK CONNECTIVITY TESTS ==="
echo "Testing if host is reachable..."
ping -c 3 $TARGET_IP || echo "❌ Ping failed (this might be normal if ICMP is blocked)"

echo "Testing if SSH port is open..."
nc -z -v -w5 $TARGET_IP 22 || echo "❌ SSH port test failed"

echo "Tracing route to target..."
traceroute -m 15 $TARGET_IP || echo "❌ Traceroute failed"

# 2. SSH client and configuration
echo "=== SSH CLIENT INFO ==="
echo "SSH version:"
ssh -V

echo "SSH config file contents:"
cat ~/.ssh/config || echo "No SSH config file found"

# 3. SSH key verification
echo "=== SSH KEY VERIFICATION ==="
if [ -f ~/.ssh/id_bn_trading ]; then
  echo "✅ SSH key exists"
  ls -la ~/.ssh/id_bn_trading
  
  echo "SSH key format:"
  ssh-keygen -l -f ~/.ssh/id_bn_trading || echo "❌ Failed to get key fingerprint"
  
  echo "Key header format (safe to show):"
  head -n 1 ~/.ssh/id_bn_trading
  
  echo "Checking key permissions:"
  chmod 600 ~/.ssh/id_bn_trading
  echo "✅ Permissions set to 600"
else
  echo "❌ SSH key not found at ~/.ssh/id_bn_trading"
fi

# 4. SSH connection test with maximum verbosity
echo "=== SSH CONNECTION TEST WITH MAXIMUM VERBOSITY ==="
echo "Attempting SSH connection with triple verbose (-vvv)..."
timeout 30 ssh -vvv -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -i ~/.ssh/id_bn_trading root@$TARGET_IP exit 2>&1 || echo "SSH connection failed"

# 5. Testing alternative configurations
echo "=== TESTING ALTERNATIVE CONFIGURATIONS ==="
echo "Testing connection without BatchMode..."
timeout 30 ssh -vvv -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -i ~/.ssh/id_bn_trading root@$TARGET_IP exit 2>&1 || echo "SSH connection failed"

echo "Testing with explicit key exchange algorithms..."
timeout 30 ssh -vvv -o KexAlgorithms=curve25519-sha256,diffie-hellman-group-exchange-sha256 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -i ~/.ssh/id_bn_trading root@$TARGET_IP exit 2>&1 || echo "SSH connection failed"

echo "===== DIAGNOSTICS COMPLETE ====="
echo
echo "If all tests failed, consider:"
echo "1. Verify the SSH key is added to the DigitalOcean droplet's authorized keys"
echo "2. Check if the DigitalOcean droplet allows SSH root login"
echo "3. Ensure firewall rules on the droplet allow SSH connections"
echo "4. Verify the droplet is running and SSH service is active"
