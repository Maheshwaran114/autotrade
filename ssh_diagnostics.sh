#!/bin/bash
# Advanced SSH Diagnostics for GitHub Actions
# This script helps troubleshoot SSH connectivity issues to DigitalOcean droplets

set -e

# Set target IP from argument or environment variable
TARGET_IP=${1:-$DROPLET_IP}

if [ -z "$TARGET_IP" ]; then
  echo "Error: No target IP provided. Usage: $0 <target_ip>"
  echo "You can also set the DROPLET_IP environment variable."
  exit 1
fi

echo "====== SSH CONNECTIVITY DIAGNOSTICS ======"
echo "Target IP: $TARGET_IP"
echo "Current date/time: $(date)"
echo "Running as user: $(whoami)"
echo "Working directory: $(pwd)"

echo "=== SSH Client Version ==="
ssh -V || echo "Failed to determine SSH version"

echo "=== SSH Known Hosts File Check ==="
if grep -q "UserKnownHostsFile /null" ~/.ssh/config 2>/dev/null; then
  echo "❌ ERROR: Found incorrect UserKnownHostsFile /null in SSH config!"
  echo "Fixing SSH config file..."
  sed -i.bak 's|UserKnownHostsFile /null|UserKnownHostsFile /dev/null|g' ~/.ssh/config
  echo "SSH config after fix:"
  cat ~/.ssh/config || echo "Could not read SSH config"
elif grep -q "UserKnownHostsFile /dev/null" ~/.ssh/config 2>/dev/null; then
  echo "✅ SSH config has correct UserKnownHostsFile setting"
  echo "SSH config contents:"
  cat ~/.ssh/config || echo "Could not read SSH config"
else
  echo "⚠️ SSH config does not contain UserKnownHostsFile setting or config file doesn't exist"
  echo "Current SSH config (if exists):"
  cat ~/.ssh/config 2>/dev/null || echo "No SSH config file found"
fi

echo "=== SSH Key Status ==="
if [ -f ~/.ssh/id_bn_trading ]; then
  echo "✅ SSH private key exists at ~/.ssh/id_bn_trading"
  # Check permissions
  SSH_PERMS=$(stat -c "%a" ~/.ssh/id_bn_trading 2>/dev/null || stat -f "%p" ~/.ssh/id_bn_trading 2>/dev/null)
  if [[ "$SSH_PERMS" == "600" ]]; then
    echo "✅ SSH key has correct permissions (600)"
  else
    echo "❌ SSH key has incorrect permissions: $SSH_PERMS (should be 600)"
    echo "Fixing permissions..."
    chmod 600 ~/.ssh/id_bn_trading
  fi
  
  # Verify key format
  if grep -q "BEGIN" ~/.ssh/id_bn_trading; then
    echo "✅ SSH key appears to be in the correct format"
    # Show fingerprint
    echo "SSH key fingerprint:"
    ssh-keygen -l -f ~/.ssh/id_bn_trading || echo "Failed to get fingerprint"
  else
    echo "❌ SSH key does not appear to be in the correct format!"
    echo "  The key should start with: BEGIN OPENSSH PRIVATE KEY or BEGIN RSA PRIVATE KEY"
    echo "  First few bytes of current key (hex):"
    head -c 20 ~/.ssh/id_bn_trading | xxd -p || echo "Failed to read key"
  fi
else
  echo "❌ SSH private key not found at ~/.ssh/id_bn_trading"
  echo "  Please ensure the key is properly set in GitHub secrets"
fi

echo "=== SSH Config ==="
if [ -f ~/.ssh/config ]; then
  echo "SSH config exists:"
  grep -v "IdentityFile" ~/.ssh/config || echo "No SSH config or empty file"
else
  echo "No SSH config file found, creating one..."
  mkdir -p ~/.ssh
  cat > ~/.ssh/config << EOF
Host *
  StrictHostKeyChecking no
  UserKnownHostsFile /dev/null
  ServerAliveInterval 60
  ServerAliveCountMax 10
  ConnectTimeout 30
  IdentitiesOnly yes
EOF
  chmod 600 ~/.ssh/config
  echo "Created SSH config file with sane defaults"
fi

echo "=== TCP Port Scan ==="
echo "Testing if SSH port 22 is open on $TARGET_IP..."
for i in {1..3}; do
  if nc -z -v -w5 $TARGET_IP 22 2>&1; then
    echo "✅ Attempt $i: Port 22 is open and accessible"
    PORT_OPEN=true
    break
  else
    echo "❌ Attempt $i: Port 22 not accessible"
    sleep 2
  fi
done

if [ "$PORT_OPEN" != "true" ]; then
  echo "Testing alternative SSH port 2222..."
  nc -z -v -w5 $TARGET_IP 2222 2>&1 && echo "✅ Port 2222 is open!" || echo "❌ Port 2222 not accessible"
fi

echo "=== DNS Resolution ==="
host $TARGET_IP || echo "Reverse DNS lookup failed"

echo "=== Network Path ==="
echo "Tracing route to target host..."
traceroute -m 15 $TARGET_IP 2>&1 || echo "Traceroute failed or not available"

echo "=== SSH Verbose Connection Test ==="
echo "Attempting SSH connection with maximum verbosity..."
timeout 30 ssh -vvv -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i ~/.ssh/id_bn_trading root@$TARGET_IP exit 2>&1 || echo "SSH connection failed or timed out"

echo "=== SSH Authentication Methods ==="
echo "Checking accepted authentication methods..."
ssh -o PreferredAuthentications=none -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@$TARGET_IP 2>&1 | grep "Authentication methods" || echo "Could not determine authentication methods"

echo "=== SSH Key Verification ==="
echo "Checking if your SSH public key is authorized on the server..."
SSH_PUBKEY=$(ssh-keygen -y -f ~/.ssh/id_bn_trading 2>/dev/null || echo "Failed to extract public key")
if [ -n "$SSH_PUBKEY" ] && [ "$SSH_PUBKEY" != "Failed to extract public key" ]; then
  echo "Public key for verification:"
  echo "$SSH_PUBKEY" | cut -d' ' -f1-2 # Only show key type and prefix for security
else
  echo "❌ Could not extract public key from private key"
fi

echo "====== SSH DIAGNOSTICS COMPLETED ======"
echo "If you're still having issues, consider:"
echo "1. Verifying the SSH_KEY_ID in GitHub secrets matches a key in DigitalOcean"
echo "2. Checking if the droplet was created with the correct SSH key"
echo "3. Trying to recreate the droplet via Terraform"
echo "4. Using the DigitalOcean console to reset the root password"
