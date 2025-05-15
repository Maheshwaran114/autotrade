#!/bin/bash
# Script to verify SSH key format and permissions
# This script helps diagnose SSH key issues in the GitHub Actions workflow

echo "===== SSH Key Format Test ====="
echo "This script will check if your SSH private key is properly formatted for use in GitHub Actions."

# Default path to test
DEFAULT_KEY_PATH="~/.ssh/id_bn_trading"
KEY_PATH=${1:-$DEFAULT_KEY_PATH}

# Expand the path if using ~
KEY_PATH=$(eval echo "$KEY_PATH")

# Check if the file exists
if [ ! -f "$KEY_PATH" ]; then
  echo "❌ Error: SSH key file not found at $KEY_PATH"
  echo "Usage: $0 [path/to/ssh_key]"
  exit 1
fi

# Check key permissions
PERMS=$(stat -f "%p" "$KEY_PATH" 2>/dev/null || stat -c "%a" "$KEY_PATH" 2>/dev/null)
echo "Current key permissions: $PERMS"

if [ "$PERMS" != "600" ]; then
  echo "⚠️ Warning: SSH key has incorrect permissions ($PERMS). Should be 600."
  echo "Fixing permissions..."
  chmod 600 "$KEY_PATH"
fi

# Verify key starts with the proper header
if grep -q "BEGIN" "$KEY_PATH" && grep -q "PRIVATE KEY" "$KEY_PATH"; then
  echo "✅ SSH key appears to be in the correct format"
  
  # Try to check if the key is valid
  if ssh-keygen -y -f "$KEY_PATH" &>/dev/null; then
    echo "✅ Key is valid and can be used to generate a public key"
    echo "Key type: $(ssh-keygen -l -f "$KEY_PATH" | cut -d ' ' -f 3)"
  else
    echo "❌ Key appears to be in the right format but is invalid"
    echo "This might be due to incorrect passphrase or corrupted key"
  fi
else
  echo "❌ SSH key does not appear to be in the correct format!"
  echo "The key should start with: BEGIN OPENSSH PRIVATE KEY or BEGIN RSA PRIVATE KEY"
  echo "First line of your key file (sensitive data redacted):"
  head -n 1 "$KEY_PATH" | sed 's/^.*$/Format: &/' | sed 's/PRIVATE KEY/----- PRIVATE KEY (redacted) -----/'
  
  echo -e "\nIs this actually a public key? Let's check:"
  if grep -q "ssh-rsa" "$KEY_PATH" || grep -q "ssh-ed25519" "$KEY_PATH"; then
    echo "⚠️ This appears to be a PUBLIC key, not a private key!"
    echo "For GitHub Actions, you need to add your PRIVATE key to the secrets, not the public key."
  fi
fi

echo -e "\n===== Recommendations ====="
echo "1. Ensure you're using the private key (not the .pub file)"
echo "2. Key should have 600 permissions (chmod 600 ~/.ssh/id_bn_trading)"
echo "3. Key should start with -----BEGIN OPENSSH PRIVATE KEY----- or similar"
echo "4. If using an SSH key with a passphrase, you may need a different approach"
echo "5. Try generating a new key pair specifically for GitHub Actions:"
echo "   ssh-keygen -t ed25519 -f ~/.ssh/github_actions_key -C \"github-actions\""
echo ""
echo "Then add the private key content to GitHub repository secrets as SSH_PRIVATE_KEY"
echo "and add the public key to your DigitalOcean account under Security > SSH Keys."
