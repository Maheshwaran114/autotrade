#!/bin/bash
# SSH Connection Test Script
# This script simulates the GitHub Actions workflow SSH steps to diagnose connectivity issues

set -e  # Exit on error

echo "===== SSH Connectivity Test Script ====="
echo "This script will test SSH connectivity similar to our GitHub Actions workflow"
echo

# Configuration - replace with real values as needed for testing
REMOTE_IP="64.227.129.85"  # Your DigitalOcean droplet IP
SSH_KEY_FILE="$HOME/.ssh/id_bn_trading"  # Path to your BN Trading SSH key

# Create temporary test directory
TEST_DIR=$(mktemp -d)
echo "Created temporary test directory: $TEST_DIR"

cleanup() {
  echo "Cleaning up temporary directory..."
  rm -rf "$TEST_DIR"
}
trap cleanup EXIT

# Ask for IP if not provided
if [ -z "$REMOTE_IP" ]; then
  echo "Please enter the IP address of a DigitalOcean droplet to test:"
  read -r REMOTE_IP
fi

# Step 1: Prepare SSH environment
echo
echo "Step 1: Setting up SSH config (similar to GitHub Actions)"
mkdir -p "$TEST_DIR/.ssh"
chmod 700 "$TEST_DIR/.ssh"

echo "Host *
  StrictHostKeyChecking no
  UserKnownHostsFile /dev/null" > "$TEST_DIR/.ssh/config"
chmod 600 "$TEST_DIR/.ssh/config"

if [ -f "$SSH_KEY_FILE" ]; then
  echo "Found SSH key at $SSH_KEY_FILE"
  cp "$SSH_KEY_FILE" "$TEST_DIR/.ssh/id_bn_trading"
  chmod 600 "$TEST_DIR/.ssh/id_bn_trading"
else
  echo "⚠️ Warning: SSH key file not found at $SSH_KEY_FILE"
  echo "Please provide the path to your SSH private key:"
  read -r SSH_KEY_FILE
  if [ -f "$SSH_KEY_FILE" ]; then
    cp "$SSH_KEY_FILE" "$TEST_DIR/.ssh/id_bn_trading"
    chmod 600 "$TEST_DIR/.ssh/id_bn_trading"
  else
    echo "❌ Error: SSH key file not found. Exiting."
    exit 1
  fi
fi

# Step 2: Test SSH connection
echo
echo "Step 2: Validating IP format"
if [[ ! "$REMOTE_IP" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "❌ Error: '$REMOTE_IP' doesn't look like a valid IP address."
  echo "Please enter the IP address of your DigitalOcean droplet:"
  read -r REMOTE_IP
  if [[ ! "$REMOTE_IP" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "❌ Error: Still not a valid IP format. Exiting."
    exit 1
  fi
fi

# Step 3: Test SSH connection with verbose output
echo
echo "Step 3: Testing SSH connection to $REMOTE_IP"
echo "Testing SSH connectivity with verbose output..."

echo "Attempting to connect with verbose output..."
ssh -v -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i "$TEST_DIR/.ssh/id_bn_trading" "root@$REMOTE_IP" exit 0 || {
  status=$?
  echo "❌ SSH connection test failed with status $status"
  echo
  echo "Troubleshooting next steps:"
  echo "1. Verify the IP address is correct: $REMOTE_IP"
  echo "2. Check that your SSH key has been added to DigitalOcean account"
  echo "   Run: ./register_ssh_key.sh"
  echo "3. Ensure the droplet was created with your SSH key ID"
  echo "4. Verify the droplet is running and accessible"
  
  # Try again with password if available
  echo
  echo "Would you like to try connecting with password authentication? (y/n)"
  read -r TRY_PASSWORD
  if [[ "$TRY_PASSWORD" == "y" ]]; then
    echo "Attempting to connect with password..."
    ssh -o StrictHostKeyChecking=no "root@$REMOTE_IP"
  fi
  
  exit $status
}

echo "✅ SSH connection successful!"

# Step 4: Test deployment commands
echo
echo "Step 4: Testing deployment commands"
ssh -o BatchMode=yes -o StrictHostKeyChecking=no -i "$TEST_DIR/.ssh/id_bn_trading" "root@$REMOTE_IP" << 'TESTEOF'
echo "SSH connection established!"
echo "Checking for /bn-trading directory..."
if [ -d /bn-trading ]; then
  echo "✅ /bn-trading directory exists"
  cd /bn-trading
  echo "Current directory: $(pwd)"
  echo "Files in current directory:"
  ls -la
else
  echo "❌ /bn-trading directory does not exist"
fi
echo "Checking Docker installation..."
if command -v docker &>/dev/null; then
  echo "✅ Docker is installed: $(docker --version)"
else
  echo "❌ Docker is not installed"
fi
if command -v docker-compose &>/dev/null; then
  echo "✅ Docker Compose is installed: $(docker-compose --version)"
else
  echo "❌ Docker Compose is not installed"
fi
TESTEOF

echo
echo "===== Test Complete ====="
echo "All SSH communication tests have passed successfully!"
echo "You should be able to deploy via SSH from GitHub Actions if the environment variables are set correctly."
