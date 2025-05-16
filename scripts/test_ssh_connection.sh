#!/bin/bash
# Test SSH connectivity to the droplet

set -e

DROPLET_IP="165.22.214.71"  # Update this with your actual droplet IP
SSH_KEY="./scripts/do_deployment_key"

echo "Testing SSH connectivity to $DROPLET_IP..."
ssh -v -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i "$SSH_KEY" root@$DROPLET_IP echo "SSH Connection Test"

if [ $? -eq 0 ]; then
  echo "✅ SSH connection successful!"
else
  echo "❌ SSH connection failed. Trying with insecure cipher options..."
  ssh -v -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o PubkeyAuthentication=yes \
    -o PreferredAuthentications=publickey -o "HostKeyAlgorithms=+ssh-rsa" \
    -o "PubkeyAcceptedAlgorithms=+ssh-rsa" -i "$SSH_KEY" root@$DROPLET_IP echo "SSH Connection Test"
  
  if [ $? -eq 0 ]; then
    echo "✅ SSH connection successful with insecure cipher options!"
  else
    echo "❌ All SSH connection attempts failed."
  fi
fi
