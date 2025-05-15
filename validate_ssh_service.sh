#!/bin/bash
# Droplet SSH Service Validator
# This script should be run on the DigitalOcean droplet to verify SSH service is properly configured

set -e

echo "===== DROPLET SSH SERVICE VALIDATOR ====="
echo "Date/Time: $(date)"
echo "Hostname: $(hostname)"
echo "IP Addresses:"
ip addr | grep 'inet ' | grep -v '127.0.0.1'

# Check if SSH daemon is running
echo "=== SSH SERVICE STATUS ==="
systemctl status sshd || service ssh status || echo "Could not get SSH service status"

# Check if SSH is listening on port 22
echo "=== SSH LISTENING STATUS ==="
if command -v netstat &> /dev/null; then
  netstat -tulpn | grep ':22 ' || echo "SSH not found listening on port 22"
elif command -v ss &> /dev/null; then
  ss -tulpn | grep ':22 ' || echo "SSH not found listening on port 22"
else
  echo "Could not check listening ports (netstat/ss not available)"
fi

# Check SSH configuration
echo "=== SSH SERVER CONFIGURATION ==="
if [ -f /etc/ssh/sshd_config ]; then
  echo "SSH configuration exists. Checking key settings..."
  
  # Check Port setting
  echo "Port setting:"
  grep -i "^Port" /etc/ssh/sshd_config || echo "Default port (22) is used"
  
  # Check listen addresses
  echo "ListenAddress setting:"
  grep -i "^ListenAddress" /etc/ssh/sshd_config || echo "Default listen address (all interfaces) is used"
  
  # Check key authentication settings
  echo "PubkeyAuthentication setting:"
  grep -i "^PubkeyAuthentication" /etc/ssh/sshd_config || echo "Default PubkeyAuthentication setting is used"
  
  # Check root login settings
  echo "PermitRootLogin setting:"
  grep -i "^PermitRootLogin" /etc/ssh/sshd_config || echo "Default PermitRootLogin setting is used"
else
  echo "❌ SSH server configuration file not found!"
fi

# Check authorized keys
echo "=== AUTHORIZED KEYS ==="
echo "Root user authorized keys:"
if [ -f /root/.ssh/authorized_keys ]; then
  echo "✅ authorized_keys file exists for root user"
  echo "Number of keys: $(wc -l < /root/.ssh/authorized_keys)"
  
  # Show key fingerprints
  echo "Key fingerprints:"
  while read -r key; do
    echo "$key" | ssh-keygen -lf - 2>/dev/null || echo "Invalid key format"
  done < /root/.ssh/authorized_keys
else
  echo "❌ No authorized_keys file for root user!"
fi

# Check firewall status
echo "=== FIREWALL STATUS ==="
# UFW Check
if command -v ufw &> /dev/null; then
  echo "UFW firewall status:"
  ufw status verbose
  
  # Check if port 22 is allowed
  if ufw status | grep -q "22/tcp.*ALLOW"; then
    echo "✅ Port 22 is allowed in UFW"
  else
    echo "❌ Port 22 may not be allowed in UFW!"
  fi
else
  echo "UFW not installed"
fi

# iptables Check
if command -v iptables &> /dev/null; then
  echo "iptables rules for SSH:"
  iptables -L INPUT -n | grep -i "22" || echo "No explicit SSH rules found in iptables"
fi

# Test connectivity from the server
echo "=== OUTBOUND CONNECTIVITY TEST ==="
echo "Testing outbound connectivity to common services:"
curl -s https://www.google.com > /dev/null 2>&1 && echo "✅ Connection to Google successful" || echo "❌ Connection to Google failed"
curl -s https://api.github.com > /dev/null 2>&1 && echo "✅ Connection to GitHub API successful" || echo "❌ Connection to GitHub API failed"

# Test local SSH connection
echo "=== LOCAL SSH CONNECTION TEST ==="
if command -v ssh &> /dev/null; then
  ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no localhost echo "SSH localhost test" &>/dev/null && echo "✅ Local SSH connection successful" || echo "❌ Local SSH connection failed"
else
  echo "SSH client not installed, cannot test local connection"
fi

echo "=== RECOMMENDED FIXES ==="
echo "If SSH service is not working properly, you can try these fixes:"
echo "1. Ensure sshd is running: systemctl start sshd"
echo "2. Verify sshd_config has proper settings:"
echo "   - PermitRootLogin yes (if you need root login)"
echo "   - PubkeyAuthentication yes"
echo "   - Port 22 (or your custom port)"
echo "3. Ensure firewall allows port 22: ufw allow 22/tcp"
echo "4. Check authorized_keys contains proper keys"
echo "5. Restart SSH after changes: systemctl restart sshd"

echo "===== SSH SERVICE VALIDATION COMPLETE ====="
