#!/bin/bash
# Server-side SSH Configuration Test
# This script should be run on the DigitalOcean droplet to check if SSH is properly configured

set -e

echo "===== SERVER-SIDE SSH CONFIGURATION TEST ====="
echo "Date/Time: $(date)"
echo "Hostname: $(hostname)"
echo "IP Addresses:"
ip addr | grep inet | grep -v '127.0.0.1' | awk '{print $2}'

# Check SSH service status
echo "=== SSH Service Status ==="
systemctl status sshd || service ssh status || echo "Could not get SSH service status"

# Check SSH configuration
echo "=== SSH Server Configuration ==="
if [ -f /etc/ssh/sshd_config ]; then
  echo "SSH configuration exists. Checking settings..."
  
  # Check important settings
  echo "PermitRootLogin setting:"
  grep -i "PermitRootLogin" /etc/ssh/sshd_config || echo "PermitRootLogin not explicitly set"
  
  echo "PubkeyAuthentication setting:"
  grep -i "PubkeyAuthentication" /etc/ssh/sshd_config || echo "PubkeyAuthentication not explicitly set"
  
  echo "PasswordAuthentication setting:"
  grep -i "PasswordAuthentication" /etc/ssh/sshd_config || echo "PasswordAuthentication not explicitly set"
  
  echo "ListenAddress setting:"
  grep -i "ListenAddress" /etc/ssh/sshd_config || echo "ListenAddress not explicitly set"
else
  echo "❌ SSH server configuration file not found!"
fi

# Check authorized keys
echo "=== Authorized Keys ==="
echo "Root user authorized keys:"
if [ -f /root/.ssh/authorized_keys ]; then
  echo "✅ authorized_keys file exists for root user"
  echo "Number of keys: $(wc -l < /root/.ssh/authorized_keys)"
  echo "Key fingerprints:"
  while read -r key; do
    echo "$key" | ssh-keygen -lf - 2>/dev/null || echo "Invalid key format"
  done < /root/.ssh/authorized_keys
else
  echo "❌ No authorized_keys file for root user!"
fi

# Check firewall status
echo "=== Firewall Status ==="
if command -v ufw >/dev/null; then
  echo "UFW status:"
  ufw status || echo "Could not get UFW status"
else
  echo "UFW not installed"
fi

if command -v iptables >/dev/null; then
  echo "iptables rules:"
  iptables -L | grep -i ssh || echo "No SSH-related iptables rules found"
else
  echo "iptables not available"
fi

# Check for connectivity to GitHub Actions IP ranges
echo "=== GitHub Actions Connectivity Test ==="
echo "Testing outbound connectivity:"
curl -s https://api.github.com || echo "Cannot connect to GitHub API"

echo "Testing connection to common GitHub Actions IP ranges:"
for ip in 13.107.42.16 20.140.41.0 20.140.48.0; do
  echo -n "Testing $ip: "
  ping -c 1 -W 2 $ip >/dev/null 2>&1 && echo "✅ Success" || echo "❌ Failed"
done

echo "===== SERVER-SIDE SSH CONFIGURATION TEST COMPLETE ====="
echo "To fix common issues:"
echo "1. Ensure 'PermitRootLogin yes' is set in /etc/ssh/sshd_config"
echo "2. Ensure 'PubkeyAuthentication yes' is set in /etc/ssh/sshd_config"
echo "3. Restart SSH with: systemctl restart sshd"
echo "4. Ensure port 22 is open in any firewalls: ufw allow 22/tcp"
