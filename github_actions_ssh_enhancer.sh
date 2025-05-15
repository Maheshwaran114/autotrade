#!/bin/bash
# GitHub Actions SSH Connection Enhancer
# This script adds specialized SSH retry logic with alternative connection settings

set -e

TARGET_IP="$1"
if [ -z "$TARGET_IP" ]; then
  echo "Error: No target IP provided"
  echo "Usage: $0 <target_ip>"
  exit 1
fi

echo "===== GITHUB ACTIONS SSH CONNECTION ENHANCER ====="
echo "Target IP: $TARGET_IP"
echo "Date/Time: $(date)"

# Verify SSH key permissions
SSH_KEY_PATH=~/.ssh/id_bn_trading
if [ -f "$SSH_KEY_PATH" ]; then
  echo "SSH key exists at $SSH_KEY_PATH"
  chmod 600 "$SSH_KEY_PATH"
  echo "✅ Fixed SSH key permissions"
  
  # Check key format (first line only, no secrets)
  FIRST_LINE=$(head -n 1 "$SSH_KEY_PATH")
  if [[ "$FIRST_LINE" == *"PRIVATE KEY"* ]]; then
    echo "✅ SSH key appears to be in the correct format"
  else
    echo "❌ SSH key might not be in the correct format!"
    echo "The first line should contain 'PRIVATE KEY'"
  fi
else
  echo "❌ SSH key not found at $SSH_KEY_PATH"
  exit 1
fi

# Fix UserKnownHostsFile path if misconfigured
if grep -q "UserKnownHostsFile /null" ~/.ssh/config 2>/dev/null; then
  echo "❌ Found incorrectly configured UserKnownHostsFile in ~/.ssh/config"
  echo "Fixing the configuration..."
  sed -i 's|UserKnownHostsFile /null|UserKnownHostsFile /dev/null|g' ~/.ssh/config
  echo "✅ Fixed SSH config"
fi

# Enhanced SSH connection options
CONNECTION_OPTS=(
  # Default options for all attempts
  "-o StrictHostKeyChecking=no"
  "-o UserKnownHostsFile=/dev/null"
  
  # Connectivity optimization options
  "-o ServerAliveInterval=5 -o ServerAliveCountMax=10 -o ConnectTimeout=60"
  "-o ConnectTimeout=90 -o BatchMode=yes -o IPQoS=throughput"
  "-o PreferredAuthentications=publickey -o TCPKeepAlive=yes -o AddressFamily=inet"
  
  # Protocol/algorithm optimizations
  "-o KexAlgorithms=curve25519-sha256,diffie-hellman-group-exchange-sha256,diffie-hellman-group14-sha256"
  "-o Ciphers=chacha20-poly1305@openssh.com,aes256-gcm@openssh.com,aes128-gcm@openssh.com"
  
  # Verbosity options
  "-v"
  "-vv"
  "-vvv"
)

# Ports to try
SSH_PORTS=(22 2222)

echo "=== PREPARING FOR CONNECTION ATTEMPTS ==="
echo "Will try ${#CONNECTION_OPTS[@]} connection option combinations on ${#SSH_PORTS[@]} ports"
echo "Total potential attempts: $((${#CONNECTION_OPTS[@]} * ${#SSH_PORTS[@]}))"

# First check if basic connectivity exists
echo "=== BASIC CONNECTIVITY CHECK ==="
if ping -c 3 -W 5 "$TARGET_IP" &>/dev/null; then
  echo "✅ Basic connectivity exists (ping successful)"
else
  echo "⚠️ Warning: Basic connectivity test failed (ping unsuccessful)"
  echo "This could be normal if ICMP is blocked"
fi

# Check if ports are accessible
for PORT in "${SSH_PORTS[@]}"; do
  echo "Checking if port $PORT is open..."
  if timeout 10 nc -z -v -w5 "$TARGET_IP" "$PORT" 2>&1; then
    echo "✅ Port $PORT is open on $TARGET_IP"
  else
    echo "❌ Port $PORT appears to be closed on $TARGET_IP"
  fi
done

# Try different SSH connection combinations
echo "=== ATTEMPTING SSH CONNECTIONS ==="
SUCCESS=false

for PORT in "${SSH_PORTS[@]}"; do
  echo "--- Trying port $PORT ---"
  for OPTS in "${CONNECTION_OPTS[@]}"; do
    echo "Attempt with options: $OPTS on port $PORT"
    
    # Try with timeout to avoid hanging
    if timeout 30 ssh $OPTS -p "$PORT" -i "$SSH_KEY_PATH" "root@$TARGET_IP" 'echo "✅ CONNECTION SUCCESSFUL"; hostname; uptime' 2>&1; then
      echo "🎉 CONNECTION SUCCESSFUL with options: $OPTS on port $PORT"
      echo "Recommended SSH command for future connections:"
      echo "ssh $OPTS -p $PORT -i $SSH_KEY_PATH root@$TARGET_IP"
      SUCCESS=true
      break 2
    else
      echo "❌ Connection failed with these options"
    fi
    
    # Short delay between attempts
    sleep 2
  done
  
  if [ "$SUCCESS" = true ]; then
    break
  fi
done

if [ "$SUCCESS" = false ]; then
  echo "=== ADDITIONAL DIAGNOSTICS ==="
  echo "All connection attempts failed. Running additional diagnostics..."
  
  # Network path analysis
  echo "Analyzing network path with traceroute:"
  traceroute -m 20 "$TARGET_IP" || echo "Traceroute failed"
  
  # Check for IP address information
  echo "IP address information:"
  whois "$TARGET_IP" | grep -E "inetnum|netname|descr|country|org" || echo "Whois lookup failed"
  
  # Check if we can connect to standard ports
  for TEST_PORT in 80 443; do
    echo "Testing connection to port $TEST_PORT:"
    timeout 5 nc -z -v -w3 "$TARGET_IP" "$TEST_PORT" 2>&1 || echo "Connection to port $TEST_PORT failed"
  done
  
  echo "=== CONNECTION RECOMMENDATIONS ==="
  echo "1. Verify that the SSH service is running on the target server"
  echo "2. Check if any firewalls are blocking the connection"
  echo "3. Verify that the SSH key is properly authorized on the target server"
  echo "4. Try connecting from a different network or using a VPN"
  
  # Exit with error code
  exit 1
else
  echo "=== CONNECTION SUCCESSFUL ==="
  echo "Successfully established SSH connection to $TARGET_IP"
  echo "Use the recommended command for future connections"
  exit 0
fi
