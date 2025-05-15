#!/bin/bash
# Network Connectivity Repair Tool for DigitalOcean Droplets
# This script helps diagnose and fix network issues with SSH connections to DigitalOcean

set -e

# Get target IP
TARGET_IP=${1:-$DROPLET_IP}

if [ -z "$TARGET_IP" ]; then
  echo "Error: No target IP provided"
  echo "Usage: $0 <target_ip>"
  exit 1
fi

echo "===== NETWORK CONNECTIVITY REPAIR TOOL ====="
echo "Target IP: $TARGET_IP"
echo "Current date/time: $(date)"

# Check if the Digital Ocean API token is available
if [ -z "$DIGITALOCEAN_TOKEN" ]; then
  # Check if it's available in a file
  if [ -f ~/.digitalocean_token ]; then
    echo "Loading DigitalOcean token from file..."
    export DIGITALOCEAN_TOKEN=$(cat ~/.digitalocean_token)
  elif [ -f $HOME/.digitalocean_token ]; then
    echo "Loading DigitalOcean token from home directory..."
    export DIGITALOCEAN_TOKEN=$(cat $HOME/.digitalocean_token)
  else
    echo "Warning: DigitalOcean API token not found. Some tests will be skipped."
    echo "To set it: export DIGITALOCEAN_TOKEN=your_token"
  fi
fi

# Function to check basic connectivity
check_basic_connectivity() {
  echo "=== BASIC CONNECTIVITY TESTS ==="
  
  # Ping test with increased count and timeout
  echo "Testing basic connectivity with ping (5 packets):"
  ping -c 5 -W 5 $TARGET_IP || { 
    echo "❌ Ping failed. This could be due to ICMP being blocked or network issues."
  }
  
  # TCP port scan with increased timeout
  echo "Testing if SSH port 22 is open with increased timeout:"
  for i in {1..3}; do
    if timeout 15 nc -z -v -w 10 $TARGET_IP 22 2>&1; then
      echo "✅ Port 22 appears to be open on attempt $i"
      PORT_OPEN=true
      break
    else
      echo "❌ Port 22 appears to be closed on attempt $i"
      sleep 3
    fi
  done
  
  # Try alternative ports
  if [ "$PORT_OPEN" != "true" ]; then
    echo "Testing alternative SSH port 2222:"
    timeout 10 nc -z -v -w 5 $TARGET_IP 2222 2>&1 && echo "✅ Port 2222 is open!" || echo "❌ Port 2222 is not accessible"
  fi
}

# Check droplet status via API
check_droplet_status() {
  if [ -n "$DIGITALOCEAN_TOKEN" ]; then
    echo "=== DROPLET STATUS CHECK ==="
    echo "Checking droplet status via DigitalOcean API..."
    
    # Find droplet by IP
    DROPLET_INFO=$(curl -s -X GET \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $DIGITALOCEAN_TOKEN" \
      "https://api.digitalocean.com/v2/droplets?per_page=200" | \
      jq -r ".droplets[] | select(.networks.v4[].ip_address==\"$TARGET_IP\")" 2>/dev/null)
    
    if [ -n "$DROPLET_INFO" ]; then
      echo "✅ Found droplet with IP $TARGET_IP"
      # Extract and show important details
      DROPLET_ID=$(echo "$DROPLET_INFO" | jq -r '.id')
      DROPLET_NAME=$(echo "$DROPLET_INFO" | jq -r '.name')
      DROPLET_STATUS=$(echo "$DROPLET_INFO" | jq -r '.status')
      
      echo "Droplet Name: $DROPLET_NAME"
      echo "Droplet ID: $DROPLET_ID"
      echo "Droplet Status: $DROPLET_STATUS"
      
      if [ "$DROPLET_STATUS" != "active" ]; then
        echo "❌ Droplet is not in 'active' state! This is likely causing the connection issues."
        echo "Current status: $DROPLET_STATUS"
        
        # Offer to power on the droplet if it's powered off
        if [ "$DROPLET_STATUS" = "off" ]; then
          echo "Would you like to power on the droplet? (y/n)"
          read -r POWER_ON
          if [[ "$POWER_ON" =~ ^[Yy]$ ]]; then
            echo "Powering on droplet ID $DROPLET_ID..."
            curl -s -X POST \
              -H "Content-Type: application/json" \
              -H "Authorization: Bearer $DIGITALOCEAN_TOKEN" \
              "https://api.digitalocean.com/v2/droplets/$DROPLET_ID/actions" \
              -d '{"type":"power_on"}' | jq '.'
            
            echo "Power-on initiated. It may take a minute for the droplet to become available."
            echo "Waiting 60 seconds before retrying connection..."
            sleep 60
          fi
        fi
      else
        echo "✅ Droplet is active and should be accepting connections"
      fi
      
      # Check if firewall is enabled
      FIREWALL_IDS=$(echo "$DROPLET_INFO" | jq -r '.firewall_ids // []')
      if [ -n "$FIREWALL_IDS" ] && [ "$FIREWALL_IDS" != "[]" ]; then
        echo "⚠️ Firewall is enabled on this droplet. Checking if SSH is allowed..."
        for FW_ID in $(echo "$FIREWALL_IDS" | jq -r '.[]'); do
          FIREWALL_INFO=$(curl -s -X GET \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer $DIGITALOCEAN_TOKEN" \
            "https://api.digitalocean.com/v2/firewalls/$FW_ID")
          
          # Check if SSH port 22 is allowed
          SSH_ALLOWED=$(echo "$FIREWALL_INFO" | jq -r '.firewall.inbound_rules[] | select(.ports=="22") // empty')
          if [ -n "$SSH_ALLOWED" ]; then
            echo "✅ SSH port 22 is allowed in firewall rules"
          else
            echo "❌ SSH port 22 may not be allowed in firewall rules!"
            echo "Please check and modify firewall rules through the DigitalOcean control panel"
          fi
        done
      else
        echo "✅ No firewall is active on this droplet"
      fi
    else
      echo "❌ No droplet found with IP $TARGET_IP"
      echo "Please verify the IP address is correct and belongs to your DigitalOcean account"
    fi
  else
    echo "Skipping droplet status check - DigitalOcean API token not provided"
  fi
}

# Function to check and fix network issues
advanced_network_diagnostics() {
  echo "=== ADVANCED NETWORK DIAGNOSTICS ==="
  
  # Try traceroute with more detailed options
  echo "Running detailed traceroute to detect network path issues:"
  traceroute -T -p 22 -m 20 $TARGET_IP || echo "Traceroute failed"
  
  # Check if there might be MTU issues
  echo "Testing for potential MTU issues:"
  ping -c 3 -M do -s 1400 $TARGET_IP || echo "MTU test failed - possible network fragmentation issues"
  
  # Check DNS resolution
  echo "Checking DNS resolution:"
  host $TARGET_IP || echo "Reverse DNS lookup failed"
  
  # Test GitHub Actions network connectivity to common services
  echo "Testing outbound connectivity from GitHub Actions runner:"
  curl -s https://www.google.com > /dev/null && echo "✅ Connection to Google successful" || echo "❌ Connection to Google failed"
  curl -s https://api.digitalocean.com > /dev/null && echo "✅ Connection to DigitalOcean API successful" || echo "❌ Connection to DigitalOcean API failed"
}

# Try alternative SSH connection methods
try_alternative_ssh_methods() {
  echo "=== TRYING ALTERNATIVE SSH METHODS ==="
  
  # First ensure the SSH key is properly set up
  if [ ! -f ~/.ssh/id_bn_trading ]; then
    echo "❌ SSH key not found at ~/.ssh/id_bn_trading"
    return
  fi
  
  echo "Testing SSH connection with alternative timeout settings:"
  # Try with explicit connect timeout and server alive settings
  timeout 30 ssh -v -o ConnectTimeout=20 -o ServerAliveInterval=5 -o ServerAliveCountMax=3 -o StrictHostKeyChecking=no -i ~/.ssh/id_bn_trading root@$TARGET_IP echo "SSH TEST" 2>&1 || echo "Alternative SSH config failed"
  
  echo "Testing SSH connection with alternative port:"
  # Try with port 2222
  timeout 30 ssh -v -p 2222 -o ConnectTimeout=15 -o StrictHostKeyChecking=no -i ~/.ssh/id_bn_trading root@$TARGET_IP echo "SSH TEST" 2>&1 || echo "Alternative port failed"
  
  # Try with IPv6 if available
  if command -v ip &> /dev/null && ip -6 addr show &> /dev/null; then
    echo "Testing if target has IPv6 connectivity:"
    host -t AAAA $TARGET_IP &> /dev/null && echo "IPv6 address found, trying IPv6 connection" || echo "No IPv6 address found"
  fi
}

# Generate recommendations based on findings
generate_recommendations() {
  echo "=== RECOMMENDATIONS ==="
  echo "Based on the diagnostics, here are possible solutions:"
  
  if [ "$PORT_OPEN" = "true" ]; then
    echo "1. Port 22 appears to be open, but SSH connection still times out."
    echo "   Possible causes:"
    echo "   - SSH service may not be running on the droplet"
    echo "   - Firewall may be blocking the connection at a different level"
    echo "   - Network routing issues between GitHub Actions and DigitalOcean"
  else
    echo "1. Port 22 appears to be closed or filtered."
    echo "   Possible fixes:"
    echo "   - Ensure the droplet is powered on and running"
    echo "   - Check DigitalOcean firewall settings to allow port 22"
    echo "   - Verify SSH service is running on the droplet"
  fi
  
  echo "2. Try restarting the droplet through the DigitalOcean control panel"
  echo "3. Verify the SSH keys are properly set up on the droplet"
  echo "4. Check if there are network restrictions in DigitalOcean or GitHub Actions"
  echo "5. As a last resort, recreate the droplet with Terraform"
}

# Main function to coordinate fixes
main() {
  # Run all diagnostic functions
  check_basic_connectivity
  check_droplet_status
  advanced_network_diagnostics
  try_alternative_ssh_methods
  generate_recommendations
  
  echo "===== NETWORK DIAGNOSTICS COMPLETED ====="
}

# Run the main function
main
