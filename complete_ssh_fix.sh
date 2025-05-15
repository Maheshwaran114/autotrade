#!/bin/bash
# COMPLETE SSH CONNECTIVITY FIX FOR GITHUB ACTIONS
# This script combines all the SSH connectivity fixes into one script

set -e

# Display script start info
echo "===== COMPLETE SSH CONNECTIVITY FIX TOOL ====="
echo "Date/Time: $(date)"
echo "User: $(whoami)"
echo "Working Directory: $(pwd)"

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the target IP (required)
TARGET_IP="${1}"

if [ -z "$TARGET_IP" ]; then
  echo -e "${RED}ERROR: No target IP address provided.${NC}"
  echo "Usage: $0 <target-ip-address> [digitalocean-token]"
  exit 1
fi

DIGITALOCEAN_TOKEN="${2}"

# Create diagnostic directory
DIAG_DIR="./ssh_diagnostics_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DIAG_DIR"
echo -e "${GREEN}Created diagnostic directory: $DIAG_DIR${NC}"

# 1. Fix SSH configuration
echo -e "${BLUE}1. Fixing SSH configuration...${NC}"

# Create SSH directory if it doesn't exist
if [ ! -d ~/.ssh ]; then
  echo "Creating SSH directory..."
  mkdir -p ~/.ssh
  chmod 700 ~/.ssh
  echo -e "${GREEN}✅ Created SSH directory with proper permissions${NC}"
else
  echo -e "${GREEN}✅ SSH directory exists${NC}"
  ls -la ~/.ssh
fi

# Check SSH config file and fix issues
if [ -f ~/.ssh/config ]; then
  echo "SSH config exists. Backing up and checking for issues..."
  cp ~/.ssh/config "$DIAG_DIR/ssh_config.backup"
  
  # Fix incorrect UserKnownHostsFile path
  if grep -q "UserKnownHostsFile /null" ~/.ssh/config; then
    echo -e "${RED}❌ Found incorrect 'UserKnownHostsFile /null' in SSH config${NC}"
    echo "Fixing the issue..."
    sed -i.bak 's|UserKnownHostsFile /null|UserKnownHostsFile /dev/null|g' ~/.ssh/config
    echo -e "${GREEN}✅ Fixed UserKnownHostsFile setting${NC}"
    echo "Before fix:"
    grep "UserKnownHostsFile" "$DIAG_DIR/ssh_config.backup"
    echo "After fix:"
    grep "UserKnownHostsFile" ~/.ssh/config
  else
    echo -e "${GREEN}✅ No incorrect UserKnownHostsFile found${NC}"
  fi
  
  # Make sure key settings are present
  if ! grep -q "StrictHostKeyChecking" ~/.ssh/config; then
    echo -e "${YELLOW}⚠️ Adding StrictHostKeyChecking no to config...${NC}"
    echo "StrictHostKeyChecking no" >> ~/.ssh/config
  fi
  
  # Ensure ServerAlive settings for persistent connections
  if ! grep -q "ServerAliveInterval" ~/.ssh/config; then
    echo -e "${YELLOW}⚠️ Adding ServerAliveInterval to config...${NC}"
    echo "ServerAliveInterval 30" >> ~/.ssh/config
  fi
  
  if ! grep -q "ServerAliveCountMax" ~/.ssh/config; then
    echo -e "${YELLOW}⚠️ Adding ServerAliveCountMax to config...${NC}"
    echo "ServerAliveCountMax 6" >> ~/.ssh/config
  fi
  
  # Set proper permissions
  chmod 600 ~/.ssh/config
  echo -e "${GREEN}✅ Set SSH config permissions to 600${NC}"
else
  echo -e "${YELLOW}⚠️ No SSH config file found. Creating a new one with optimal settings...${NC}"
  
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
  TCPKeepAlive yes
  AddressFamily inet
  IPQoS throughput
EOF
  chmod 600 ~/.ssh/config
  echo -e "${GREEN}✅ Created new SSH config with proper settings${NC}"
fi

echo -e "${BLUE}Current SSH config:${NC}"
cat ~/.ssh/config | tee "$DIAG_DIR/ssh_config.current"

# 2. Check SSH key files
echo -e "\n${BLUE}2. Verifying SSH key files...${NC}"

# Check if the key file exists
if [ -f ~/.ssh/id_bn_trading ]; then
  echo -e "${GREEN}✅ SSH key exists at ~/.ssh/id_bn_trading${NC}"
  
  # Make a redacted copy for diagnostics
  head -1 ~/.ssh/id_bn_trading > "$DIAG_DIR/key_first_line.txt"
  
  # Check key format (without revealing sensitive data)
  if head -1 ~/.ssh/id_bn_trading | grep -q "BEGIN"; then
    echo -e "${GREEN}✅ Key appears to be in the correct format${NC}"
    
    # Save the key info (not the key itself)
    ssh-keygen -l -f ~/.ssh/id_bn_trading > "$DIAG_DIR/key_fingerprint.txt" 2>/dev/null || echo -e "${RED}❌ Could not get key fingerprint${NC}"
  else
    echo -e "${RED}❌ Key does not appear to be in the correct format!${NC}"
    echo "The key should begin with -----BEGIN OPENSSH PRIVATE KEY----- or -----BEGIN RSA PRIVATE KEY-----"
    echo "First few characters (redacted):"
    head -c 10 ~/.ssh/id_bn_trading | tr -c '[:print:]' '?' | sed 's/./?/g'
  fi
  
  # Ensure proper permissions
  chmod 600 ~/.ssh/id_bn_trading
  echo -e "${GREEN}✅ Set key permissions to 600${NC}"
else
  echo -e "${RED}❌ SSH key not found at ~/.ssh/id_bn_trading${NC}"
  echo "Make sure the SSH_PRIVATE_KEY secret is properly set in GitHub"
fi

# 3. Network connectivity tests
echo -e "\n${BLUE}3. Running network connectivity tests to ${TARGET_IP}...${NC}"

# Basic connectivity test
echo "Running basic connectivity test (ping)..."
ping -c 3 "${TARGET_IP}" > "$DIAG_DIR/ping_test.txt" 2>&1 || echo -e "${YELLOW}⚠️ Ping failed - this may be expected if ICMP is blocked${NC}"

# TCP connection test to SSH port
echo "Testing TCP connection to SSH port (22)..."
timeout 10 bash -c "cat < /dev/null > /dev/tcp/${TARGET_IP}/22" 2> "$DIAG_DIR/tcp_test.txt"
if [ $? -eq 0 ]; then
  echo -e "${GREEN}✅ TCP connection to SSH port successful${NC}"
else
  echo -e "${RED}❌ TCP connection to SSH port failed${NC}"
  echo "This could indicate a firewall blocking SSH access from GitHub Actions IP ranges"
fi

# 4. Enhanced SSH connection tests
echo -e "\n${BLUE}4. Testing SSH connections with different parameters...${NC}"

# Basic SSH connection test
echo "Test 1: Basic SSH connection..."
ssh -v -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -i ~/.ssh/id_bn_trading root@${TARGET_IP} exit > "$DIAG_DIR/ssh_test_1.txt" 2>&1
if [ $? -eq 0 ]; then
  echo -e "${GREEN}✅ Basic SSH connection successful${NC}"
  SSH_WORKS=true
else
  echo -e "${RED}❌ Basic SSH connection failed${NC}"
  SSH_WORKS=false
fi

# Try alternative connection methods if basic fails
if [ "$SSH_WORKS" = "false" ]; then
  echo "Test 2: SSH with alternative key exchange algorithms..."
  ssh -v -o KexAlgorithms=curve25519-sha256,diffie-hellman-group-exchange-sha256 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -i ~/.ssh/id_bn_trading root@${TARGET_IP} exit > "$DIAG_DIR/ssh_test_2.txt" 2>&1
  if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ SSH with alternative algorithms successful${NC}"
    SSH_WORKS=true
  else
    echo -e "${RED}❌ SSH with alternative algorithms failed${NC}"
  fi
fi

if [ "$SSH_WORKS" = "false" ]; then
  echo "Test 3: SSH with extended timeout and debug..."
  ssh -vvv -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=60 -o ServerAliveInterval=10 -o ServerAliveCountMax=6 -i ~/.ssh/id_bn_trading root@${TARGET_IP} exit > "$DIAG_DIR/ssh_test_3.txt" 2>&1
  if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ SSH with extended timeout successful${NC}"
    SSH_WORKS=true
  else
    echo -e "${RED}❌ SSH with extended timeout failed${NC}"
  fi
fi

# 5. Check DigitalOcean Droplet Status if token is provided
if [ -n "$DIGITALOCEAN_TOKEN" ]; then
  echo -e "\n${BLUE}5. Checking DigitalOcean droplet status...${NC}"
  
  # Install curl and jq if needed
  which curl > /dev/null || { echo "Installing curl..."; apt-get update && apt-get install -y curl; }
  which jq > /dev/null || { echo "Installing jq..."; apt-get update && apt-get install -y jq; }
  
  # Get droplet status
  echo "Querying DigitalOcean API for droplet status..."
  DROPLET_STATUS=$(curl -s -X GET \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $DIGITALOCEAN_TOKEN" \
    "https://api.digitalocean.com/v2/droplets?per_page=200" | \
    jq -r ".droplets[] | select(.networks.v4[].ip_address==\"$TARGET_IP\") | .status" 2>/dev/null)
  
  echo "Droplet status: ${DROPLET_STATUS:-unknown}"
  echo "Droplet status: ${DROPLET_STATUS:-unknown}" > "$DIAG_DIR/droplet_status.txt"
  
  # Try to power on the droplet if it's off
  if [ "$DROPLET_STATUS" = "off" ]; then
    echo -e "${RED}❌ Droplet is powered off! Attempting to power it on...${NC}"
    
    # First find the droplet ID
    DROPLET_ID=$(curl -s -X GET \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $DIGITALOCEAN_TOKEN" \
      "https://api.digitalocean.com/v2/droplets?per_page=200" | \
      jq -r ".droplets[] | select(.networks.v4[].ip_address==\"$TARGET_IP\") | .id" 2>/dev/null)
    
    if [ -n "$DROPLET_ID" ]; then
      echo "Found droplet ID: $DROPLET_ID. Powering on..."
      
      # Send power on request
      POWER_RESULT=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $DIGITALOCEAN_TOKEN" \
        "https://api.digitalocean.com/v2/droplets/$DROPLET_ID/actions" \
        -d '{"type":"power_on"}')
      
      echo "$POWER_RESULT" > "$DIAG_DIR/power_on_result.json"
      echo -e "${YELLOW}⚠️ Power on request sent. Waiting for droplet to boot...${NC}"
      
      # Wait for the droplet to start
      echo "Waiting up to 60 seconds for the droplet to boot..."
      for i in {1..12}; do
        echo "Checking boot status (attempt $i/12)..."
        sleep 5
        
        # Check droplet status again
        CURRENT_STATUS=$(curl -s -X GET \
          -H "Content-Type: application/json" \
          -H "Authorization: Bearer $DIGITALOCEAN_TOKEN" \
          "https://api.digitalocean.com/v2/droplets/$DROPLET_ID" | \
          jq -r ".droplet.status" 2>/dev/null)
        
        echo "Current status: $CURRENT_STATUS"
        
        if [ "$CURRENT_STATUS" = "active" ]; then
          echo -e "${GREEN}✅ Droplet is now active!${NC}"
          break
        fi
      done
      
      # Final status check
      FINAL_STATUS=$(curl -s -X GET \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $DIGITALOCEAN_TOKEN" \
        "https://api.digitalocean.com/v2/droplets/$DROPLET_ID" | \
        jq -r ".droplet.status" 2>/dev/null)
      
      if [ "$FINAL_STATUS" = "active" ]; then
        echo -e "${GREEN}✅ Droplet is now active. Waiting 30 seconds for services to start...${NC}"
        sleep 30
        
        # Try SSH again after droplet is powered on
        echo "Testing SSH connection after power on..."
        ssh -v -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=30 -i ~/.ssh/id_bn_trading root@${TARGET_IP} exit > "$DIAG_DIR/ssh_after_poweron.txt" 2>&1
        if [ $? -eq 0 ]; then
          echo -e "${GREEN}✅ SSH connection successful after power on!${NC}"
          SSH_WORKS=true
        else
          echo -e "${RED}❌ SSH connection still failing after power on${NC}"
        fi
      else
        echo -e "${RED}❌ Droplet power on failed or timed out. Status: $FINAL_STATUS${NC}"
      fi
    else
      echo -e "${RED}❌ Could not find droplet ID for IP: $TARGET_IP${NC}"
    fi
  elif [ "$DROPLET_STATUS" = "active" ]; then
    echo -e "${GREEN}✅ Droplet is active${NC}"
  else
    echo -e "${YELLOW}⚠️ Droplet status is: ${DROPLET_STATUS:-unknown}${NC}"
  fi
else
  echo -e "${YELLOW}⚠️ No DigitalOcean token provided. Skipping droplet status check.${NC}"
fi

# 6. Deploy SSH validation script
if [ "$SSH_WORKS" = "true" ]; then
  echo -e "\n${BLUE}6. Deploying SSH validation script to server...${NC}"
  
  # Create the validation script
  cat > /tmp/validate_ssh_service.sh << 'EOF'
#!/bin/bash
# SSH Service Validator
echo "===== SSH SERVICE VALIDATOR ====="
echo "Running on: $(hostname) at $(date)"

# Check SSH service
echo "=== SSH SERVICE STATUS ==="
systemctl status sshd || service ssh status

# Check SSH config
echo "=== SSH CONFIG ==="
grep -v "^#" /etc/ssh/sshd_config | grep -v "^$"

# Check authorized keys
echo "=== AUTHORIZED KEYS ==="
find /root/.ssh -type f -name "authorized_keys" -exec cat {} \; | wc -l
find /root/.ssh -type f -name "authorized_keys" -exec ls -la {} \;

# Check firewall
echo "=== FIREWALL STATUS ==="
ufw status || iptables -L | grep -i ssh

echo "===== VALIDATION COMPLETE ====="
EOF

  # Copy and execute on remote server
  echo "Copying validation script to server..."
  scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i ~/.ssh/id_bn_trading /tmp/validate_ssh_service.sh root@${TARGET_IP}:/root/ > "$DIAG_DIR/scp_result.txt" 2>&1
  
  if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Script copied successfully${NC}"
    
    echo "Running validation script on server..."
    ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i ~/.ssh/id_bn_trading root@${TARGET_IP} "chmod +x /root/validate_ssh_service.sh && /root/validate_ssh_service.sh" > "$DIAG_DIR/validation_result.txt" 2>&1
    
    if [ $? -eq 0 ]; then
      echo -e "${GREEN}✅ Validation script executed successfully${NC}"
      cat "$DIAG_DIR/validation_result.txt"
    else
      echo -e "${RED}❌ Failed to execute validation script${NC}"
    fi
  else
    echo -e "${RED}❌ Failed to copy validation script to server${NC}"
  fi
else
  echo -e "${YELLOW}⚠️ Skipping SSH validation script deployment as SSH connection failed${NC}"
fi

# 7. Final report
echo -e "\n${BLUE}7. Final connectivity report${NC}"

if [ "$SSH_WORKS" = "true" ]; then
  echo -e "${GREEN}✅ SSH CONNECTION SUCCESSFUL${NC}"
  echo "The fixes applied have resolved the SSH connectivity issues."
else
  echo -e "${RED}❌ SSH CONNECTION FAILED${NC}"
  echo "Recommended next steps:"
  echo "1. Check firewall rules on the DigitalOcean droplet"
  echo "2. Verify SSH service is running on the droplet"
  echo "3. Ensure GitHub Actions IP ranges are allowed in any security groups"
  echo "4. Check if the SSH key in GitHub secrets matches the authorized key on the droplet"
  echo "5. Verify network connectivity between GitHub Actions and DigitalOcean"
fi

echo -e "\n${BLUE}All diagnostics have been saved to: $DIAG_DIR${NC}"
echo -e "${GREEN}===== SSH CONNECTIVITY FIX COMPLETE =====${NC}"
