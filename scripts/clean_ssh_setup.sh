#!/bin/bash

# Simple and Clean SSH Connection Troubleshooter
# This script checks and fixes SSH connectivity issues to a DigitalOcean droplet

set -e  # Exit on any error

# Colors for better readability
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}   SSH Connection Troubleshooter       ${NC}"
echo -e "${BLUE}=======================================${NC}"

# Server details
SERVER_IP="152.42.157.29"
SSH_USER="root"
SSH_PORT=22

# Step 1: Verify server is reachable
echo -e "\n${YELLOW}Step 1: Checking server reachability...${NC}"
if ping -c 3 $SERVER_IP > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Server is reachable via ping${NC}"
else
    echo -e "${RED}✗ Server is not responding to ping${NC}"
    echo -e "${YELLOW}  This could be normal if ICMP is blocked. Continuing...${NC}"
fi

# Step 2: Check if SSH port is open
echo -e "\n${YELLOW}Step 2: Checking if SSH port is open...${NC}"
if nc -z -w 5 $SERVER_IP $SSH_PORT > /dev/null 2>&1; then
    echo -e "${GREEN}✓ SSH port $SSH_PORT is open${NC}"
else
    echo -e "${RED}✗ SSH port $SSH_PORT is closed or blocked${NC}"
    echo -e "${RED}  Check firewall settings on the server${NC}"
    exit 1
fi

# Step 3: Check for existing SSH keys
echo -e "\n${YELLOW}Step 3: Checking for existing SSH keys...${NC}"
if ls -l ~/.ssh/id_* 2>/dev/null | grep -v ".pub" > /dev/null; then
    echo -e "${GREEN}✓ Existing SSH keys found:${NC}"
    ls -l ~/.ssh/id_* 2>/dev/null | grep -v ".pub"
else
    echo -e "${YELLOW}! No SSH keys found in ~/.ssh/${NC}"
    echo -e "${YELLOW}  Let's generate a new key pair${NC}"
fi

# Step 4: Create a clean SSH key specifically for DigitalOcean
echo -e "\n${YELLOW}Step 4: Creating a fresh SSH key for DigitalOcean...${NC}"
SSH_KEY_FILE="$HOME/.ssh/id_do_root"

# Backup any existing key with this name
if [ -f "$SSH_KEY_FILE" ]; then
    echo -e "${YELLOW}  Backing up existing key...${NC}"
    mv "$SSH_KEY_FILE" "$SSH_KEY_FILE.bak.$(date +%s)"
    mv "$SSH_KEY_FILE.pub" "$SSH_KEY_FILE.pub.bak.$(date +%s)" 2>/dev/null || true
fi

# Generate a fresh key
echo -e "${YELLOW}  Generating new ED25519 key (more secure and compatible)...${NC}"
ssh-keygen -t ed25519 -f "$SSH_KEY_FILE" -N "" -C "root@digitalocean-$(date +%Y%m%d)"

# Set proper permissions
chmod 600 "$SSH_KEY_FILE"
chmod 644 "$SSH_KEY_FILE.pub"

# Display the public key
echo -e "\n${BLUE}Here's the public key that should be added to DigitalOcean:${NC}"
cat "$SSH_KEY_FILE.pub"
echo -e "\n"

# Step 5: Verify the key format
echo -e "\n${YELLOW}Step 5: Verifying key format...${NC}"
if ssh-keygen -l -f "$SSH_KEY_FILE" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Key format is valid${NC}"
    echo -e "${BLUE}  Key fingerprint: $(ssh-keygen -l -f $SSH_KEY_FILE | awk '{print $2}')${NC}"
else
    echo -e "${RED}✗ Key format is invalid${NC}"
    exit 1
fi

# Step 6: Configure SSH config file
echo -e "\n${YELLOW}Step 6: Configuring SSH config...${NC}"
CONFIG_FILE="$HOME/.ssh/config"

# Create or update SSH config
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${YELLOW}  Creating new SSH config file${NC}"
    touch "$CONFIG_FILE"
    chmod 600 "$CONFIG_FILE"
fi

# Check if entry exists and update/create as needed
if grep -q "Host digitalocean" "$CONFIG_FILE"; then
    echo -e "${YELLOW}  Updating existing 'digitalocean' entry in SSH config${NC}"
    sed -i '' '/Host digitalocean/,/^$/d' "$CONFIG_FILE"
fi

# Add the new entry
echo -e "${YELLOW}  Adding digitalocean entry to SSH config${NC}"
cat << EOF >> "$CONFIG_FILE"
# DigitalOcean Trading Server
Host digitalocean
    HostName $SERVER_IP
    User $SSH_USER
    IdentityFile $SSH_KEY_FILE
    StrictHostKeyChecking accept-new
    PubkeyAcceptedKeyTypes +ssh-ed25519
    ServerAliveInterval 60

EOF

echo -e "${GREEN}✓ SSH config updated${NC}"

# Step 7: Remove any cached keys for this IP
echo -e "\n${YELLOW}Step 7: Cleaning up known_hosts cache...${NC}"
ssh-keygen -R $SERVER_IP > /dev/null 2>&1 || true
echo -e "${GREEN}✓ Known hosts cache cleaned${NC}"

# Step 8: Test connection with the new key
echo -e "\n${YELLOW}Step 8: Testing SSH connection...${NC}"
echo -e "${YELLOW}  Attempting to connect to $SERVER_IP with the new key...${NC}"
echo -e "${YELLOW}  (This may fail if you haven't added the key to DigitalOcean yet)${NC}"

# Save the output to a variable to analyze
SSH_RESULT=$(ssh -i "$SSH_KEY_FILE" -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$SSH_USER@$SERVER_IP" "echo Connection successful" 2>&1 || true)

if echo "$SSH_RESULT" | grep -q "Connection successful"; then
    echo -e "${GREEN}✓ SSH connection successful!${NC}"
    
    # Verify Docker status
    echo -e "\n${YELLOW}Step 9: Checking Docker status...${NC}"
    DOCKER_VERSION=$(ssh -i "$SSH_KEY_FILE" "$SSH_USER@$SERVER_IP" "docker --version 2>/dev/null || echo 'not installed'")
    echo -e "${GREEN}  Docker: $DOCKER_VERSION${NC}"
    
    # Check if port 5004 is open
    echo -e "\n${YELLOW}Step 10: Checking port 5004...${NC}"
    PORT_STATUS=$(ssh -i "$SSH_KEY_FILE" "$SSH_USER@$SERVER_IP" "netstat -tuln | grep 5004 || echo 'Port 5004 not in use'")
    echo -e "${GREEN}  Port 5004: $PORT_STATUS${NC}"
    
    echo -e "\n${GREEN}✓ SSH connection is working perfectly!${NC}"
    echo -e "${GREEN}  Now you can use this key for your deployment scripts.${NC}"
    echo -e "${GREEN}  SSH Command: ssh -i $SSH_KEY_FILE $SSH_USER@$SERVER_IP${NC}"
    echo -e "${GREEN}  Or simply: ssh digitalocean${NC}"
    
    # Update the fix_deployment.sh script
    echo -e "\n${YELLOW}Update your deployment scripts? (y/n)${NC}"
    read -p ">>> " UPDATE_SCRIPTS
    
    if [[ $UPDATE_SCRIPTS == "y" ]]; then
        echo -e "${YELLOW}Updating deployment scripts...${NC}"
        
        # Fix deployment script
        DEPLOY_SCRIPT="/Users/dharanyamahesh/Documents/GitHub/autotrade/scripts/fix_deployment.sh"
        if [ -f "$DEPLOY_SCRIPT" ]; then
            echo -e "${YELLOW}Updating $DEPLOY_SCRIPT${NC}"
            sed -i '' "s|SSH_KEY_PATH=.*|SSH_KEY_PATH=\"$SSH_KEY_FILE\"|" "$DEPLOY_SCRIPT" 2>/dev/null || true
            echo -e "${GREEN}✓ Updated $DEPLOY_SCRIPT${NC}"
        fi
        
        # Fix other deployment scripts
        SCRIPTS_DIR="/Users/dharanyamahesh/Documents/GitHub/autotrade/scripts"
        find "$SCRIPTS_DIR" -type f -name "*.sh" -exec grep -l "SSH_KEY_PATH=" {} \; | while read file; do
            echo -e "${YELLOW}Updating $file${NC}"
            sed -i '' "s|SSH_KEY_PATH=.*|SSH_KEY_PATH=\"$SSH_KEY_FILE\"|" "$file" 2>/dev/null || true
            echo -e "${GREEN}✓ Updated $file${NC}"
        done
    fi
    
else
    echo -e "${RED}✗ SSH connection failed${NC}"
    echo -e "${RED}  Error: $SSH_RESULT${NC}"
    
    if echo "$SSH_RESULT" | grep -q "Permission denied"; then
        echo -e "${YELLOW}! Authentication issue: The key hasn't been added to DigitalOcean${NC}"
        echo -e "${YELLOW}  Please add this public key to your DigitalOcean droplet:${NC}"
        cat "$SSH_KEY_FILE.pub"
        
        echo -e "\n${YELLOW}Important: After adding the key to DigitalOcean, wait a minute for it to propagate${NC}"
        echo -e "${YELLOW}Then run this command to test the connection:${NC}"
        echo -e "${GREEN}  ssh -i $SSH_KEY_FILE $SSH_USER@$SERVER_IP${NC}"
    fi
fi

echo -e "\n${BLUE}=======================================${NC}"
echo -e "${BLUE}   SSH Connection Setup Complete       ${NC}"
echo -e "${BLUE}=======================================${NC}"
