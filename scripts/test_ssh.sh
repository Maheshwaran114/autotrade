ssh-keygen -t ed25519 -f ~/.ssh/digitalocean_key -N ""#!/bin/bash

# Simple script to test SSH connectivity to DigitalOcean server
# Run this after adding your public key to the server

set -e  # Exit on any error

# Colors for better readability
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}SSH Connectivity Test${NC}"
echo -e "${BLUE}===================${NC}"

# Server information
SERVER_IP="152.42.157.29"
SSH_USER="root"
KEY_PATH="$HOME/.ssh/do_trading_key"  # Path to the newly generated key

# Check if the key exists
if [ ! -f "$KEY_PATH" ]; then
    echo -e "${RED}Error: SSH key not found at $KEY_PATH${NC}"
    echo -e "${YELLOW}Please run the reset_ssh.sh script first to generate the key.${NC}"
    exit 1
fi

echo -e "${YELLOW}Testing SSH connection to $SERVER_IP...${NC}"

# Set correct permissions for SSH key
chmod 600 "$KEY_PATH"

# Try to connect using the key
echo -e "${YELLOW}Attempting to connect using SSH key...${NC}"
if ssh -i "$KEY_PATH" -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$SSH_USER@$SERVER_IP" "echo Connection successful"; then
    echo -e "${GREEN}SUCCESS! SSH connection established.${NC}"
    
    # Check Docker status while we're connected
    echo -e "\n${YELLOW}Checking Docker status...${NC}"
    ssh -i "$KEY_PATH" "$SSH_USER@$SERVER_IP" "docker --version || echo 'Docker not installed'"
    
    # Check if port 5004 is accessible
    echo -e "\n${YELLOW}Checking port 5004...${NC}"
    ssh -i "$KEY_PATH" "$SSH_USER@$SERVER_IP" "nc -z -v localhost 5004 || echo 'Port 5004 not in use'"
    
    echo -e "\n${GREEN}SSH connectivity test complete. Everything is working properly!${NC}"
    echo -e "${YELLOW}You can now update your deployment scripts to use this key.${NC}"
    
    # Provide a quick command to update the scripts
    echo -e "\n${BLUE}To update your deployment scripts, run:${NC}"
    echo -e "${GREEN}find /Users/dharanyamahesh/Documents/GitHub/autotrade/scripts -type f -name \"*.sh\" -exec sed -i '' \"s|~/.ssh/[a-zA-Z0-9_]*|$KEY_PATH|g\" {} \\;${NC}"
else
    echo -e "${RED}SSH connection failed.${NC}"
    echo -e "${YELLOW}Possible reasons:${NC}"
    echo -e "1. The public key hasn't been added to the server yet"
    echo -e "2. The server is not reachable (check network)"
    echo -e "3. SSH service is not running on the server"
    
    # Check if server is reachable at all
    echo -e "\n${YELLOW}Checking if server is reachable...${NC}"
    if ping -c 3 "$SERVER_IP" > /dev/null 2>&1; then
        echo -e "${GREEN}Server is reachable (ping successful).${NC}"
        echo -e "${YELLOW}This suggests that the SSH key hasn't been properly added to the server.${NC}"
    else
        echo -e "${RED}Server is not responding to ping.${NC}"
        echo -e "${YELLOW}The server might be down or blocking ICMP requests.${NC}"
    fi
    
    # Check if SSH port is open
    echo -e "\n${YELLOW}Checking if SSH port is open...${NC}"
    if nc -z -w 5 "$SERVER_IP" 22 > /dev/null 2>&1; then
        echo -e "${GREEN}SSH port is open.${NC}"
        echo -e "${YELLOW}The server is accepting SSH connections but rejecting your key.${NC}"
    else
        echo -e "${RED}SSH port is closed or filtered.${NC}"
        echo -e "${YELLOW}The server might have SSH disabled or there's a firewall blocking port 22.${NC}"
    fi
    
    exit 1
fi
