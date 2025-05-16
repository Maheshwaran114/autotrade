#!/bin/bash
# SSH Connection Debugging Script
# This script helps diagnose SSH connection issues in DigitalOcean deployments

# Text formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color
BOLD='\033[1m'

echo -e "${BOLD}SSH Connection Debugging Tool${NC}\n"

# Checking for required parameters
if [ $# -lt 1 ]; then
    echo -e "${RED}Error: Missing required parameters.${NC}"
    echo -e "Usage: $0 <droplet_ip> [private_key_file]"
    echo -e "  droplet_ip      - IP address of the DigitalOcean droplet"
    echo -e "  private_key_file - Path to the SSH private key file (optional if using ssh-agent)"
    exit 1
fi

DROPLET_IP=$1
SSH_KEY_FILE=$2
SSH_USERNAME="root" # Default username for DigitalOcean droplets

echo -e "${BOLD}Step 1: Checking network connectivity${NC}"
echo -e "Testing connection to ${DROPLET_IP}..."

# Check if host is reachable via ping
if ping -c 3 -W 5 ${DROPLET_IP} > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Host is reachable via ping${NC}"
else
    echo -e "${YELLOW}⚠ Unable to ping host (this may be normal if ICMP is blocked)${NC}"
fi

# Check if SSH port is open
if nc -z -w 5 ${DROPLET_IP} 22 > /dev/null 2>&1; then
    echo -e "${GREEN}✓ SSH port (22) is open${NC}"
else
    echo -e "${RED}✗ SSH port (22) is closed or blocked${NC}"
    echo -e "Please check your firewall settings and ensure SSH is enabled on the droplet."
    exit 1
fi

echo -e "\n${BOLD}Step 2: Checking SSH key format${NC}"

# If key file provided, check its format
if [ ! -z "$SSH_KEY_FILE" ]; then
    if [ ! -f "$SSH_KEY_FILE" ]; then
        echo -e "${RED}✗ SSH key file does not exist: $SSH_KEY_FILE${NC}"
        exit 1
    fi
    
    echo -e "Checking private key format..."
    
    # Check header/footer
    if grep -q "BEGIN OPENSSH PRIVATE KEY" "$SSH_KEY_FILE" && grep -q "END OPENSSH PRIVATE KEY" "$SSH_KEY_FILE"; then
        echo -e "${GREEN}✓ SSH key appears to be in OpenSSH format${NC}"
    elif grep -q "BEGIN RSA PRIVATE KEY" "$SSH_KEY_FILE" && grep -q "END RSA PRIVATE KEY" "$SSH_KEY_FILE"; then
        echo -e "${GREEN}✓ SSH key appears to be in PEM/RSA format${NC}"
    else
        echo -e "${RED}✗ SSH key file does not appear to have valid header/footer${NC}"
        echo -e "Make sure your key has proper BEGIN/END markers and is not corrupted."
    fi
    
    # Check permissions
    KEY_PERMS=$(stat -c "%a" "$SSH_KEY_FILE" 2>/dev/null || stat -f "%Lp" "$SSH_KEY_FILE" 2>/dev/null)
    if [ "$KEY_PERMS" = "600" ] || [ "$KEY_PERMS" = "400" ]; then
        echo -e "${GREEN}✓ SSH key file has correct permissions${NC}"
    else
        echo -e "${YELLOW}⚠ SSH key file has incorrect permissions: $KEY_PERMS (should be 600)${NC}"
        echo -e "Run: chmod 600 $SSH_KEY_FILE"
    fi
    
    # Try to extract public key
    echo -e "\nAttempting to extract public key from private key..."
    PUB_KEY=$(ssh-keygen -y -f "$SSH_KEY_FILE" 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Successfully extracted public key:${NC}"
        echo "$PUB_KEY"
        echo -e "\nVerify this matches the key uploaded to DigitalOcean console."
    else
        echo -e "${RED}✗ Failed to extract public key. The key might be invalid or using an unsupported format.${NC}"
    fi
    
    SSH_OPTS="-i $SSH_KEY_FILE"
else
    echo -e "${YELLOW}⚠ No SSH key file provided, will try using ssh-agent or default keys${NC}"
    SSH_OPTS=""
fi

echo -e "\n${BOLD}Step 3: Testing SSH connection with verbose output${NC}"
echo -e "Attempting to connect to ${SSH_USERNAME}@${DROPLET_IP}..."

# Try a test connection with maximum verbosity
echo -e "Running: ssh ${SSH_OPTS} -vvv ${SSH_USERNAME}@${DROPLET_IP} 'echo Connection successful'"
SSH_OUTPUT=$(ssh ${SSH_OPTS} -vvv -o BatchMode=yes -o ConnectTimeout=10 ${SSH_USERNAME}@${DROPLET_IP} 'echo Connection successful' 2>&1)
SSH_STATUS=$?

echo -e "\n${BOLD}Connection test output:${NC}"
echo "$SSH_OUTPUT"

if [ $SSH_STATUS -eq 0 ]; then
    echo -e "\n${GREEN}✓ SSH connection successful!${NC}"
else
    echo -e "\n${RED}✗ SSH connection failed with exit code: $SSH_STATUS${NC}"
    
    # Check for common error patterns
    if echo "$SSH_OUTPUT" | grep -q "Permission denied (publickey)"; then
        echo -e "${YELLOW}⚠ Authentication failure: Permission denied (publickey)${NC}"
        echo -e "Possible causes:"
        echo -e "  1. The private key doesn't match any authorized key on the server"
        echo -e "  2. The SSH key in DigitalOcean console doesn't match your local key"
        echo -e "  3. The authorized_keys file permissions on the server are incorrect"
    elif echo "$SSH_OUTPUT" | grep -q "Host key verification failed"; then
        echo -e "${YELLOW}⚠ Host key verification failed${NC}"
        echo -e "Run: ssh-keygen -R ${DROPLET_IP}"
    elif echo "$SSH_OUTPUT" | grep -q "Connection timed out"; then
        echo -e "${YELLOW}⚠ Connection timed out${NC}"
        echo -e "Check firewall settings on both client and server sides"
    fi
fi

echo -e "\n${BOLD}Recommendations:${NC}"
echo -e "1. Verify the SSH key ID in terraform.tfvars matches the one in DigitalOcean console"
echo -e "2. If using GitHub Actions, make sure the SSH_PRIVATE_KEY secret:"
echo -e "   - Has the correct format (including BEGIN/END lines)"
echo -e "   - Matches the public key registered in DigitalOcean"
echo -e "3. Try creating a new SSH key pair and updating both DigitalOcean and GitHub Secrets:"
echo -e "   ssh-keygen -t rsa -b 4096 -f ./do_deployment_key"
echo -e "   # Add do_deployment_key.pub to DigitalOcean console"
echo -e "   # Add content of do_deployment_key to GitHub Secrets"
echo -e "4. For GitHub Actions ssh-action, try adding these options:"
echo -e "   - use_insecure_cipher: true"
echo -e "   - key_type: rsa"

exit $SSH_STATUS
