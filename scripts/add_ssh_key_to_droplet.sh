#!/bin/bash
# Script to add SSH key directly to a droplet using the DigitalOcean API
# This is a workaround for when the SSH key fingerprint approach isn't working

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color
BOLD='\033[1m'

echo -e "${BOLD}DigitalOcean Direct SSH Key Installer${NC}\n"

# Check arguments
if [ $# -lt 3 ]; then
    echo -e "${RED}Error: Missing required parameters.${NC}"
    echo -e "Usage: $0 <do_token> <droplet_ip> <ssh_pubkey_file>"
    echo -e "  do_token        - Your DigitalOcean API token"
    echo -e "  droplet_ip      - IP address of the droplet to connect to"
    echo -e "  ssh_pubkey_file - Path to the SSH public key file to add"
    exit 1
fi

DO_TOKEN="$1"
DROPLET_IP="$2"
SSH_PUBKEY_FILE="$3"

if [ ! -f "$SSH_PUBKEY_FILE" ]; then
    echo -e "${RED}Error: SSH public key file not found: $SSH_PUBKEY_FILE${NC}"
    exit 1
fi

# Get the public key content
SSH_PUBKEY=$(cat "$SSH_PUBKEY_FILE")
if [ -z "$SSH_PUBKEY" ]; then
    echo -e "${RED}Error: SSH public key file is empty: $SSH_PUBKEY_FILE${NC}"
    exit 1
fi

echo -e "Checking for droplet with IP: ${DROPLET_IP}..."

# Get the droplet ID from the IP
DROPLETS_RESPONSE=$(curl -s -X GET \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $DO_TOKEN" \
    "https://api.digitalocean.com/v2/droplets?per_page=200")

# Check if API call was successful
if echo "$DROPLETS_RESPONSE" | grep -q "message.*unauthorized"; then
    echo -e "${RED}Error: API access denied. Please check your DO_TOKEN.${NC}"
    exit 1
fi

# Try to extract the droplet ID
DROPLET_ID=$(echo "$DROPLETS_RESPONSE" | grep -o '"id":[0-9]*,"name":"[^"]*","memory":[0-9]*,"vcpus":[0-9]*,"disk":[0-9]*,"locked":\w*,"status":"active","kernel":null,"created_at":"[^"]*","features":\[[^]]*\],"backup_ids":\[\],"next_backup_window":null,"snapshot_ids":\[\],"image":{[^}]*},"volume_ids":\[\],"size":{[^}]*},"size_slug":"[^"]*","networks":{"v4":\[{"ip_address":"'"$DROPLET_IP"'"' | grep -o '"id":[0-9]*' | head -1 | cut -d':' -f2)

if [ -z "$DROPLET_ID" ]; then
    echo -e "${RED}Error: Could not find a droplet with IP: $DROPLET_IP${NC}"
    exit 1
fi

echo -e "${GREEN}Found droplet with ID: $DROPLET_ID${NC}"

# Now we need to get the droplet's password to execute commands through the DigitalOcean Metadata service
echo -e "\nAttempting to add SSH key to droplet..."

# Create a temporary scripts
SSH_SCRIPT=$(cat <<'EOF'
#!/bin/bash
# Ensure authorized_keys file exists
mkdir -p ~/.ssh
touch ~/.ssh/authorized_keys
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys

# Add the SSH key
echo "SSH_KEY_PLACEHOLDER" >> ~/.ssh/authorized_keys

# Ensure SSH is properly configured
sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
sed -i 's/^#\?PubkeyAuthentication.*/PubkeyAuthentication yes/' /etc/ssh/sshd_config
systemctl restart ssh
echo "SSH key added successfully"
EOF
)

# Replace placeholder with actual key
SSH_SCRIPT="${SSH_SCRIPT/SSH_KEY_PLACEHOLDER/$SSH_PUBKEY}"

# Create payload for console access
CONSOLE_PAYLOAD="{\"command\":\"$SSH_SCRIPT\"}"

# Use the DigitalOcean API to execute command via console access
CONSOLE_RESPONSE=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $DO_TOKEN" \
    -d "$CONSOLE_PAYLOAD" \
    "https://api.digitalocean.com/v2/droplets/$DROPLET_ID/actions")

# Check if API call was successful
if echo "$CONSOLE_RESPONSE" | grep -q "message.*cannot execute"; then
    echo -e "${RED}Error: Failed to execute console command.${NC}"
    echo "$CONSOLE_RESPONSE"
    exit 1
fi

echo -e "${GREEN}Successfully initiated SSH key installation on droplet.${NC}"
echo -e "${YELLOW}Note: It may take a few minutes for the changes to take effect.${NC}"
echo -e "You can now try SSH connection with: ssh -i /path/to/private/key root@$DROPLET_IP"
