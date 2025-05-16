#!/bin/bash
# Script to get SSH key ID from DigitalOcean API

# Text formatting
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
BOLD='\033[1m'

echo -e "${BOLD}DigitalOcean SSH Key ID Finder${NC}\n"

# Check for DO token
if [ -z "$1" ]; then
  echo "Usage: $0 <digitalocean_token> [key_fingerprint]"
  echo "  digitalocean_token: Your DigitalOcean API token"
  echo "  key_fingerprint: Optional - the fingerprint of the key to find (format: xx:xx:xx...)"
  
  # Try to get token from terraform.tfvars
  if [ -f "../infra/terraform.tfvars" ]; then
    echo "Attempting to extract token from terraform.tfvars..."
    DO_TOKEN=$(grep -o 'digitalocean_token *= *"[^"]*"' ../infra/terraform.tfvars | cut -d'"' -f2)
    
    if [ -n "$DO_TOKEN" ]; then
      echo "Found token in terraform.tfvars!"
    else
      echo "Could not find token in terraform.tfvars."
      exit 1
    fi
  else
    echo "terraform.tfvars not found."
    exit 1
  fi
else
  DO_TOKEN="$1"
fi

FINGERPRINT=""
if [ -n "$2" ]; then
  FINGERPRINT="$2"
  echo -e "Looking for key with fingerprint: ${BLUE}$FINGERPRINT${NC}"
fi

echo "Querying DigitalOcean API for SSH keys..."

# Make API request to list SSH keys
RESPONSE=$(curl -s -X GET \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $DO_TOKEN" \
  "https://api.digitalocean.com/v2/account/keys")

# Check if the request was successful
if echo "$RESPONSE" | grep -q '"message":"Unauthorized"'; then
  echo "Error: Unauthorized. Please check your API token."
  exit 1
fi

# Extract and display SSH keys
echo -e "\n${BOLD}SSH Keys in your DigitalOcean Account:${NC}"
echo -e "ID\tName\t\tFingerprint"
echo -e "---------------------------------------------------"

# Parse JSON with grep and awk (simple approach)
echo "$RESPONSE" | grep -Eo '"id":[0-9]+|"name":"[^"]+"|"fingerprint":"[^"]+"' | 
  awk 'BEGIN{i=0} 
       {
         if ($0 ~ /"id":/) {
           id = substr($0, 6)
           i++
         }
         else if ($0 ~ /"name":/) {
           name = substr($0, 9)
           gsub(/"/, "", name)
         }
         else if ($0 ~ /"fingerprint":/) {
           fp = substr($0, 15)
           gsub(/"/, "", fp)
           printf "%s\t%s\t%s\n", id, name, fp
           
           # If fingerprint matches, highlight it
           if (length("'$FINGERPRINT'") > 0 && fp == "'$FINGERPRINT'") {
             matched_id = id
             matched_name = name
             matched_fp = fp
           }
         }
       }'

# If a specific fingerprint was provided and found, highlight it
if [ -n "$FINGERPRINT" ] && [ -n "$matched_id" ]; then
  echo -e "\n${GREEN}✓ Found matching key!${NC}"
  echo -e "ID: ${BOLD}$matched_id${NC}"
  echo -e "Name: $matched_name"
  echo -e "Fingerprint: $matched_fp"
  
  echo -e "\n${BOLD}To update your Terraform configuration:${NC}"
  echo -e "Edit infra/terraform.tfvars and set:"
  echo -e "${BLUE}ssh_key_ids = [\"$matched_id\"]${NC}"
else
  if [ -n "$FINGERPRINT" ]; then
    echo -e "\nNo key found with fingerprint: $FINGERPRINT"
  else
    echo -e "\nNote: To find a specific key, rerun with the fingerprint:"
    echo -e "./get_ssh_key_id.sh <token> <fingerprint>"
  fi
fi
