#!/bin/zsh

# Zerodha Setup Script
# This script guides you through the Zerodha API setup process

echo "===== Zerodha API Setup Script ====="
echo "This script will help you set up the Zerodha API integration."

PROJECT_DIR="$(dirname "$0")/.."
CREDENTIALS_FILE="${PROJECT_DIR}/config/credentials.json"

# Create logs directory
mkdir -p "${PROJECT_DIR}/logs"

# Check if credentials file exists
if [ -f "$CREDENTIALS_FILE" ]; then
    echo "Credentials file found at: ${CREDENTIALS_FILE}"
    echo "Current configuration:"
    cat "$CREDENTIALS_FILE" | grep -v "access_token"
    
    read -p "Do you want to continue using these credentials? (y/n): " use_existing
    if [[ "$use_existing" != "y" && "$use_existing" != "Y" ]]; then
        echo "Let's update your credentials..."
        
        read -p "Enter your Zerodha API Key: " api_key
        read -p "Enter your Zerodha API Secret: " api_secret
        
        cat > "$CREDENTIALS_FILE" << EOF
{
    "api_key": "$api_key",
    "api_secret": "$api_secret",
    "access_token": ""
}
EOF
        echo "Credentials updated successfully!"
    fi
else
    echo "No credentials file found. Let's create one..."
    
    read -p "Enter your Zerodha API Key: " api_key
    read -p "Enter your Zerodha API Secret: " api_secret
    
    cat > "$CREDENTIALS_FILE" << EOF
{
    "api_key": "$api_key",
    "api_secret": "$api_secret",
    "access_token": ""
}
EOF
    echo "Credentials file created successfully!"
fi

# Generate access token
echo
echo "Now we'll generate an access token for Zerodha API."
echo "This will open a browser window where you'll need to log in to your Zerodha account."
echo "After logging in, you'll be redirected to localhost, and the token will be captured automatically."
echo

read -p "Press Enter to continue to the token generation process..." 

# Run the token generator script
python "${PROJECT_DIR}/scripts/generate_access_token.py"
TOKEN_STATUS=$?

if [ $TOKEN_STATUS -eq 0 ]; then
    echo
    echo "===== Setup Complete ====="
    echo "Your Zerodha API is now configured and ready to use."
    echo "The access token will expire at 6 AM tomorrow and needs to be refreshed daily."
    echo
    echo "Would you like to set up automatic daily token refresh checks?"
    read -p "Set up daily token check (y/n): " setup_cron
    
    if [[ "$setup_cron" == "y" || "$setup_cron" == "Y" ]]; then
        # Check if the cron job already exists
        EXISTING_CRON=$(crontab -l 2>/dev/null | grep "refresh_zerodha_token.sh")
        
        if [ -z "$EXISTING_CRON" ]; then
            # Add the cron job
            (crontab -l 2>/dev/null; echo "30 7 * * 1-5 ${PROJECT_DIR}/scripts/refresh_zerodha_token.sh >> ${PROJECT_DIR}/logs/cron_zerodha.log 2>&1") | crontab -
            echo "Daily token check scheduled for 7:30 AM on weekdays."
        else
            echo "A token refresh cron job already exists:"
            echo "$EXISTING_CRON"
        fi
    else
        echo "You can manually refresh your token by running:"
        echo "python ${PROJECT_DIR}/scripts/generate_access_token.py"
    fi
else
    echo
    echo "===== Token Generation Failed ====="
    echo "There was a problem generating the access token."
    echo "Please check the logs and try again manually."
    echo "Run: python ${PROJECT_DIR}/scripts/generate_access_token.py"
fi

echo
echo "For more information on managing your Zerodha tokens, see:"
echo "${PROJECT_DIR}/docs/TOKEN_MANAGEMENT.md"
echo
echo "For general Zerodha setup information, see:"
echo "${PROJECT_DIR}/docs/ZERODHA_SETUP.md"
