#!/bin/zsh

# Daily Zerodha Access Token Refresh Script
# This script checks if the token is still valid and logs any issues

# Change to project directory
cd "$(dirname "$0")/.."
PROJECT_DIR="$(pwd)"

# Setup paths
LOG_FILE="${PROJECT_DIR}/logs/token_refresh.log"
SCRIPT_PATH="${PROJECT_DIR}/scripts/generate_access_token.py"

# Create logs directory if it doesn't exist
mkdir -p "${PROJECT_DIR}/logs"

# Log function
log() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "${LOG_FILE}"
  echo "$1"
}

log "========== Starting Zerodha token refresh check =========="

# Check if the token is valid
"${SCRIPT_PATH}" --check
TOKEN_STATUS=$?

if [ $TOKEN_STATUS -eq 0 ]; then
  log "Access token is still valid. No action needed."
  exit 0
fi

# Token is invalid, try to refresh
log "Access token is invalid or expired."
log "Attempting scheduled token refresh..."

"${SCRIPT_PATH}" --scheduled
REFRESH_STATUS=$?

if [ $REFRESH_STATUS -eq 0 ]; then
  log "Token refresh successful!"
else
  log "WARNING: Token refresh failed. Manual intervention required."
  log "Run '${SCRIPT_PATH}' manually to generate a new token."
  
  # You could add notification commands here
  # Example: sending an email alert
  # mail -s "Zerodha Token Expired" your@email.com < "${LOG_FILE}"
fi

log "========== Token refresh check completed =========="
exit $REFRESH_STATUS
