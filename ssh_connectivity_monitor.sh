#!/bin/bash
# SSH CONNECTIVITY MONITOR 
# This script monitors the SSH connectivity and server status at regular intervals
# It helps track any intermittent connectivity issues and provides early warnings

set -e

echo "===== SSH CONNECTIVITY MONITORING TOOL ====="
echo "Date/Time: $(date)"
echo "User: $(whoami)"

# Default values
TARGET_IP="${1}"
INTERVAL="${2:-300}" # Default to 5 minutes
DURATION="${3:-3600}" # Default to 1 hour
LOG_FILE="ssh_connectivity_log_$(date +%Y%m%d_%H%M%S).log"

# Validate input
if [ -z "$TARGET_IP" ]; then
  echo "❌ ERROR: No target IP address provided."
  echo "Usage: $0 <target-ip-address> [interval-seconds] [duration-seconds]"
  exit 1
fi

echo "Starting SSH connectivity monitoring:"
echo "- Target IP: $TARGET_IP"
echo "- Check interval: $INTERVAL seconds"
echo "- Monitoring duration: $DURATION seconds"
echo "- Log file: $LOG_FILE"

# Initialize counters
TOTAL_CHECKS=0
SUCCESSFUL_CHECKS=0
FAILED_CHECKS=0

# Calculate number of checks
CHECKS=$((DURATION / INTERVAL))
if [ $CHECKS -lt 1 ]; then CHECKS=1; fi

echo "Will perform $CHECKS checks in total."

# Log header
echo "===== SSH CONNECTIVITY MONITORING LOG =====" > "$LOG_FILE"
echo "Started at: $(date)" >> "$LOG_FILE"
echo "Target IP: $TARGET_IP" >> "$LOG_FILE"
echo "Check interval: $INTERVAL seconds" >> "$LOG_FILE"
echo "----------------------------------------" >> "$LOG_FILE"

# Monitoring loop
START_TIME=$(date +%s)
END_TIME=$((START_TIME + DURATION))

while [ $(date +%s) -lt $END_TIME ]; do
  CHECK_TIME=$(date +"%Y-%m-%d %H:%M:%S")
  echo "[$CHECK_TIME] Performing connectivity check #$((TOTAL_CHECKS + 1))..."
  
  # Echo dots to show progress
  echo -n "Testing connectivity: "
  
  # Ping test
  PING_RESULT=$(ping -c 1 -W 5 "$TARGET_IP" >/dev/null 2>&1 && echo "success" || echo "failed")
  echo -n "."
  
  # TCP test to SSH port
  TCP_RESULT=$(nc -z -w 5 "$TARGET_IP" 22 >/dev/null 2>&1 && echo "success" || echo "failed")
  echo -n "."
  
  # SSH test with timeout
  SSH_TEST_CMD="timeout 10 ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o ConnectTimeout=5 -i ~/.ssh/id_bn_trading root@$TARGET_IP exit 2>/dev/null"
  SSH_RESULT=$(eval "$SSH_TEST_CMD" && echo "success" || echo "failed")
  echo "."
  
  # Update counters
  TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
  
  # Log results
  if [[ "$PING_RESULT" == "success" && "$TCP_RESULT" == "success" && "$SSH_RESULT" == "success" ]]; then
    echo "[$CHECK_TIME] ✅ All connectivity checks PASSED"
    echo "[$CHECK_TIME] ✅ All connectivity checks PASSED" >> "$LOG_FILE"
    SUCCESSFUL_CHECKS=$((SUCCESSFUL_CHECKS + 1))
  else
    echo "[$CHECK_TIME] ❌ Some connectivity checks FAILED:"
    echo "   - Ping: $PING_RESULT"
    echo "   - TCP port 22: $TCP_RESULT"
    echo "   - SSH login: $SSH_RESULT"
    
    echo "[$CHECK_TIME] ❌ Connectivity check FAILED" >> "$LOG_FILE"
    echo "   - Ping: $PING_RESULT" >> "$LOG_FILE"
    echo "   - TCP port 22: $TCP_RESULT" >> "$LOG_FILE"
    echo "   - SSH login: $SSH_RESULT" >> "$LOG_FILE"
    
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
    
    # Run more detailed diagnostics on failure
    echo "Running detailed diagnostics for failed check..."
    {
      echo "=== DETAILED DIAGNOSTICS AT $CHECK_TIME ==="
      echo "IP route:"
      ip route
      echo "Traceroute (max 10 hops):"
      traceroute -m 10 -T "$TARGET_IP" 2>&1 || echo "Traceroute failed"
      echo "SSH verbose connection attempt:"
      timeout 15 ssh -v -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -i ~/.ssh/id_bn_trading root@$TARGET_IP exit 2>&1 || echo "SSH verbose check failed"
      echo "========================================="
    } >> "${LOG_FILE}.diagnostics"
  fi
  
  # Calculate success rate
  SUCCESS_RATE=$((SUCCESSFUL_CHECKS * 100 / TOTAL_CHECKS))
  echo "Current success rate: $SUCCESS_RATE% ($SUCCESSFUL_CHECKS/$TOTAL_CHECKS)"
  
  # Check if we've reached the desired number of checks
  if [ $TOTAL_CHECKS -ge $CHECKS ]; then
    break
  fi
  
  # Wait for next interval
  echo "Waiting $INTERVAL seconds until next check..."
  sleep $INTERVAL
done

# Final summary
echo "===== MONITORING COMPLETE ====="
echo "Total checks: $TOTAL_CHECKS"
echo "Successful: $SUCCESSFUL_CHECKS"
echo "Failed: $FAILED_CHECKS"
echo "Success rate: $((SUCCESSFUL_CHECKS * 100 / TOTAL_CHECKS))%"
echo "Detailed logs available in: $LOG_FILE"
if [ $FAILED_CHECKS -gt 0 ]; then
  echo "Diagnostics for failed checks: ${LOG_FILE}.diagnostics"
fi

# Log summary
echo "===== MONITORING SUMMARY =====" >> "$LOG_FILE"
echo "Ended at: $(date)" >> "$LOG_FILE"
echo "Total checks: $TOTAL_CHECKS" >> "$LOG_FILE"
echo "Successful: $SUCCESSFUL_CHECKS" >> "$LOG_FILE"
echo "Failed: $FAILED_CHECKS" >> "$LOG_FILE"
echo "Success rate: $((SUCCESSFUL_CHECKS * 100 / TOTAL_CHECKS))%" >> "$LOG_FILE"
