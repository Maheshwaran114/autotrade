#!/bin/bash
# SERVER-SIDE HEALTH CHECK FOR BN TRADING APPLICATION
# This script runs on the server to check the application health
# It provides detailed diagnostics about the running application

set -e

echo "===== BN TRADING APPLICATION HEALTH CHECK ====="
echo "Date/Time: $(date)"
echo "Running on: $(hostname)"
echo "User: $(whoami)"

# Log directory
LOG_DIR="/var/log/bn-trading"
mkdir -p "$LOG_DIR" 2>/dev/null || true
HEALTH_LOG="$LOG_DIR/health_check_$(date +%Y%m%d_%H%M%S).log"

# Start logging
echo "===== HEALTH CHECK LOG =====" > "$HEALTH_LOG"
echo "Started at: $(date)" >> "$HEALTH_LOG"
echo "Running on: $(hostname)" >> "$HEALTH_LOG"
echo "----------------------------------------" >> "$HEALTH_LOG"

# 1. Check system resources
echo "1. Checking system resources..."
echo "==== SYSTEM RESOURCES ====" >> "$HEALTH_LOG"
echo "CPU usage:" >> "$HEALTH_LOG"
top -bn1 | head -15 >> "$HEALTH_LOG"
echo "" >> "$HEALTH_LOG"

echo "Memory usage:" >> "$HEALTH_LOG"
free -h >> "$HEALTH_LOG"
echo "" >> "$HEALTH_LOG"

echo "Disk usage:" >> "$HEALTH_LOG"
df -h >> "$HEALTH_LOG"
echo "" >> "$HEALTH_LOG"

# 2. Check Docker status
echo "2. Checking Docker status..."
echo "==== DOCKER STATUS ====" >> "$HEALTH_LOG"
if command -v docker >/dev/null 2>&1; then
  echo "Docker service status:" >> "$HEALTH_LOG"
  systemctl status docker | grep Active >> "$HEALTH_LOG" || echo "Failed to get Docker status" >> "$HEALTH_LOG"
  echo "" >> "$HEALTH_LOG"
  
  echo "Running containers:" >> "$HEALTH_LOG"
  docker ps >> "$HEALTH_LOG" || echo "Failed to list Docker containers" >> "$HEALTH_LOG"
  echo "" >> "$HEALTH_LOG"
  
  # 3. Check BN Trading container specifically
  echo "3. Checking BN Trading container..."
  echo "==== BN TRADING CONTAINER ====" >> "$HEALTH_LOG"
  
  BN_CONTAINER=$(docker ps | grep -i "trading" | awk '{print $1}' | head -1)
  if [ -n "$BN_CONTAINER" ]; then
    echo "✅ BN Trading container is running with ID: $BN_CONTAINER"
    echo "Container found: $BN_CONTAINER" >> "$HEALTH_LOG"
    
    echo "Container details:" >> "$HEALTH_LOG"
    docker inspect "$BN_CONTAINER" >> "$HEALTH_LOG" 2>&1 || echo "Failed to inspect container" >> "$HEALTH_LOG"
    echo "" >> "$HEALTH_LOG"
    
    echo "Container logs (last 50 lines):" >> "$HEALTH_LOG"
    docker logs --tail 50 "$BN_CONTAINER" >> "$HEALTH_LOG" 2>&1 || echo "Failed to get container logs" >> "$HEALTH_LOG"
    echo "" >> "$HEALTH_LOG"
  else
    echo "❌ BN Trading container is not running!"
    echo "ERROR: BN Trading container not found" >> "$HEALTH_LOG"
    
    # List all containers
    echo "All containers (including stopped):" >> "$HEALTH_LOG"
    docker ps -a >> "$HEALTH_LOG" 2>&1 || echo "Failed to list all containers" >> "$HEALTH_LOG"
    echo "" >> "$HEALTH_LOG"
  fi
else
  echo "❌ Docker is not installed or not in PATH!"
  echo "ERROR: Docker command not found" >> "$HEALTH_LOG"
fi

# 4. Check application port
echo "4. Checking application port..."
echo "==== APPLICATION PORT CHECK ====" >> "$HEALTH_LOG"
if command -v netstat >/dev/null 2>&1; then
  echo "Port 5000 status:" >> "$HEALTH_LOG"
  netstat -tulpn | grep :5000 >> "$HEALTH_LOG" || echo "Port 5000 not in use" >> "$HEALTH_LOG"
else
  echo "Netstat not available, trying ss command:" >> "$HEALTH_LOG"
  ss -tulpn | grep :5000 >> "$HEALTH_LOG" || echo "Port 5000 not in use" >> "$HEALTH_LOG"
fi
echo "" >> "$HEALTH_LOG"

# 5. Check HTTP endpoint
echo "5. Checking HTTP endpoint..."
echo "==== HTTP ENDPOINT CHECK ====" >> "$HEALTH_LOG"
if command -v curl >/dev/null 2>&1; then
  echo "Main endpoint response:" >> "$HEALTH_LOG"
  curl -s -m 5 http://localhost:5000/ >> "$HEALTH_LOG" 2>&1 || echo "Failed to connect to application endpoint" >> "$HEALTH_LOG"
  echo "" >> "$HEALTH_LOG"
  
  echo "Health endpoint response:" >> "$HEALTH_LOG"
  curl -s -m 5 http://localhost:5000/health >> "$HEALTH_LOG" 2>&1 || echo "Failed to connect to health endpoint" >> "$HEALTH_LOG"
  echo "" >> "$HEALTH_LOG"
else
  echo "❌ curl is not installed or not in PATH!"
  echo "ERROR: curl command not found" >> "$HEALTH_LOG"
fi

# 6. Check firewall status
echo "6. Checking firewall status..."
echo "==== FIREWALL STATUS ====" >> "$HEALTH_LOG"
if command -v ufw >/dev/null 2>&1; then
  echo "UFW status:" >> "$HEALTH_LOG"
  ufw status >> "$HEALTH_LOG" 2>&1 || echo "Failed to get UFW status" >> "$HEALTH_LOG"
else
  echo "UFW not found, checking iptables:" >> "$HEALTH_LOG"
  iptables -L | grep -i "port.*5000\|port.*ssh\|port.*22" >> "$HEALTH_LOG" 2>&1 || echo "No specific rules for ports 5000 or 22 found" >> "$HEALTH_LOG"
fi
echo "" >> "$HEALTH_LOG"

# 7. Check logs
echo "7. Checking application logs..."
echo "==== APPLICATION LOGS ====" >> "$HEALTH_LOG"
if [ -d "/bn-trading/logs" ]; then
  echo "Application log files:" >> "$HEALTH_LOG"
  ls -la /bn-trading/logs >> "$HEALTH_LOG" 2>&1 || echo "Failed to list log files" >> "$HEALTH_LOG"
  echo "" >> "$HEALTH_LOG"
  
  # Get most recent log file
  RECENT_LOG=$(ls -t /bn-trading/logs/*.log 2>/dev/null | head -1)
  if [ -n "$RECENT_LOG" ]; then
    echo "Most recent log file ($RECENT_LOG, last 20 lines):" >> "$HEALTH_LOG"
    tail -20 "$RECENT_LOG" >> "$HEALTH_LOG" 2>&1 || echo "Failed to read log file" >> "$HEALTH_LOG"
  else
    echo "No log files found" >> "$HEALTH_LOG"
  fi
else
  echo "Application log directory not found" >> "$HEALTH_LOG"
fi

# 8. Summary
echo "8. Generating summary..."
echo "==== HEALTH CHECK SUMMARY ====" >> "$HEALTH_LOG"

# Determine overall status
if docker ps | grep -q -i "trading"; then
  if curl -s -m 2 http://localhost:5000/health 2>/dev/null | grep -q "healthy"; then
    OVERALL_STATUS="✅ HEALTHY"
  else
    OVERALL_STATUS="⚠️ PARTIALLY HEALTHY (container running but health endpoint not responding)"
  fi
else
  OVERALL_STATUS="❌ UNHEALTHY (container not running)"
fi

echo "Overall status: $OVERALL_STATUS" >> "$HEALTH_LOG"
echo "Health check completed at: $(date)" >> "$HEALTH_LOG"

# Print summary to console
echo ""
echo "===== HEALTH CHECK SUMMARY ====="
echo "Overall status: $OVERALL_STATUS"
echo "Detailed log saved to: $HEALTH_LOG"
echo "Run 'cat $HEALTH_LOG' to view the full report"
