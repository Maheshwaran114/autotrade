#!/bin/bash
# Comprehensive network diagnostics script for GitHub Actions
# This can be used to debug connectivity issues with DigitalOcean API and SSH

set -e

echo "====== COMPREHENSIVE NETWORK DIAGNOSTICS ======"
echo "Running extended network diagnostics for CI/CD pipeline..."

echo "=== System Information ==="
uname -a
cat /etc/os-release 2>/dev/null || echo "OS release info not available"

echo "=== Date and Time ==="
date
date -u  # UTC time

echo "=== Environment Variables (REDACTED) ==="
env | grep -v -E "TOKEN|SECRET|PASSWORD|KEY" | sort

echo "=== DNS Configuration ==="
echo "DNS Servers:"
cat /etc/resolv.conf 2>/dev/null || echo "resolv.conf not found"

echo "DNS resolver test:"
dig +short google.com || echo "Dig failed"

echo "=== Network Connection Tests ==="
echo "Testing internet connectivity:"
ping -c 3 8.8.8.8 || echo "Ping to Google DNS failed"
ping -c 3 1.1.1.1 || echo "Ping to Cloudflare DNS failed"

echo "Testing DigitalOcean API connectivity:"
curl -v --max-time 10 https://api.digitalocean.com/v2/regions 2>&1 | grep -E 'Connected to|Failed to connect'

echo "=== DNS Lookup Tests ==="
echo "DigitalOcean API DNS lookup:"
nslookup api.digitalocean.com || echo "nslookup failed"

echo "GitHub DNS lookup:"
nslookup github.com || echo "GitHub nslookup failed"

echo "Multiple DNS resolvers test:"
nslookup github.com 8.8.8.8 || echo "nslookup via Google DNS failed"
nslookup github.com 1.1.1.1 || echo "nslookup via Cloudflare DNS failed"

echo "=== Network Interface Information ==="
ifconfig -a 2>/dev/null || ip addr show || echo "Network interface commands unavailable"

echo "=== Route Information ==="
ip route 2>/dev/null || route -n 2>/dev/null || echo "IP route command failed"

echo "=== HTTP Proxy Settings ==="
env | grep -i proxy || echo "No proxy settings found"

echo "=== DigitalOcean API Test ==="
echo "Testing API with token (response code only):"
curl -s -o /dev/null -w "Response code: %{http_code}\n" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $DIGITALOCEAN_TOKEN" \
  "https://api.digitalocean.com/v2/account" || echo "API test failed"

echo "=== Service Status Check ==="
timeout 15 curl -s https://status.digitalocean.com/api/v2/summary.json | \
  grep -o '"status":[^,}]*' || echo "DigitalOcean status check failed"

echo "=== SSL/TLS Debug ==="
echo | openssl s_client -connect api.digitalocean.com:443 2>&1 | grep "Verification" || echo "SSL connectivity test failed"

echo "=== SSH Configuration Check ==="
echo "SSH version:"
ssh -V || echo "SSH version check failed"

echo "SSH config file:"
if [ -f ~/.ssh/config ]; then
  grep -v "IdentityFile" ~/.ssh/config || echo "No SSH config or empty file"
else
  echo "No SSH config file found"
fi

echo "=== Checking SSH Connectivity ==="
echo "Testing GitHub SSH connectivity:"
ssh-keygen -R github.com 2>/dev/null || true
ssh -T -v -o StrictHostKeyChecking=no -o ConnectTimeout=5 git@github.com 2>&1 | \
  grep -E 'connecting|Connected|Connection' || echo "GitHub SSH test failed"

echo "=== Firewall Status ==="
sudo iptables -L -n 2>/dev/null || echo "Unable to check iptables rules"

echo "=== Additional Diagnostics ==="
if command -v traceroute &>/dev/null; then
  echo "Running traceroute to DigitalOcean API:"
  traceroute -m 15 api.digitalocean.com 2>&1 || echo "Traceroute failed"
else
  echo "Traceroute not available"
fi

if command -v mtr &>/dev/null; then
  echo "Running MTR to DigitalOcean API (summary):"
  mtr --report --report-cycles=3 api.digitalocean.com || echo "MTR failed"
else
  echo "MTR not available"
fi

echo "====== NETWORK DIAGNOSTICS COMPLETED ======"
