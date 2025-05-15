#!/bin/bash
# Script to help troubleshoot network issues in GitHub Actions
# This can be used in the "Perform network diagnostics" step

echo "Running network diagnostics..."

echo "=== System Information ==="
uname -a
cat /etc/os-release 2>/dev/null || echo "OS release info not available"

echo "=== DNS Configuration ==="
echo "DNS Servers:"
cat /etc/resolv.conf 2>/dev/null || echo "resolv.conf not found"

echo "=== Network Connection Tests ==="
echo "Testing DigitalOcean API connectivity:"
curl -v --max-time 10 https://api.digitalocean.com/v2/regions 2>&1 | grep -E 'Connected to|Failed to connect'

echo "=== DNS Lookup Tests ==="
nslookup api.digitalocean.com 2>&1 || echo "nslookup failed"
host api.digitalocean.com 2>&1 || echo "host lookup failed"

echo "=== Route Information ==="
ip route 2>/dev/null || echo "IP route command failed"

echo "=== HTTP Proxy Settings ==="
env | grep -i proxy || echo "No proxy settings found"

echo "=== DigitalOcean API Test ==="
curl -s -o /dev/null -w "Response code: %{http_code}\n" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $DIGITALOCEAN_TOKEN" \
  "https://api.digitalocean.com/v2/account" || echo "API test failed"

echo "=== Traceroute to DigitalOcean ==="
traceroute -m 15 api.digitalocean.com 2>&1 || echo "Traceroute failed or not available"

echo "=== Checking SSH Connectivity ==="
ssh-keygen -R github.com 2>/dev/null
ssh -T -v -o StrictHostKeyChecking=no -o ConnectTimeout=5 git@github.com 2>&1 | grep -E 'connecting|Connected|Connection'

echo "Network diagnostics completed"
