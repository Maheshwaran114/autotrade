#!/bin/bash
# Test script for IP extraction

# Simulate the terraform output with debug info
echo "Creating test file with simulated terraform output..."
echo "64.227.149.109::debug::Terraform exited with code 0." > test_output.txt

echo "Contents of test file:"
cat test_output.txt

echo "Trying multiple extraction methods:"

echo "Method 1: Using grep:"
IP1=$(grep -Eo '([0-9]{1,3}\.){3}[0-9]{1,3}' test_output.txt)
echo "Extracted IP with grep: $IP1"

echo "Method 2: Using cut:"
IP2=$(cat test_output.txt | cut -d':' -f1)
echo "Extracted IP with cut: $IP2"

echo "Method 3: Using sed:"
IP3=$(sed -n 's/\([0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}\).*/\1/p' test_output.txt)
echo "Extracted IP with sed: $IP3"

echo "Method 4: Using awk:"
IP4=$(awk -F'::' '{print $1}' test_output.txt)
echo "Extracted IP with awk: $IP4"

# Cleanup
rm test_output.txt

echo "Test complete"
