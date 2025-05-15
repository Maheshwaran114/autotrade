#!/bin/bash
# Test SSH key content validation script
# This script helps validate if your SSH key is in the right format

echo "===== SSH Key Format Test ====="
echo "This script will check if your SSH key is in the correct format"
echo

# Create temporary directory
TEST_DIR=$(mktemp -d)
echo "Created temporary test directory: $TEST_DIR"

cleanup() {
  echo "Cleaning up temporary directory..."
  rm -rf "$TEST_DIR"
}
trap cleanup EXIT

# Save key to file
echo "Please enter your SSH PRIVATE key (not public key)"
echo "Paste the key below (Ctrl+D when done):"
cat > "$TEST_DIR/test_key.txt"
echo

# Check key format
if grep -q "BEGIN .* PRIVATE KEY" "$TEST_DIR/test_key.txt"; then
  echo "✅ Key appears to be in the correct private key format."
  echo "It contains 'BEGIN ... PRIVATE KEY' which is expected."
  chmod 600 "$TEST_DIR/test_key.txt"
else
  echo "❌ Key doesn't appear to be in private key format."
  echo "A private SSH key typically starts with '-----BEGIN OPENSSH PRIVATE KEY-----'"
  echo "or '-----BEGIN RSA PRIVATE KEY-----'"
  echo
  echo "What you provided might be a public key, which starts with:"
  echo "- 'ssh-rsa ...' or"
  echo "- 'ssh-ed25519 ...' or" 
  echo "- 'ssh-dss ...'"
  echo
  echo "For SSH authentication, you need the private key, not the public key."
  echo "The private key is usually found in files like id_rsa, id_ed25519 (without .pub extension)"
  echo "The public key is usually found in files ending with .pub"
fi

# Display first and last line for validation
echo
echo "First line of provided key:"
head -n 1 "$TEST_DIR/test_key.txt"
echo "..."
echo "Last line of provided key:"
tail -n 1 "$TEST_DIR/test_key.txt"
echo

echo "To use this key with test_ssh_connection.sh, save it to a file with:"
echo "nano ~/.ssh/do_droplet_key"
echo "# paste your private key here"
echo "# save with Ctrl+O, then Ctrl+X"
echo "chmod 600 ~/.ssh/do_droplet_key"
echo
echo "Then run the test script with:"
echo "./test_ssh_connection.sh"
echo "# When prompted for SSH key path, enter: ~/.ssh/do_droplet_key"
