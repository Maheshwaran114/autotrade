# SSH Connection Testing Guide

This guide explains how to use the `test_ssh_connection.sh` script to verify SSH connectivity to your DigitalOcean droplet.

## Required Information

To run the SSH connection test, you'll need:

1. **DigitalOcean Droplet IP Address** - The IP address of your DigitalOcean droplet
   - This can be an existing droplet or one created by your workflow
   - You can view your droplet IPs in the DigitalOcean dashboard or by using the `list_droplets.sh` script

2. **SSH Private Key** - The private key used for authentication
   - This should be the same key that's configured in your GitHub secrets as `SSH_PRIVATE_KEY`
   - The public key should already be added to your DigitalOcean account
   - Default location is `$HOME/.ssh/id_rsa`, but you can specify a different path when prompted

3. **DigitalOcean API Token** (optional) - Only needed for listing or managing droplets
   - Not required for the basic SSH connectivity test
   - Required if you want to use the `list_droplets.sh` or `cleanup_droplets.sh` scripts

## Running the Test

1. Make the script executable:
   ```bash
   chmod +x test_ssh_connection.sh
   ```

2. Run the script:
   ```bash
   ./test_ssh_connection.sh
   ```

3. When prompted, enter:
   - The IP address of your DigitalOcean droplet
   - The path to your SSH private key (if not in the default location)

## What the Test Checks

This test script:

1. Creates a temporary SSH environment similar to GitHub Actions
2. Validates the IP address format
3. Tests basic SSH connectivity
4. Checks for the `/bn-trading` directory and Docker installations
5. Reports success or troubleshooting steps

## Using with List Droplets Script

To find the IP of your droplet before testing:

```bash
export DIGITALOCEAN_TOKEN=your_token_here
./list_droplets.sh
```

Then use that IP address in the SSH test.

## Common Issues

1. **Invalid SSH key format**: Ensure your key is in the correct format (OpenSSH or RSA)
2. **Permission issues**: The script sets appropriate permissions, but if you're providing your own key file, ensure it has `600` permissions
3. **Firewall restrictions**: Ensure your droplet's firewall allows SSH connections on port 22
4. **Invalid IP address**: Double-check the IP address of your droplet

## Next Steps

If the test is successful, your GitHub Actions workflow should be able to connect via SSH. If it fails, review the error messages and troubleshooting steps provided by the script.
