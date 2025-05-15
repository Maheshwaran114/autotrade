# SSH Testing Instructions

## Important Note About SSH Keys

The SSH key you provided (`ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFaDmFyYb1lN4b2quiZSs8s5eclAamBC2bKM4oiT0n0o`) appears to be a **public key**, but for SSH authentication you need the **private key**.

A private key typically starts with:
```
-----BEGIN OPENSSH PRIVATE KEY-----
```
or
```
-----BEGIN RSA PRIVATE KEY-----
```

If you're using GitHub Actions, the private key should be the one you've stored in your GitHub repository secrets as `SSH_PRIVATE_KEY`.

## Testing Steps

1. First, find your SSH private key:
   - If you created an ED25519 key, it should be in `~/.ssh/id_ed25519` (not `id_ed25519.pub`)
   - If you created an RSA key, it should be in `~/.ssh/id_bn_trading` (not `id_bn_trading.pub`)

2. Run the SSH connection test:
   ```zsh
   ./test_ssh_connection.sh
   ```

3. When prompted for the SSH key path (if your key isn't in the default location), enter the full path to your private key file.

## Testing with DigitalOcean API

The DigitalOcean API token you provided can be used to list and manage your droplets:

```zsh
export DIGITALOCEAN_TOKEN="YOUR_DO_TOKEN_HERE"
./list_droplets.sh
```

## Creating a New Private Key (if needed)

If you can't find your private key or need to create a new one:

1. Generate a new key pair:
   ```zsh
   ssh-keygen -t ed25519 -C "your_email@example.com"
   ```

2. Add the public key to DigitalOcean:
   - Copy the contents of `~/.ssh/id_ed25519.pub`
   - Add it to DigitalOcean under Settings > Security > SSH Keys

3. Update your GitHub repository secrets with the new private key.

## Checking Key Format

You can use the validation script to check if your key is in the correct format:

```zsh
./validate_ssh_key.sh
```

## Testing with Specific Configuration

The test script is now configured to use:
- IP address: 64.227.129.85
- Default SSH key location: ~/.ssh/id_bn_trading

If your private key is in a different location, you'll be prompted to provide the path.
