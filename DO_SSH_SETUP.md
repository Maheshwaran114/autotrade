# DigitalOcean SSH Key Setup

Since your droplet is already running and only allows public key authentication, we need to register your SSH key with DigitalOcean and then either:

1. Recreate the droplet with your SSH key
2. Update the droplet to include your SSH key

## Option 1: Register Your SSH Key with DigitalOcean

First, run the `register_ssh_key.sh` script to add your SSH key to your DigitalOcean account:

```bash
cd ~/Documents/GitHub/autotrade
./register_ssh_key.sh
```

This will:
- Take your public key from `~/.ssh/id_bn_trading.pub`
- Register it with your DigitalOcean account
- Give you the SSH key ID (important for Terraform and GitHub Actions)

## Option 2: Check If You Have Console Access

If you can access the DigitalOcean web console, you might be able to log in to the droplet and add your key manually:

1. Log in to DigitalOcean dashboard
2. Go to Droplets
3. Find your droplet (IP: 64.227.129.85)
4. Click on "Access" and then "Launch Console"
5. Log in with root credentials
6. Once logged in, run:
   ```
   mkdir -p ~/.ssh
   echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFaDmFyYb1lN4b2quiZSs8s5eclAamBC2bKM4oiT0n0o umamaheswaran.114@gmail.com" >> ~/.ssh/authorized_keys
   chmod 600 ~/.ssh/authorized_keys
   ```

## Option 3: Recreate the Droplet with Terraform

If you can't access the droplet, the easiest solution might be to destroy and recreate it with your SSH key:

1. Register your SSH key with DigitalOcean (Option 1)
2. Update your `terraform.tfvars` file with the new SSH key ID
3. Run Terraform to recreate the droplet:
   ```bash
   cd ~/Documents/GitHub/autotrade/infra
   terraform destroy -auto-approve
   terraform apply -auto-approve
   ```

## Testing After Setup

Once your SSH key is properly set up, you can test the connection:

```bash
cd ~/Documents/GitHub/autotrade
./test_ssh_connection.sh
```

## For GitHub Actions

For your GitHub Actions workflow to use your SSH key:

1. Add these secrets to your GitHub repository:
   - `SSH_PRIVATE_KEY`: The content of `~/.ssh/id_bn_trading`
   - `SSH_KEY_ID`: The ID from the DigitalOcean key registration

2. Make sure your workflow is using these secrets correctly.
