# SSH Connection Troubleshooting Guide

This guide provides solutions for common SSH connection issues in GitHub Actions workflows, particularly when deploying to DigitalOcean droplets.

## Common Issues and Solutions

### 1. UserKnownHostsFile "/null" Error

**Problem**: SSH config contains `UserKnownHostsFile /null` which is incorrect.

**Solution**: 
- Use `/dev/null` instead, which is the correct path for discarding host key verification
- Run `fix_ssh_config.sh` to automatically fix this issue

### 2. SSH Connection Timeout

**Problem**: SSH connection times out when trying to connect to the droplet.

**Solutions**:
- Increase connection timeout values (we've set to 180 seconds in the workflow)
- Use multiple connection attempts
- Ensure firewall rules allow connections from GitHub Actions IP ranges
- Verify the droplet is running and SSH service is active

### 3. SSH Authentication Failures

**Problem**: SSH authentication fails when connecting to the droplet.

**Solutions**:
- Ensure correct permissions on SSH key (should be 600)
- Verify SSH key is properly formatted
- Check that the public key is added to the droplet's authorized_keys
- Make sure PermitRootLogin and PubkeyAuthentication are enabled on the server

### 4. Network Connectivity Issues

**Problem**: Network connectivity problems prevent SSH connection.

**Solutions**:
- Verify the droplet's IP address is correct
- Check if port 22 is open and reachable
- Test basic connectivity with ping and traceroute
- Ensure no network restrictions are in place

## Scripts for Diagnosing Issues

We've provided several scripts to help diagnose SSH issues:

1. `enhanced_ssh_diagnostics.sh` - Tests SSH connectivity with extended diagnostics
2. `fix_ssh_config.sh` - Checks and fixes common SSH configuration issues
3. `check_server_ssh.sh` - Server-side script to verify SSH configuration on the droplet
4. `ssh_diagnostics.sh` - General SSH diagnostics
5. `network_diagnostics.sh` - Network connectivity diagnostics

## Best Practices for SSH in GitHub Actions

1. **SSH Configuration**:
   - Always use `/dev/null` for `UserKnownHostsFile` when disabling host key checking
   - Set appropriate log levels (VERBOSE is recommended, DEBUG can be too verbose)
   - Use BatchMode for non-interactive sessions

2. **SSH Key Handling**:
   - Ensure keys have 600 permissions
   - Store SSH keys securely in GitHub Secrets
   - Use dedicated keys for GitHub Actions

3. **Connection Parameters**:
   - Use reasonable connection timeouts (30-60 seconds)
   - Implement retry logic with increasing delays
   - Use ServerAliveInterval to keep connections alive

4. **Debugging**:
   - Use `-vvv` for maximum verbosity when troubleshooting
   - Check both client and server configurations
   - Verify network connectivity before attempting SSH

## Example Fix for UserKnownHostsFile Issue

```bash
# Incorrect
UserKnownHostsFile /null

# Correct
UserKnownHostsFile /dev/null
```

## Running the Fix Script

To fix common SSH configuration issues, run:

```bash
./fix_ssh_config.sh
```

For server-side verification, run this on your DigitalOcean droplet:

```bash
./check_server_ssh.sh
```
