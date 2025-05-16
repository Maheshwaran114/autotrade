# Bank Nifty Trading System Deployment Guide

This guide provides information about the deployment process for the Bank Nifty Trading System.

## Deployment Architecture

The Bank Nifty Trading System is deployed on a DigitalOcean droplet using Docker containers. The deployment is managed through GitHub Actions, which automatically builds and deploys the application whenever changes are pushed to the main branch.

## Deployment Process

1. **GitHub Actions Build**: When code is pushed to the main branch, GitHub Actions:
   - Builds the Docker image
   - Pushes the image to Docker Hub
   - Provisions or updates the DigitalOcean infrastructure using Terraform
   - Deploys the application to the droplet

2. **Infrastructure**: The system runs on:
   - DigitalOcean Droplet (s-2vcpu-4gb) in the Bangalore region
   - Docker for containerization
   - PostgreSQL for data storage

3. **SSH Authentication**: The system uses SSH keys for secure authentication:
   - The SSH key is registered with DigitalOcean
   - Multiple fallback mechanisms ensure deployment succeeds even if the primary key has issues

## Current Deployment

- **Droplet IP**: 165.22.214.71
- **Application URL**: http://165.22.214.71:5000
- **Last Deployment**: May 16, 2025

## Troubleshooting

If deployment fails due to SSH authentication issues:

1. Check the GitHub Secrets:
   - Ensure `SSH_PRIVATE_KEY` is correctly formatted with BEGIN/END lines
   - Verify `DIGITALOCEAN_TOKEN` is valid and has write permissions

2. Run the diagnostic script:
   ```bash
   ./scripts/debug_ssh.sh 165.22.214.71 ./scripts/do_deployment_key
   ```

3. Update SSH keys in DigitalOcean if needed:
   ```bash
   ./scripts/fix_ssh_auth.sh your_do_token 165.22.214.71
   ```

## Security Considerations

- The SSH key used for deployment should be kept secure
- The DigitalOcean API token should have appropriate permissions
- Database credentials should be managed securely
