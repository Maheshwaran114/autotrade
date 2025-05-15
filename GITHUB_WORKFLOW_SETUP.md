# Setting Up Your GitHub Actions Workflow

This guide will help you set up your GitHub Actions workflow for deploying to DigitalOcean.

## 1. Prepare GitHub Repository Secrets

Run the script to prepare GitHub secrets:

```bash
./prepare_github_secrets.sh
```

This creates a `github_secrets.txt` file with all the values you need to add as secrets:

- `SSH_KEY_ID`: The ID of your SSH key in DigitalOcean
- `SSH_PRIVATE_KEY`: Your SSH private key content
- `DIGITALOCEAN_TOKEN`: Your DigitalOcean API token
- `DOCKERHUB_USERNAME`: Your Docker Hub username (if using)
- `DOCKERHUB_TOKEN`: Your Docker Hub access token (if using)

Add these secrets in your GitHub repository:
1. Go to your GitHub repository
2. Click on "Settings" > "Secrets and variables" > "Actions"
3. Click "New repository secret" for each secret

## 2. Use the Updated Terraform Configuration

The `terraform.tfvars` file has been updated with your:
- DigitalOcean API token
- SSH key ID (47586025)

This ensures Terraform will use your SSH key when creating new droplets.

## 3. Optional: Update Existing Droplets

If you have existing droplets that need your SSH key, run:

```bash
./update_droplet_ssh_key.sh
```

This adds your SSH key to the existing droplet at IP 64.227.129.85.

## 4. Test Your Workflow

1. Push a small change to trigger the workflow:
   ```bash
   git add .
   git commit -m "Update configuration for GitHub Actions"
   git push
   ```

2. Monitor the workflow in GitHub Actions:
   - Go to your GitHub repository
   - Click on "Actions" tab
   - Find your running workflow and verify it completes successfully

## Troubleshooting

If the workflow fails:

1. Check the workflow logs for errors
2. Verify all secrets are correctly set in GitHub
3. Run the SSH test script locally to verify connectivity:
   ```bash
   ./test_ssh_connection.sh
   ```
4. Check if too many droplets exist:
   ```bash
   ./list_droplets.sh
   ```
   
Remember to clean up unused droplets:
```bash
./cleanup_droplets.sh
```
