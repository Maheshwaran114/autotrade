# GitHub Actions Workflow Testing Checklist

Use this checklist to verify that all fixes to the CI/CD pipeline are working correctly.

## Pre-deployment Checks

- [ ] Verify all required GitHub secrets are set:
  - `DIGITALOCEAN_TOKEN`
  - `SSH_KEY_ID`
  - `SSH_PRIVATE_KEY` 
  - `DOCKERHUB_USERNAME`
  - `DOCKERHUB_TOKEN`

- [ ] Check existing DigitalOcean droplets:
  ```bash
  export DIGITALOCEAN_TOKEN=your_token
  ./list_droplets.sh
  ```

- [ ] Clean up unnecessary droplets if needed:
  ```bash
  export DIGITALOCEAN_TOKEN=your_token
  ./cleanup_droplets.sh
  ```

## Testing the SSH Connectivity Locally

- [ ] Test SSH connectivity with local script:
  ```bash
  ./test_ssh_connection.sh
  ```

- [ ] Verify SSH key format is correct:
  ```bash
  # SSH private key should begin with
  -----BEGIN OPENSSH PRIVATE KEY-----
  # or
  -----BEGIN RSA PRIVATE KEY-----
  ```

## Workflow Testing

- [ ] Push a small change to trigger the workflow
- [ ] Monitor these key steps in the workflow:
  - ✓ Docker build and push
  - ✓ Check for existing droplets
  - ✓ Terraform apply (if no existing droplet)
  - ✓ Extract droplet IP
  - ✓ SSH connection
  - ✓ Application deployment
  - ✓ Deployment verification
  - ✓ Droplet count warning

## Post-deployment Verification

- [ ] Verify the app is running:
  ```bash
  curl http://<droplet_ip>:5000/
  ```

- [ ] SSH into the droplet and check Docker containers:
  ```bash
  ssh root@<droplet_ip>
  cd /bn-trading
  docker-compose ps
  docker-compose logs
  ```

## Troubleshooting

If the workflow fails, check the following:

1. **SSH Issues**:
   - Review the SSH retry logs
   - Verify SSH key is correctly formatted in GitHub secrets
   - Try the `test_ssh_connection.sh` script

2. **Terraform Issues**:
   - Check the raw Terraform output
   - Verify the DigitalOcean token has correct permissions
   - Try running Terraform locally

3. **Docker Issues**:
   - Verify Docker Hub credentials
   - Check if the image builds locally
   - Ensure ports are correctly configured

4. **Multiple Droplet Issues**:
   - Use `list_droplets.sh` to see all droplets
   - Use `cleanup_droplets.sh` to remove unwanted droplets
   - Check if the workflow is selecting the correct droplet

Refer to the [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) document for more detailed solutions.
