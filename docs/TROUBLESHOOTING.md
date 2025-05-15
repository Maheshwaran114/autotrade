# CI/CD Pipeline Troubleshooting Guide

This guide provides solutions for common issues with the Bank Nifty Trading CI/CD pipeline.

## Common Issues and Solutions

### Docker Build Issues

**Symptom**: Docker image build hangs or fails with timeout errors.

**Solutions**:
1. Check if you're using dependencies that require extensive compilation. Use pre-built wheels when possible.
2. Ensure Docker has sufficient resources (memory, CPU) allocated.
3. Consider using `--build-arg BUILDKIT_INLINE_CACHE=1` for faster builds.
4. Use the provided `build_and_scan.sh` script locally to test the build process.

### Terraform Provider Issues

**Symptom**: Terraform fails with provider errors or unauthorized access.

**Solutions**:
1. Verify that the `DIGITALOCEAN_TOKEN` secret is correctly set in GitHub repository settings.
2. Ensure the token has sufficient permissions for creating resources.
3. Confirm that the required provider block is correctly configured in `main.tf`.
4. Try running Terraform locally with the same token to verify it works.

### SSH Connection Problems

**Symptom**: GitHub Actions workflow cannot connect to the provisioned droplet.

**Solutions**:
1. Verify that the `SSH_PRIVATE_KEY` and `SSH_KEY_ID` secrets are correctly set.
2. Ensure the SSH key is registered with DigitalOcean and the ID is correct.
3. Check if the droplet's firewall allows SSH connections (port 22).
4. Increase the wait time for the droplet to initialize (currently set to 30 attempts).

### Application Deployment Issues

**Symptom**: Docker containers start but the application is not accessible.

**Solutions**:
1. SSH into the droplet and check Docker container logs: `docker-compose logs`.
2. Verify that the application is binding to the correct port and IP (`0.0.0.0`).
3. Ensure the firewall allows connections to port 5000.
4. Check if the database container is running and properly connected.

### Terraform Output Issues

**Symptom**: The workflow cannot extract the floating IP from Terraform outputs.

**Solutions**:
1. Manually run `terraform output -json` to see the structure of the outputs.
2. Verify that the `floating_ip` output is defined in the Terraform configuration.
3. Try the fallback mechanism using the droplet's IP instead if the floating IP isn't available.

## Debugging Steps

1. Check the GitHub Actions workflow logs for each job.
2. Examine the deploy script's output to identify the exact step that failed.
3. SSH into the droplet to investigate:
   ```
   ssh root@<droplet_ip>
   cd /bn-trading
   docker-compose ps
   docker-compose logs
   ```
4. Test the application's endpoints manually:
   ```
   curl http://localhost:5000/
   curl http://localhost:5000/health
   ```

## Security Considerations

1. Use a dedicated DigitalOcean API token with limited permissions.
2. Ensure SSH keys are properly secured and rotated regularly.
3. Consider using Docker secrets for sensitive information.
4. Run regular vulnerability scans using the provided `build_and_scan.sh` script.

## Additional Resources

- [DigitalOcean API Documentation](https://docs.digitalocean.com/reference/api/)
- [Terraform DigitalOcean Provider](https://registry.terraform.io/providers/digitalocean/digitalocean/latest/docs)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
