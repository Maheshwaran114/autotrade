# terraform.tfvars
# DigitalOcean API token (Create this in your DigitalOcean account under API > Tokens)
digitalocean_token = "YOUR_DO_TOKEN" # Replace with actual token in secure environment

# SSH key fingerprints from your DigitalOcean account
ssh_key_fingerprints = [
  "82:65:7b:a3:39:c6:06:c0:a7:b6:1b:15:ca:6c:28:73",  # GitHub Actions Deployment Key
  "fc:fc:d7:31:3e:1b:5f:ce:bb:02:f5:d9:e0:3b:f5:62",  # deploy_key_1747377542
  "42:5f:7d:c0:2f:8f:2f:0d:ca:f3:cf:55:0c:f3:b9:3a"   # deploy_key_1747377634
]
