terraform {
  required_providers {
    digitalocean = {
      source = "digitalocean/digitalocean"
      version = "~> 2.0"
    }
  }
}

provider "digitalocean" {
  # The provider will use the DIGITALOCEAN_TOKEN environment variable if token is not specified
  token = var.digitalocean_token
}

resource "digitalocean_droplet" "bn_trading" {
  image    = "docker-20-04"
  name     = "bn-trading-server"
  region   = "blr1"
  size     = "s-2vcpu-4gb"
  ssh_keys = var.ssh_key_ids
}

# We're not using floating IPs due to account limits
# If you need to use an existing floating IP, you can uncomment and modify this section
# resource "digitalocean_floating_ip_assignment" "bn_trading_ip_assignment" {
#   ip_address = "your-existing-floating-ip"
#   droplet_id = digitalocean_droplet.bn_trading.id
# }

output "droplet_ip" {
  value = digitalocean_droplet.bn_trading.ipv4_address
}
