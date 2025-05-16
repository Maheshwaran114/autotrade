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

# Make floating IP optional by using 'count'
resource "digitalocean_floating_ip" "bn_trading_ip" {
  count      = var.use_floating_ip ? 1 : 0
  droplet_id = digitalocean_droplet.bn_trading.id
  region     = digitalocean_droplet.bn_trading.region
}

output "droplet_ip" {
  value = digitalocean_droplet.bn_trading.ipv4_address
}

output "floating_ip" {
  value = var.use_floating_ip && length(digitalocean_floating_ip.bn_trading_ip) > 0 ? digitalocean_floating_ip.bn_trading_ip[0].ip_address : "No floating IP created"
}
