terraform {
  required_providers {
    digitalocean = {
      source = "digitalocean/digitalocean"
      version = "~> 2.0"
    }
  }
}

provider "digitalocean" {
  token = var.digitalocean_token
}

resource "digitalocean_droplet" "bn-trading" {
  image  = "docker-20-04"
  name   = "bn-trading-server"
  region = "blr1"
  size   = "s-2vcpu-4gb"
  ssh_keys = [var.ssh_key_id]
}

# Make floating IP optional using count
resource "digitalocean_floating_ip" "bn-trading-ip" {
  # Only create this resource if the create_floating_ip variable is true
  count      = var.create_floating_ip ? 1 : 0
  droplet_id = digitalocean_droplet.bn-trading.id
  region     = digitalocean_droplet.bn-trading.region
}

output "droplet_ip" {
  value = digitalocean_droplet.bn-trading.ipv4_address
  description = "The public IP address of the Bank Nifty Trading server"
}

output "floating_ip" {
  # Use the droplet's regular IP if no floating IP is created
  value = var.create_floating_ip ? digitalocean_floating_ip.bn-trading-ip[0].ip_address : digitalocean_droplet.bn-trading.ipv4_address
  description = "The IP address to use for the Bank Nifty Trading server (floating IP if available, otherwise regular IP)"
}
