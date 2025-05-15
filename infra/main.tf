provider "digitalocean" {
  token = var.digitalocean_token
}

resource "digitalocean_droplet" "bn_trading" {
  image    = "docker-20-04"
  name     = "bn-trading-server"
  region   = "blr1"
  size     = "s-2vcpu-4gb"
  ssh_keys = var.ssh_key_ids
}

resource "digitalocean_floating_ip" "bn_trading_ip" {
  droplet_id = digitalocean_droplet.bn_trading.id
  region     = digitalocean_droplet.bn_trading.region
}

output "droplet_ip" {
  value = digitalocean_droplet.bn_trading.ipv4_address
}

output "floating_ip" {
  value = digitalocean_floating_ip.bn_trading_ip.ip_address
}
