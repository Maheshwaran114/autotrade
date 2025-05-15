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

resource "digitalocean_floating_ip" "bn-trading-ip" {
  droplet_id = digitalocean_droplet.bn-trading.id
  region     = digitalocean_droplet.bn-trading.region
}

output "droplet_ip" {
  value = digitalocean_droplet.bn-trading.ipv4_address
  description = "The public IP address of the Bank Nifty Trading server"
}

output "floating_ip" {
  value = digitalocean_floating_ip.bn-trading-ip.ip_address
  description = "The floating IP address assigned to the Bank Nifty Trading server"
}
