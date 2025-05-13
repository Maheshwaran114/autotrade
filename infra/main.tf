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

output "droplet_ip" {
  value = digitalocean_droplet.bn-trading.ipv4_address
  description = "The public IP address of the Bank Nifty Trading server"
}
