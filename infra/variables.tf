variable "digitalocean_token" {
  description = "DigitalOcean API token"
  type        = string
  sensitive   = true
}

variable "ssh_key_id" {
  description = "SSH key ID for droplet access"
  type        = string
}
