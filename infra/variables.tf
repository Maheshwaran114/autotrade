variable "digitalocean_token" {
  description = "DigitalOcean API token"
  type        = string
  sensitive   = true
}

variable "ssh_key_fingerprints" {
  description = "List of SSH key fingerprints to add to the droplet"
  type        = list(string)
  default     = []
}
