variable "digitalocean_token" {
  description = "DigitalOcean API token"
  type        = string
  sensitive   = true
}

variable "ssh_key_id" {
  description = "SSH key ID for droplet access"
  type        = string
}

variable "create_floating_ip" {
  description = "Whether to create a floating IP (set to false if you've reached your account limit)"
  type        = bool
  default     = false
}
