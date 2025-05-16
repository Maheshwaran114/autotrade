variable "digitalocean_token" {
  description = "DigitalOcean API token"
  type        = string
  sensitive   = true
}

variable "ssh_key_ids" {
  description = "List of SSH key IDs to add to the droplet"
  type        = list(string)
  default     = []
}

variable "use_floating_ip" {
  description = "Whether to create and use a floating IP"
  type        = bool
  default     = false
}
