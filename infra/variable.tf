variable "resource_group_location" {
  type        = string
  default     = "uksouth"
  description = "Location of the resource group."
}

variable "resource_group_name_prefix" {
  type        = string
  default     = "rg"
  description = "Prefix of the resource group name."
}

variable "frontend_image" {
  type        = string
  description = "Full GHCR frontend image address, including tag or digest"
}

variable "backend_image" {
  type        = string
  description = "Full GHCR backend image address, including tag or digest"
}
