output "resource_group_name" {
  description = "RG containing the app."
  value       = azurerm_resource_group.rg.name
}

output "app_url" {
  description = "Public HTTPS address of the app"
  value       = "https://${azurerm_container_app.pp.ingress[0].fqdn}"
}
