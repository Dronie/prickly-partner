output "resource_group_name" {
  value = azurerm_resource_group.rg.name
}

output "public_ip_address" {
  value = azurerm_linux_virtual_machine.pp_vm.public_ip_address
}

output "ssh_private_key" {
  value = try(
    azapi_resource_action.ssh_public_key_gen.output.privateKey,
    jsondecode(azapi_resource_action.ssh_public_key_gen.output).privateKey
  )
  sensitive = true
}
