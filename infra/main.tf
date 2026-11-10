resource "random_pet" "rg_name" {
  prefix = var.resource_group_name_prefix
}

resource "azurerm_resource_group" "rg" {
  location = var.resource_group_location
  name     = random_pet.rg_name.id
}

# Stores application and platform logs.
resource "azurerm_log_analytics_workspace" "logs" {
  name                = "pp-logs"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name

  sku               = "PerGB2018"
  retention_in_days = 30
}

# provides the container app infrastructure
resource "azurerm_container_app_environment" "pp" {
  name                = "pp-env"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name

  log_analytics_workspace_id = azurerm_log_analytics_workspace.logs.id
  # Sets a consumption profile for serverless hosting.
  workload_profile {
    name                  = "Consumption"
    workload_profile_type = "Consumption"
    minimum_count         = 0
    maximum_count         = 0
  }
}

resource "azurerm_container_app" "pp" {
  name                         = "prickly-partner"
  resource_group_name          = azurerm_resource_group.rg.name
  container_app_environment_id = azurerm_container_app_environment.pp.id

  revision_mode         = "Single"
  workload_profile_name = "Consumption"

  ingress {
    external_enabled           = true
    allow_insecure_connections = false
    target_port                = 80
    transport                  = "auto"

    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }

  template {
    min_replicas = 0
    max_replicas = 1

    http_scale_rule {
      name                = "http-requests"
      concurrent_requests = "10"
    }

    container {
      name   = "frontend"
      image  = var.frontend_image
      cpu    = 0.25
      memory = "0.5Gi"

      env {
        name  = "BACKEND_ADDRESS"
        value = "127.0.0.1:8000"
      }

      readiness_probe {
        transport               = "HTTP"
        port                    = 80
        path                    = "/"
        interval_seconds        = 10
        timeout                 = 3
        success_count_threshold = 1
      }
    }

    container {
      name   = "backend"
      image  = var.backend_image
      cpu    = 0.25
      memory = "0.5Gi"

      startup_probe {
        transport               = "HTTP"
        port                    = 8000
        path                    = "/health"
        interval_seconds        = 10
        timeout                 = 3
        failure_count_threshold = 10
      }

      readiness_probe {
        transport               = "HTTP"
        port                    = 8000
        path                    = "/health"
        interval_seconds        = 10
        timeout                 = 3
        success_count_threshold = 1
      }
    }
  }
}


