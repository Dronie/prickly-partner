terraform {
  required_version = ">= 1.6.0"

  cloud {
    organization = "sroesch-terraform"

    workspaces {
      project = "prickly-partner"
      name    = "prickly-partner-prod"
    }
  }

  required_providers {
    azapi = {
      source  = "azure/azapi"
      version = "~>1.5"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~>3.117.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~>3.0"
    }
  }
}

provider "azurerm" {
  features {}
}
