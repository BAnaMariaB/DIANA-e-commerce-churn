#!/usr/bin/env bash
# Delete the whole resource group — nukes every Azure resource the deploy
# script created (Container App, environment, ACR, log analytics).
# Useful before re-deploying from scratch or to stop any (small) running cost.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "ERROR: ${ENV_FILE} not found." >&2
    exit 1
fi
# shellcheck disable=SC1090
source "${ENV_FILE}"

: "${AZURE_SUBSCRIPTION_ID:?}"
: "${RESOURCE_GROUP:?}"

az account set --subscription "${AZURE_SUBSCRIPTION_ID}"

echo "About to delete resource group: ${RESOURCE_GROUP}"
read -r -p "Are you sure? [y/N] " confirm
if [[ "${confirm}" != "y" && "${confirm}" != "Y" ]]; then
    echo "Aborted."
    exit 0
fi

az group delete --name "${RESOURCE_GROUP}" --yes --no-wait
echo "Deletion initiated (runs in background)."
