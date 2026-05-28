#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# DIANA — One-shot Azure Container Apps deployment.
#
# What this script does:
#   1. Loads env vars from azure/.env
#   2. Creates the resource group (idempotent).
#   3. Creates an Azure Container Registry (Basic SKU — cheapest tier).
#   4. Builds the Docker image *inside Azure* with `az acr build` (no local
#      Docker daemon needed; the build runs on Azure-managed infra).
#   5. Creates a Container Apps environment.
#   6. Creates / updates the Container App with scale-to-zero (min 0, max 2),
#      0.5 vCPU + 1 GiB RAM — well within the Container Apps free monthly
#      grant (180 000 vCPU-sec + 360 000 GiB-sec).
#   7. Prints the public FQDN of the deployed API.
#
# Prereqs:
#   - Azure CLI installed and `az login` done.
#   - data/raw/ECommerceDataset2.xlsx present in the repo root (used during
#     the image build to train the model).
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "ERROR: ${ENV_FILE} not found. Copy azure/.env.example to azure/.env and edit it." >&2
    exit 1
fi

# shellcheck disable=SC1090
source "${ENV_FILE}"

: "${AZURE_SUBSCRIPTION_ID:?must be set in azure/.env}"
: "${RESOURCE_GROUP:?must be set in azure/.env}"
: "${LOCATION:?must be set in azure/.env}"
: "${ACR_NAME:?must be set in azure/.env}"
: "${ACA_ENVIRONMENT:?must be set in azure/.env}"
: "${ACA_APP_NAME:?must be set in azure/.env}"
: "${IMAGE_TAG:=v1}"

IMAGE_NAME="diana-churn-api"
IMAGE_REF="${ACR_NAME}.azurecr.io/${IMAGE_NAME}:${IMAGE_TAG}"

echo "── Azure context ─────────────────────────────────────────────"
echo "Subscription : ${AZURE_SUBSCRIPTION_ID}"
echo "Resource grp : ${RESOURCE_GROUP}"
echo "Location     : ${LOCATION}"
echo "ACR          : ${ACR_NAME}"
echo "Env          : ${ACA_ENVIRONMENT}"
echo "App          : ${ACA_APP_NAME}"
echo "Image        : ${IMAGE_REF}"
echo "──────────────────────────────────────────────────────────────"

az account set --subscription "${AZURE_SUBSCRIPTION_ID}"

# Make sure the Container Apps extension + all required resource providers
# are registered (free, one-off per subscription). New free subscriptions
# don't have these turned on by default; without them `az group create` and
# `az acr create` fail with MissingSubscriptionRegistration.
az extension add --name containerapp --upgrade --only-show-errors >/dev/null
az provider register --namespace Microsoft.App --wait --only-show-errors >/dev/null
az provider register --namespace Microsoft.OperationalInsights --wait --only-show-errors >/dev/null
az provider register --namespace Microsoft.ContainerRegistry --wait --only-show-errors >/dev/null

echo
echo "[1/6] Resource group"
az group create \
    --name "${RESOURCE_GROUP}" \
    --location "${LOCATION}" \
    --only-show-errors >/dev/null

echo "[2/6] Azure Container Registry (Basic SKU)"
az acr create \
    --resource-group "${RESOURCE_GROUP}" \
    --name "${ACR_NAME}" \
    --sku Basic \
    --admin-enabled true \
    --only-show-errors >/dev/null

echo "[3/6] Build image inside ACR (this runs the training pipeline; ~5–10 min)"
az acr build \
    --registry "${ACR_NAME}" \
    --image "${IMAGE_NAME}:${IMAGE_TAG}" \
    --file Dockerfile \
    "${REPO_ROOT}"

echo "[4/6] Container Apps environment"
az containerapp env create \
    --name "${ACA_ENVIRONMENT}" \
    --resource-group "${RESOURCE_GROUP}" \
    --location "${LOCATION}" \
    --only-show-errors >/dev/null

# Pull ACR admin creds so the Container App can authenticate against the registry.
ACR_USERNAME="$(az acr credential show -n "${ACR_NAME}" --query username -o tsv)"
ACR_PASSWORD="$(az acr credential show -n "${ACR_NAME}" --query 'passwords[0].value' -o tsv)"

echo "[5/6] Create or update Container App"
if az containerapp show \
        --name "${ACA_APP_NAME}" \
        --resource-group "${RESOURCE_GROUP}" \
        --only-show-errors >/dev/null 2>&1; then
    echo "  (updating existing app)"
    az containerapp update \
        --name "${ACA_APP_NAME}" \
        --resource-group "${RESOURCE_GROUP}" \
        --image "${IMAGE_REF}" \
        --only-show-errors >/dev/null
    az containerapp registry set \
        --name "${ACA_APP_NAME}" \
        --resource-group "${RESOURCE_GROUP}" \
        --server "${ACR_NAME}.azurecr.io" \
        --username "${ACR_USERNAME}" \
        --password "${ACR_PASSWORD}" \
        --only-show-errors >/dev/null
else
    echo "  (creating new app)"
    az containerapp create \
        --name "${ACA_APP_NAME}" \
        --resource-group "${RESOURCE_GROUP}" \
        --environment "${ACA_ENVIRONMENT}" \
        --image "${IMAGE_REF}" \
        --registry-server "${ACR_NAME}.azurecr.io" \
        --registry-username "${ACR_USERNAME}" \
        --registry-password "${ACR_PASSWORD}" \
        --target-port 8000 \
        --ingress external \
        --cpu 0.5 \
        --memory 1.0Gi \
        --min-replicas 0 \
        --max-replicas 2 \
        --env-vars "PORT=8000" \
        --only-show-errors >/dev/null
fi

echo "[6/6] Reading public endpoint"
FQDN="$(az containerapp show \
    --name "${ACA_APP_NAME}" \
    --resource-group "${RESOURCE_GROUP}" \
    --query properties.configuration.ingress.fqdn -o tsv)"

cat <<EOF

✅ Deployment complete.

   URL    : https://${FQDN}
   Health : https://${FQDN}/health
   Docs   : https://${FQDN}/docs
   Info   : https://${FQDN}/info
   Predict: POST https://${FQDN}/predict   (see azure/sample_request.json)

   Sample call:
     curl -s -X POST https://${FQDN}/predict \\
       -H 'Content-Type: application/json' \\
       --data @azure/sample_request.json

   To tear everything down:
     bash azure/teardown.sh

EOF
