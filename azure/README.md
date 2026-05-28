# Azure Deployment

This folder contains everything needed to deploy the DIANA churn prediction
API to **Azure Container Apps** on a free / very-low-cost setup.

## Why Container Apps?

| Service | Cost on free tier | Fit for this project |
|---|---|---|
| **Azure Container Apps** | 180 000 vCPU-sec + 360 000 GiB-sec / month free, scale-to-zero | ✅ Best fit — serverless containers, no idle cost |
| Azure App Service (F1) | Free, but 60 min CPU/day cap and no custom containers on F1 | ❌ Custom container needs B1 (~$13/mo) |
| Azure ML Online Endpoint | No free tier, ~$10+/mo minimum | ❌ Overkill, costs money |

With scale-to-zero (`min-replicas 0`) the app costs **€0** when nobody is
calling it, and only spins up on demand.

## Architecture

```
   ┌────────────┐      git push      ┌────────────────────┐
   │ Developer  ├──────────────────▶ │  GitHub repo       │
   └────────────┘                    └─────────┬──────────┘
                                               │ (manual or GH Actions)
                                               ▼
                           ┌─────────────────────────────────┐
                           │ az acr build (cloud build, no   │
                           │   local Docker needed)          │
                           └─────────────────┬───────────────┘
                                             ▼
            ┌───────────────────────┐    pulls image   ┌───────────────────────┐
            │ Azure Container       │ ◀────────────────│ Azure Container Apps  │
            │ Registry (Basic)      │                  │   diana-churn-api     │
            └───────────────────────┘                  │   ingress: external   │
                                                       │   scale: 0–2 replicas │
                                                       └─────────┬─────────────┘
                                                                 │ HTTPS
                                                                 ▼
                                                       ┌───────────────────────┐
                                                       │  POST /predict        │
                                                       │  GET  /health /info   │
                                                       └───────────────────────┘
```

## Prereqs

1. **Azure CLI** ≥ 2.60. Install: <https://learn.microsoft.com/cli/azure/install-azure-cli>
2. Authenticated: `az login`
3. The raw dataset present at `data/raw/ECommerceDataset2.xlsx` (the Docker
   build trains the model inside the builder stage and bakes the artifact
   into the image).

## One-shot deploy

```bash
# 1. Configure
cp azure/.env.example azure/.env
# Edit azure/.env — set AZURE_SUBSCRIPTION_ID and a unique ACR_NAME

# 2. Deploy
bash azure/deploy.sh
```

The script:

1. Creates the resource group.
2. Creates an Azure Container Registry (Basic SKU — ~€0.15/day, deleted by
   `teardown.sh`).
3. Builds the image inside Azure via `az acr build` (no local Docker daemon
   needed). The build runs `src/training/save_artifact.py`, so the trained
   model lands inside the image.
4. Creates a Container Apps environment.
5. Deploys the Container App with `min-replicas 0` (scale-to-zero) and
   `max-replicas 2`.
6. Prints the public HTTPS URL.

Expected end-to-end time: **6–10 minutes** (most of it is the cloud build
running training).

## Test it

```bash
# Replace with the FQDN printed by deploy.sh
APP_URL="https://diana-churn-api.<random>.westeurope.azurecontainerapps.io"

curl -s "${APP_URL}/health"
curl -s "${APP_URL}/info" | python -m json.tool

curl -s -X POST "${APP_URL}/predict" \
     -H 'Content-Type: application/json' \
     --data @azure/sample_request.json | python -m json.tool
```

Expected response shape:

```json
{
  "predictions": [1],
  "probabilities": [0.84],
  "model_name": "XGBoost",
  "n_features": 47
}
```

## Tear down

```bash
bash azure/teardown.sh
```

Deletes the whole resource group. Everything else (subscription, account)
is untouched.

## CI/CD

`.github/workflows/deploy-azure.yml` redeploys the app whenever `main` is
updated. Required GitHub repository secrets:

| Secret | Where to get it |
|---|---|
| `AZURE_CREDENTIALS` | `az ad sp create-for-rbac --name diana-churn-cicd --role contributor --scopes /subscriptions/<SUB_ID>/resourceGroups/<RG> --sdk-auth` |
| `AZURE_RG` | Your resource group name |
| `AZURE_ACR_NAME` | Your ACR name |
| `AZURE_ACA_APP_NAME` | Your Container App name |

## Cost notes

A typical demo run for the grader (build + a few requests + 24 h running):

- ACR Basic: ~€0.15
- Container Apps build + idle (scale-to-zero): **€0**
- Container Apps active request time: well within the free monthly grant

Run `bash azure/teardown.sh` once grading is over to be sure.
