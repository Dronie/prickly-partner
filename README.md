# Prickly Partner

A browser-based coin game where a human plays against a trained reinforcement-learning policy. The project brings together a TypeScript game, Python model inference, ONNX export tooling, container publishing and Terraform deployment to Azure Container Apps.

[Play online](https://pp.stefan-roesch.com)

The hosted app is configured to scale to zero when idle, so the first request after inactivity may take longer while the containers and models start.

## Play the game

Click **Play Now**, then choose **Up**, **Down**, **Left**, **Right** or **Stay** using the on-screen buttons. You are red; the AI is blue. Both players move simultaneously after you choose an action.

- Play takes place on a 5 × 5 grid. Movement stops at the edges.
- Collecting your own coin earns you **+1**.
- Collecting the other player's coin earns you **+1** and gives them **−2**.
- Collected coins respawn. After 100 steps, a new episode starts automatically; the displayed scores carry over.
- The episode-ending step returns zero rewards, matching the Python environment's reset behaviour.

The browser currently selects AI model **1**, with opponent ID **0**. There is no model-selection UI. Game state lives in the browser and is lost on refresh; the backend only handles inference requests.

## Run locally with Docker

Install Docker with Docker Compose, then run these commands from the repository root:

```bash
docker compose up --build -d
```

Open **http://localhost:8080**. The frontend serves the page and forwards API requests to the backend. The ten ONNX models in `web/models/` are included in the backend image; exporting or training models is not required to play.

Check readiness and view logs:

```bash
curl --fail --show-error http://localhost:8080/health
docker compose logs -f
```

A ready backend returns:

```json
{"status":"ready","loaded_models":10}
```

Stop the app with:

```bash
docker compose down
```

## Develop locally

Use **Node.js 24**, **Python 3.10** and **uv** to match the container and CI setup. Run both processes from the repository root in separate terminals.

Backend:

```bash
uv sync --locked --no-default-groups --python 3.10
uv run --no-sync python -m uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

Frontend:

```bash
npm ci
npm run dev
```

Open the URL printed by Vite. It updates the page as you edit and proxies `/api` requests to port 8000. Check backend health directly at **http://127.0.0.1:8000/health**; the Vite proxy does not currently forward `/health`.

Build the production frontend with `npm run build`. Output is written to `dist/`.

## How it works

```text
Browser: TypeScript game state and canvas rendering
    │ POST /api/action
    ▼
Nginx (production) or Vite proxy (development)
    ▼
FastAPI: request validation and service lifecycle
    ▼
ONNX Runtime: cached CPU inference sessions
    │ {"action": 0..4}
    ▼
Browser: apply both moves, calculate rewards and redraw
```

The backend loads all ten ONNX models at startup and checks their input/output interfaces. Each request supplies 100 binary observation values; the inference service appends a ten-value one-hot opponent encoding, producing the model's 110-value input. It converts the five output logits to probabilities and samples an action, so repeated requests need not return the same action.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Reports whether all ten models are loaded |
| `POST /api/action` | Validates an observation and returns a sampled action |
| `/docs` on the backend port | FastAPI's interactive API documentation |

The action request contains `observation` (100 integers, each 0 or 1), `agent` (`red` or `blue`), `agent_id` (0–9) and `opponent_id` (0–9). The caller supplies the observation in the acting agent's perspective; the backend does not transform it based on `agent`.

Action IDs are **0 right, 1 left, 2 up, 3 down, 4 stay**. Invalid request payloads return HTTP 422. Nginx exposes `/api/` and `/health`, but does not proxy the backend's `/docs` route.

## Tests and CI

```bash
uv sync --locked --no-default-groups --extra test --python 3.10
uv run --no-sync python -m pytest
npm ci
npm test
npm run build
```

The Python tests cover basic environment behaviour, backend health, inference responses and request validation. Vitest checks the TypeScript game's initial state, simultaneous moves, observation layout and rewards.

GitHub Actions provides three workflows:

| Workflow | What it does |
| --- | --- |
| [CI](.github/workflows/ci.yml) | Python syntax checks and tests, TypeScript tests/build, and Docker Compose smoke tests for the frontend, health and inference |
| [Validate exporter](.github/workflows/export.yml) | Exports twice, checks model counts and compares filename-aware ONNX checksums; runs manually or for relevant pull-request changes |
| [Publish container images](.github/workflows/publish-images.yml) | Manually builds and pushes frontend/backend images to GHCR, tagged with the source commit |

Publishing images does not deploy them or automatically wait for CI. Run it for a commit whose checks have passed.

## Export models to ONNX

The repository retains Flax/Orbax checkpoints under `models/meta_adv/` and exported policies under `web/models/`. Exporting uses the optional `export` dependencies; the inference backend does not need JAX or Flax.

Build the exporter and write its results to a separate host directory:

```bash
docker build -f Dockerfile.export -t prickly-partner-exporter .
mkdir -p exported-models

docker run --rm \
  --mount type=bind,source="$(pwd)/models",target=/app/models,readonly \
  --mount type=bind,source="$(pwd)/exported-models",target=/app/web/models \
  prickly-partner-exporter
```

The container exports each immediate model directory under `models/meta_adv/`, writes `.onnx` files and `model-count.txt` into `exported-models/`, then exits. Each source model directory must contain `renderer_policy_config.json` and `checkpoint_0/`.

The exporter is tailored to the existing two-hidden-layer, 64-unit ReLU actor architecture. It checks ONNX validity and compares ONNX and Flax outputs on three inputs with `rtol=atol=1e-5`. It is not a general-purpose converter for arbitrary models. The repeated-export workflow checks byte-for-byte reproducibility separately from numerical agreement.

To deploy newly exported policies, replace the intended files in `web/models/`, run the tests and publish a new backend image. The service expects `z=-1.0_agent_0_model.onnx` through `z=-1.0_agent_9_model.onnx`.

## Deploy to Azure Container Apps

[infra/](infra/) defines a resource group, Log Analytics workspace, Consumption environment and one Container App containing both containers. Each container receives 0.25 vCPU and 0.5 GiB RAM. The app uses HTTPS ingress, startup/readiness probes and an HTTP scaling rule with zero to one replicas.

The same frontend image works locally and in Azure through `BACKEND_ADDRESS`:

| Environment | Backend address |
| --- | --- |
| Docker Compose | `backend:8000` |
| Two containers in one Azure Container App | `127.0.0.1:8000` |

Nginx substitutes this variable into `nginx.conf.template` when the container starts.

The configuration currently targets the project's HCP Terraform organisation and workspace in `infra/providers.tf`. To deploy your own copy, configure your own workspace and Azure credentials first. The existing authentication approach uses `ARM_CLIENT_ID`, `ARM_CLIENT_SECRET`, `ARM_TENANT_ID` and `ARM_SUBSCRIPTION_ID` as HCP environment variables, with the client secret marked sensitive. Register the Azure resource providers `Microsoft.App` and `Microsoft.OperationalInsights` in the target subscription.

After publishing images, set HCP **Terraform variables** `frontend_image` and `backend_image` to the full GHCR addresses, for example:

```text
ghcr.io/dronie/prickly-partner-frontend:sha-<full-commit-hash>
ghcr.io/dronie/prickly-partner-backend:sha-<full-commit-hash>
```

Then run:

```bash
terraform -chdir=infra init
terraform -chdir=infra fmt -check
terraform -chdir=infra validate
terraform -chdir=infra plan -out=main.tfplan
# Review the plan before applying: this creates billable Azure resources.
terraform -chdir=infra apply main.tfplan
terraform -chdir=infra output -raw app_url
```

The custom domain and managed certificate were configured separately in Azure and GoDaddy; they are not declared in the current Terraform files. Review future plans for changes to that binding. Scaling to zero reduces idle compute costs, but logging and other usage can still incur charges.

[vm_infra/](vm_infra/) preserves the earlier VM approach as a reference. It points to the same HCP workspace as `infra/`; do not apply it against the active Container Apps deployment.

## Original desktop viewer

The original JAX environment and Pygame viewer remain available for local two-player keyboard play and Flax checkpoint playback:

```bash
uv sync --locked --no-default-groups --extra export --python 3.10
uv run --no-sync python coin_game.py
```

This requires a graphical desktop. Red uses **W/A/S/D**, blue uses **I/J/K/L**. During model playback, **Space** pauses/resumes and **N** advances one step; **Escape/Q** exits. Select checkpoint directories on the start screen. A `renderer_policy_config.json` beside each checkpoint helps the viewer reconstruct its policy configuration.

## Repository guide

| Path | Contents |
| --- | --- |
| `src/` | Browser game, rendering, API client and styling |
| `backend/` | FastAPI endpoints and ONNX inference service |
| `web/models/` | Ten ONNX policies used by the backend |
| `models/meta_adv/` | Source checkpoints and policy manifests |
| `tools/` | Checkpoint-to-ONNX export scripts |
| `tests/` | Python and TypeScript tests |
| `coin_game.py` | Original JAX game and desktop viewer |
| `Dockerfile.*`, `compose.yaml` | Runtime and exporter containers; local orchestration |
| `infra/` | Active Azure Container Apps Terraform configuration |
| `vm_infra/` | Previous VM infrastructure, retained for reference |
| `.github/workflows/` | CI, export validation and image publishing |
