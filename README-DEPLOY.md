# VSLNet: GitHub Actions → Dokploy

This repository uses the same CI-before-deploy pattern as the Signify
MediaPipe-ASL service. A pull request to `main` builds and tests the inference
image. A push to `main`, or a manual run from `main`, requests a deployment from
Dokploy only after CI succeeds. The workflow does not push code, create a
Dokploy Application, or prove that a requested deployment later became healthy.

## What runs in the container

The Dockerfile uses Python 3.11, CPU PyTorch, FFmpeg, MediaPipe and the
inference-only dependency list. It copies the Git-tracked
`checkpoint/best_v1.pth` (400-gloss VSLNet weights) and
`dataset/root/gloss.csv` into the image. It does not train or change either
asset. The process runs as UID 10001, binds port 8001, and provides:

| Route | Purpose | Exposure |
|---|---|---|
| `GET /health` | Process and checkpoint startup health | Internal |
| `POST /predict` | Existing clip fallback from Spring Boot | Internal only |
| `WS /ws/vsl/stream` | Experimental browser frame stream, session token required | Public path only |

The clip route has no user authentication of its own, so **do not publish the
entire Python application at a public root domain**. Spring Boot remains the
authorization layer for clips. The WebSocket validates its own short-lived
token and exact browser Origin. The model is still isolated sign recognition,
not continuous sentence translation. Streaming has not been validated on real
annotated VSL video or measured against the two-second p95 goal.

## Configure one Dokploy Application

Create or reuse one VSLNet **Application** from this GitHub repository and
branch `main`. Use Build Type `Dockerfile`, repository root as context, path
`Dockerfile`, and container port `8001`. No separate WebSocket service is
required. Disable an additional GitHub auto-deploy webhook for this Application
so only the CI workflow triggers deploy. Keep one replica and one Python
process: the current Python sessions and Java token registry are in memory.

Assign a TLS-enabled domain such as `vsl.example.com` to **only** the public
path `/ws/vsl/stream`, with no Strip Path or Internal Path rewrite. Confirm
the resulting Traefik rule does not route `/predict`, `/docs`, `/openapi.json`
or `/health` publicly. Traefik handles the WebSocket upgrade on its HTTP router.
For Spring Boot to reach `/predict`, put both Applications on a verified shared
internal Docker network and use the VSLNet service's actual resolvable name.
Do not assume a Dokploy display name is a DNS name.

| Variable | Set in | Example or source |
|---|---|---|
| `SIGNIFY_VSL_STREAMING_SECRET` | VSLNet **and** Spring Boot | Same randomly generated value, at least 32 UTF-8 bytes |
| `VSLNET_ALLOWED_ORIGINS` | VSLNet | Exact frontend origin, e.g. `https://app.example.com` |
| `VSLNET_BACKEND_URL` | VSLNet | Spring Boot internal HTTP URL, e.g. `http://BACKEND_DNS:8080` |
| `SIGNIFY_VSL_WS_URL` | Spring Boot | Public browser URL, e.g. `wss://vsl.example.com/ws/vsl/stream` |
| `SIGNIFY_VSL_BASE_URL` | Spring Boot | VSLNet internal HTTP URL, e.g. `http://VSLNET_DNS:8001` |

Create the shared secret once and store it as a protected Dokploy variable (or
in a secret manager), never in the frontend or Git:

```powershell
[Convert]::ToBase64String([Security.Cryptography.RandomNumberGenerator]::GetBytes(48))
```

The built image already sets `VSLNET_HOST=0.0.0.0`, `VSLNET_PORT=8001` and
`VSLNET_POSE_MODE=cli`. The bind address is necessary for Dokploy's proxy to
reach the container. `video_to_pose` comes from the pinned `pose-format`
package; FFmpeg is installed by apt. Streaming uses the same FastAPI process
and does not launch a second server.

## GitHub repository secrets

Set these **in the VLSNet GitHub repository**. Secrets from the MediaPipe-ASL
repository are not shared automatically:

| Secret | Value |
|---|---|
| `DOKPLOY_URL` | HTTPS base URL of your Dokploy panel |
| `DOKPLOY_API_KEY` | Dokploy key allowed to deploy the Application |
| `DOKPLOY_APPLICATION_ID` | ID of the existing VSLNet Application |

The runtime `SIGNIFY_VSL_STREAMING_SECRET` belongs in Dokploy, not GitHub
Actions. The workflow calls Dokploy's `POST /api/application.deploy` only
after its image build, Python regression suite, and container health smoke
test pass. A successful API response only means Dokploy accepted the request;
check the Application Deployments and Logs tabs for the final result.

## Local build and smoke test

From this directory, if a Docker Linux engine is available:

```powershell
docker build -t vslnet:local .
Copy-Item .env.example .env
# Set SIGNIFY_VSL_STREAMING_SECRET in .env if testing WebSocket sessions.
docker compose config
docker compose up --build -d
docker compose ps
Invoke-RestMethod http://127.0.0.1:8001/health
docker compose logs --tail=100 vslnet
```

`docker-compose.yml` is only for local smoke testing. With an empty secret,
`/predict` can still start but WebSocket authentication fails closed. To test
the complete streaming path, run Spring Boot and the React call UI with a
real active call. The standalone healthcheck does not prove recognition
accuracy. Stop local containers with `docker compose down` when finished.

If a deployment fails, check the image build logs, CPU/RAM capacity, model
startup, health state, shared-network DNS, public path routing, TLS, exact
Origin, and both services' secret values. Roll back by redeploying a known
healthy Dokploy deployment or reverting the source commit on `main`; rerunning
an older GitHub workflow will still ask Dokploy to clone the Application's
current configured branch.

References: [Dokploy Dockerfile builds](https://docs.dokploy.com/docs/core/applications/build-type),
[domains](https://docs.dokploy.com/docs/core/domains),
[variables](https://docs.dokploy.com/docs/core/variables),
[application deploy API](https://docs.dokploy.com/docs/api/application),
[Traefik WebSocket routing](https://doc.traefik.io/traefik/user-guides/websocket/).
