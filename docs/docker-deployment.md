# Docker deployment

The project provides separate Docker targets for the credential-free dashboard
and the complete experiment pipeline.

This keeps the public dashboard lightweight while preserving a reproducible
environment for retrieval, evaluation, observability, and model experiments.

## Deployment targets

### Dashboard target

The `dashboard` target contains only the packages and files required to display
saved evaluation results:

- Streamlit
- Pandas
- Plotly
- Dashboard application
- Public-source metadata
- Verified benchmark
- Quality-gate configuration
- Versioned evaluation artifacts

It does not require hosted-model credentials, Ollama, PyTorch, FAISS, MLflow,
or access to external APIs.

### Pipeline target

The `pipeline` target installs the complete Python project and its experiment
dependencies. It supports data validation and the reproducible evaluation
commands documented in the repository.

Large local model files are not copied into the image. Ollama remains an
optional external local runtime.

## Build and run the dashboard

Build the lightweight dashboard image:

```bash
docker compose build
```
Start it:
```bash
docker compose up -d
```
Open:
```
http://localhost:8501
```
Inspect the service:
```bash
docker compose ps
```
Check container health:
```bash
CONTAINER_ID="$(docker compose ps -q dashboard)"

docker inspect \
  --format='{{.State.Status}} {{.State.Health.Status}}' \
  "$CONTAINER_ID"
```
Check the HTTP health endpoint:
```bash
python - <<'PY'
import urllib.request

with urllib.request.urlopen(
    "http://localhost:8501/_stcore/health",
    timeout=5,
) as response:
    print("status:", response.status)
    print("body:", response.read().decode())
PY
```
Stop and remove the container and Compose network:
```bash
docker compose down
```
## Tested dashboard result
The dashboard container was validated locally with:
- Container state: running
- Docker health status: healthy
- Health endpoint status: 200
- Health response body: ok
- Published port: 8501
- Image size: 878 MB
The original single-stage image was 10.8 GB because it installed the full
machine-learning and experiment stack for a saved-results dashboard.

Separating the dashboard and pipeline targets reduced the dashboard image size by approximately 91.9%.
## Build the complete pipeline image
Build the full experiment environment separately:
```bash
docker build \
  --target pipeline \
  --tag enterprise-rag-pipeline:local \
  .
```
Run its default source and benchmark validation:
```bash
docker run --rm \
  enterprise-rag-pipeline:local
```
Run another project command by overriding the default command:
```bash
docker run --rm \
  enterprise-rag-pipeline:local \
  python -m pytest -q
```
Experiments that write new reports can mount a local output directory. Saved
baseline reports required by the dashboard are already included in the
dashboard image.
## Port configuration
Port 8501 is used by default. Override the host port with:
```bash
STREAMLIT_PORT=8502 docker compose up -d
```
The dashboard will then be available at:
```bash
http://localhost:8502
```
## Security and operational controls
The images:
- Run as the unprivileged ragapp user with UID 10001
- Do not copy .env files into the build context
- Exclude local MLflow databases, traces, logs, caches, and virtual environments
- Include a Docker health check
- Disable Streamlit usage-statistics collection
- Use a fixed Python major and minor version
- Pin dashboard dependency versions
- Keep hosted-model credentials optional
Reproducibility boundaries
The saved-results dashboard works without network access after the image has
been built.

Building the image requires package-index access unless the required layers or packages are already cached.

Local Ollama models are intentionally not bundled because they are several
gigabytes and have separate licenses and runtime requirements.

The pipeline image provides the project dependencies and code, while individual hosted-model experiments still require their provider-specific credentials.