# Project Antigravity — Deployment Guide

Three deployment options for different team profiles and use cases.

---

## Option A — Git Clone (Technical Analysts)

**Best for:** Analysts comfortable with the command line.

### Setup

```bash
# 1. Clone the private repository
git clone https://github.com/YOUR_ORG/antigravity.git
cd antigravity

# 2. Create virtual environment and install dependencies (requires uv)
uv venv --python 3.12
uv sync

# 3. Configure environment variables
cp .env.template .env
# Edit .env and fill in your API keys (OpenAI, Claude, etc.)
```

### Run — Interactive Mode

```bash
.venv\Scripts\python.exe tools/0_launcher.py
```

You will be prompted to:
1. Enter the data source path (e.g. `tmp/raw_data`)
2. Select workflow (full pipeline or individual modules)
3. Select delivery format (JSON, WebApp, Notebook, PowerPoint)
4. Select LLM provider (OpenAI, Claude, Kimi, Perplexity)

### Run — Non-Interactive Mode

```bash
.venv\Scripts\python.exe tools/0_launcher.py --non-interactive --provider openai
```

Optional overrides:

```bash
--source tmp/raw_data          # Path to the folder with input ZIP files
--provider openai              # LLM provider: claude | openai | kimi | perplexity
--modules descriptive regression  # Override module selection
```

### What to exclude from Git

The `.gitignore` already excludes:
- `.env` — contains secret API keys
- `tmp/` — pipeline outputs (large, per-run data)
- `.venv/` — Python virtual environment
- `test_input/` — raw survey ZIP files (client data, not to be committed)

---

## Option B — Streamlit Web UI (Non-Technical Users / Client Sharing)

**Best for:** Analysts who prefer a GUI, or when sharing results with a client.

### Prerequisites

Install dependencies first (same as Option A, steps 1–3).

### Run the Dashboard

```bash
.venv\Scripts\python.exe -m streamlit run app.py
```

Then open the URL shown in the terminal (typically `http://localhost:8501`).

### What the UI shows

- **Executive Overview** — confidence level, modules completed, key recommendations
- **Detailed Analysis** — expandable cards per module with charts (Plotly) and AI insights
  - Academic/Statistical commentary tab
  - Business/Marketing translation tab
- **Run Pipeline** section (bottom of the sidebar/main area) — allows running the full
  pipeline from the browser with provider and module selection, live log streaming,
  and auto-display of results on completion
- **Raw Data (JSON)** — collapsible export of the full delivery payload

### Sharing with the client

Run with `--server.address 0.0.0.0` to expose on the local network:

```bash
.venv\Scripts\python.exe -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

> **Note:** For external access, use a tunnel like [ngrok](https://ngrok.com/) or deploy to
> Streamlit Community Cloud / an internal server.

---

## Option C — Docker (IT-Managed / Reproducible Deployment)

**Best for:** IT-managed deployments, CI/CD pipelines, consistent environments.

### Prerequisites

- Docker Desktop installed (no Python or uv required)
- `.env` file configured with API keys

### Build the image

```bash
docker build -t antigravity .
```

### Run the pipeline

```bash
# Linux / macOS
docker run --rm \
  -v $(pwd)/tmp:/app/tmp \
  -v $(pwd)/.env:/app/.env \
  antigravity

# Windows PowerShell
docker run --rm `
  -v ${PWD}/tmp:/app/tmp `
  -v ${PWD}/.env:/app/.env `
  antigravity
```

The container runs in non-interactive mode by default. Output files are written
to `./tmp/` on the host via the bind mount.

### Run the Streamlit UI in Docker

```bash
docker run --rm -p 8501:8501 \
  -v $(pwd)/tmp:/app/tmp \
  -v $(pwd)/.env:/app/.env \
  antigravity streamlit run app.py --server.address 0.0.0.0
```

---

## Summary — Which Option to Use?

| Use Case | Recommended Option |
|---|---|
| Analyst comfortable with CLI | **Option A** (Git clone) |
| Analyst who prefers a GUI | **Option B** (Streamlit UI) |
| Sharing results with a client | **Option B** (share localhost/ngrok URL) |
| IT-managed or reproducible deployment | **Option C** (Docker) |
| CI/CD pipeline / automation | **Option C** (Docker) or **Option A** with `--non-interactive` |

---

## Environment Variables (.env)

Copy `.env.template` and fill in values:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
MOONSHOT_API_KEY=...        # for Kimi
PERPLEXITY_API_KEY=pplx-... # for Perplexity
```

Not all keys are required — only the provider you select.
