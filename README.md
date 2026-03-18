# WealthWise Agent

Autonomous AI financial advisor that analyzes transaction history and forecasts spending trends via natural language queries.

**Live demo:** http://3.89.224.121:8501

---

## Features

- Natural language financial queries ("How much did I spend on food last month?")
- ML-based spending forecasts — scikit-learn Pipeline with 4 temporal features
- Four interchangeable LLM backends selectable from the sidebar without restarting the app:
  - Claude Haiku 3.5 (`claude-haiku-3-5-20251001`)
  - Claude Sonnet 4.5 (`claude-sonnet-4-5-20251014`)
  - Gemini 2.0 Flash (`gemini-2.0-flash`)
  - Groq Llama 3.3 70B (`llama-3.3-70b-versatile`)
- Per-query token usage display and context window progress bar
- Persistent conversation memory via LangGraph `SqliteSaver` (SQLite checkpoints)
- Automatic provider fallback if primary LLM initialisation fails
- User-facing error messages for rate limits, quota exhaustion, and tool failures
- Groq tool-call retry logic (up to 3 attempts with fresh agent rebuild per attempt)
- Structured logging with per-query elapsed time across all modules
- ML model S3 fallback — downloads `spending_model.pkl` from S3 if not found locally
- CI/CD pipeline with pytest coverage gate (80% floor) blocking deploy on failure

## Architecture

### Stack

| Layer | Technology |
|---|---|
| Agent runtime | LangGraph `StateGraph` — custom ReAct loop with `_should_continue` conditional routing |
| LLM providers | Claude Haiku 3.5 · Claude Sonnet 4.5 · Gemini 2.0 Flash · Groq Llama 3.3 70B |
| Transaction analysis | `PythonAstREPLTool` (`python_analyst`) — pandas DataFrame, pre-loaded from CSV |
| Spending forecast | `predict_spending_trend` `@tool` — scikit-learn Pipeline, 4-feature LinearRegression |
| Conversation memory | `LangGraph SqliteSaver` → `data/checkpoints.db` |
| Context window limits | Claude: 200k · Gemini: 1M · Groq: 128k |
| ML experiment tracking | MLflow (SQLite backend → `mlflow.db`) |
| Frontend | Streamlit |
| Config & validation | `pydantic-settings` — fails fast on missing API keys at startup |
| Package management | uv |
| Infrastructure | Docker · Docker Compose · AWS EC2 (t2.micro) |
| CI/CD | GitHub Actions — `test` job gates `deploy` job (`needs: test`) |

### Agent Flow

```
User query
    │
    ▼
Streamlit (app.py)
    │  thread_id (UUID per session)
    ▼
process_query(query, provider, thread_id)
    │
    │  builds fresh StateGraph + SqliteSaver on each call
    ▼
┌──────────────────────────────────────────────┐
│  StateGraph (AgentState: messages list)       │
│                                               │
│  ┌────────────┐   tool_calls present          │
│  │ agent node │──────────────────────────┐    │
│  │   (LLM)    │                          │    │
│  └────────────┘                          ▼    │
│        │                         ┌──────────┐ │
│        │ no tool_calls           │  tools   │ │
│        │                         │  node    │ │
│        ▼                         └──────────┘ │
│       END ◀── "PREDICTION COMPLETE" ──────────┘
│                in ToolMessage                  │
│                                               │
│       END ◀── no tool_calls in AIMessage      │
└──────────────────────────────────────────────┘
    │
    ▼
_extract_response() + _extract_usage()
    │
    ▼
(response_text, usage_dict) → Streamlit
```

**Routing logic (`_should_continue`):**
- `AIMessage` with no `tool_calls` → `END`
- `AIMessage` with `tool_calls` → `"tools"`
- `ToolMessage` containing `"PREDICTION COMPLETE"` → `END`
- Any other `ToolMessage` → `"agent"` (continue loop)

## Project Structure

```
wealthwise-agent/
├── app.py                        # Streamlit UI — chat, model selector sidebar, token usage display
├── src/
│   ├── __init__.py               # Package marker
│   ├── config.py                 # pydantic-settings Settings — validates API keys on startup
│   ├── graph.py                  # StateGraph agent, get_llm(), process_query(), retry logic
│   ├── tools.py                  # predict_spending_trend @tool, _build_python_analyst factory
│   └── logger.py                 # Structured logging setup shared across all modules
├── scripts/
│   ├── generate_data.py          # Generates synthetic transactions.csv (90-day trend data)
│   └── train_pipeline.py         # Trains scikit-learn Pipeline, MLflow tracking, S3 upload
├── tests/
│   ├── __init__.py               # Package marker
│   ├── test_graph.py             # 18 tests — get_llm, _extract_response, process_query, routing
│   └── test_tools.py             # 18 tests — ForecastInput validator, predict tool, S3 fallback
├── data/
│   └── checkpoints.db            # SqliteSaver conversation memory (auto-created at runtime)
├── models/                       # Trained model artifact (generated by train_pipeline.py)
├── pyproject.toml                # Project metadata and all dependencies
├── uv.lock                       # Pinned lockfile (155 packages)
├── Dockerfile                    # uv-based image — copies uv binary from ghcr.io/astral-sh/uv
├── docker-compose.yml            # Single-service compose — maps port 8501, reads .env
├── deploy.sh                     # EC2 deploy — clone-or-pull, prune, disk check, compose up
├── .github/
│   └── workflows/
│       └── deploy.yml            # CI/CD — uv test gate then SSH deploy to EC2
├── pytest.ini                    # pytest config (testpaths, addopts, naming conventions)
└── .env.example                  # Environment variable template
```

> `data/transactions.csv`, `models/spending_model.pkl`, `mlflow.db`, `mlruns/`, `.venv/`, and `data/checkpoints.db` are git-ignored (regenerated at runtime).

## Quick Start

### Prerequisites

- Python 3.11+
- uv — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- At least one LLM API key — Groq is free at [console.groq.com](https://console.groq.com)

### Local Setup

```bash
git clone https://github.com/himanshusaini11/wealthwise-agent.git
cd wealthwise-agent
uv sync
cp .env.example .env
# Edit .env — set MODEL_PROVIDER and the matching API key
uv run python scripts/generate_data.py
uv run python scripts/train_pipeline.py
uv run streamlit run app.py
```

Open http://localhost:8501.

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `MODEL_PROVIDER` | Yes | `claude-haiku` · `claude-sonnet` · `gemini` · `groq` |
| `ANTHROPIC_API_KEY` | If `claude-haiku` or `claude-sonnet` | [console.anthropic.com](https://console.anthropic.com) |
| `GROQ_API_KEY` | If `groq` | Free at [console.groq.com](https://console.groq.com) |
| `GOOGLE_API_KEY` | If `gemini` | Google AI Studio |
| `AWS_DEFAULT_REGION` | No | S3 model backup region (default: `us-east-1`) |
| `S3_BUCKET_NAME` | No | S3 bucket for model artifact fallback |
| `R2_THRESHOLD` | No | Training quality gate (default: `-1.0`; use `0.3` in prod) |

pydantic-settings validates that the API key for the active `MODEL_PROVIDER` is non-empty and raises a `ValidationError` at startup if it is missing.

## Testing

```bash
uv run pytest tests/ -v --cov=src --cov-report=term-missing
```

**36 tests · 83% total coverage**

| Module | Coverage | Tested behaviours |
|---|---|---|
| `src/config.py` | 94% | Provider literals, per-provider key validation, missing key detection |
| `src/tools.py` | 92% | `ForecastInput` natural-language parser (13 cases), predict tool, S3 fallback, `_load_model` error |
| `src/logger.py` | 91% | Logger initialisation |
| `src/graph.py` | 74% | `get_llm` (4 providers), `_extract_response` (5 cases + 2 edge cases), `process_query` tuple/error, `_should_continue` (4 routing cases) |

## ML Model

### Feature Engineering

| Feature | Description |
|---|---|
| `Days_Since_Start` | Days elapsed from the first transaction date |
| `day_of_week` | 0 = Monday … 6 = Sunday |
| `month` | Calendar month (1–12) |
| `is_weekend` | 1 if Saturday or Sunday, else 0 |

Fixed recurring categories (`Rent`, `Subscriptions`) are excluded from training — they are not trend-driven and would otherwise dominate the regression.

### Pipeline

```
StandardScaler → LinearRegression
```

- Split: 80 / 20 train / test (`random_state=42`)
- Metrics logged to MLflow (`sqlite:///mlflow.db`, experiment `WealthWise_Forecast`) on every run
- Quality gate: raises `ValueError` if R² < `R2_THRESHOLD`
- Artifact saved to `models/spending_model.pkl` and optionally uploaded to S3

### Retrain

```bash
uv run python scripts/generate_data.py   # regenerate transactions.csv
uv run python scripts/train_pipeline.py  # retrain and log to MLflow
```

## CI/CD

Two-job GitHub Actions workflow triggered on push to `main`:

```
push to main
    │
    ▼
test job
  - actions/checkout@v3
  - astral-sh/setup-uv@v4
  - actions/setup-python@v4 (3.11)
  - uv sync --frozen
  - uv run pytest --cov=src --cov-fail-under=80
    │
    │ must pass
    ▼
deploy job  (needs: test)
  - appleboy/ssh-action → EC2
    - clone repo on first deploy, git pull on subsequent
    - chmod +x deploy.sh && ./deploy.sh
      (down → image prune → disk check → compose up --build -d)
  - on failure → email alert via dawidd6/action-send-mail
```

Deploy is blocked if any test fails or coverage drops below 80%.

## Docker

```bash
docker-compose up --build
```

The image copies the uv binary from `ghcr.io/astral-sh/uv:latest`, runs `uv sync --frozen --no-dev` to install production dependencies, then starts Streamlit on port 8501.

```dockerfile
FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY . .
EXPOSE 8501
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Known Limitations

- **Synthetic training data** — the ML model is trained on generated transactions. Forecast accuracy will improve substantially with real bank export data.
- **Groq tool-call reliability** — `llama-3.3-70b-versatile` has an intermittent HTTP 400 `tool_use_failed` error (~30% rate on complex queries). The retry loop handles most occurrences; Gemini or Claude are more reliable for production use.
- **Free-tier API quotas** — the sidebar model selector lets users switch providers live without restarting the app if a quota is hit.
- **Single-user SQLite checkpoints** — `data/checkpoints.db` is a local file; concurrent multi-user deployments would require a shared checkpoint backend.

## License

MIT
