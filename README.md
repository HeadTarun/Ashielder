# AIshield 

ADK ReAct-style agent with:

- Gemini as the primary LLM
- Groq via LiteLLM as fallback
- Local tools for DistilBERT classification, URL spam detection, and embedding analysis

## Run

```bash
uv run python main.py "analyze this text"
uv run python main.py "https://example.com"
uv run python main.py --threat "https://example.com"
uv run python main.py --decision "https://example.com"
```

## Environment

The app always loads `.env` with `python-dotenv` before creating the ADK agents or local model tools.

```bash
cp .env.example .env
```

Fill in provider keys only for the models you plan to use. The local ONNX model paths work with the folders in this repo by default.

## Agent entrypoints

- `main.root_agent` uses Gemini by default
- `tri_model_agent.agent.root_agent` is the standard ADK package entrypoint
- `tri_model_agent_demo.agent.root_agent` is the ADK server-safe local demo app for hackathons
- `main.fallback_agent` uses Groq via LiteLLM when LiteLLM support is installed; otherwise it is `None`

## Local ML tools

- `analyze_threat` runs the hackathon-ready verdict pipeline and writes to ADK session state
- `decide_url_threat` runs the grounded URL intelligence decision layer with local ML, RDAP, Google Safe Browsing, and curated corpus citations
- `route_request` classifies input and dispatches to the best local model
- `classify_request` uses `distilbert_m2`
- `detect_url_spam` uses `lighturlnet`
- `analyze_embedding` uses `embed`
- `gmail_fetch_tool` fetches the latest read-only Gmail messages after explicit Google OAuth consent
- `analyze_latest_gmail` runs the mailbox security diagnosis report using deterministic scoring
- `remember_user_fact` stores user-scoped facts through ADK state
- `search_session_memory` searches recent analysis history and remembered facts

## Session and memory

- `SESSION_SERVICE` uses ADK `InMemorySessionService`
- `MEMORY_SERVICE` uses ADK `InMemoryMemoryService`
- `create_runner()` returns an ADK `Runner` with session and memory services attached
- `LoadMemoryTool()` is registered with the agent for ADK memory retrieval when the runner provides memory

## Hackathon demo angle

Show the agent as an explainable security triage copilot:

1. Ask it to analyze a suspicious URL.
2. Ask why it reached the verdict.
3. Ask it to remember a team preference such as "always escalate high risk links".
4. Analyze another URL and ask what it remembers from the session.

Strong next additions:

- Streamlit dashboard with verdict, model evidence, and session timeline.
- Small eval set for safe URL, spam URL, and semantic-analysis prompts.
- One-click "export incident report" artifact.

When the hosted Gemini API is overloaded, use the `tri_model_agent_demo` app in `adk web`. It is an ADK `BaseAgent` that runs the local ML pipeline and ADK session state without a remote model call.

## Optional environment variables

- `GOOGLE_API_KEY`
- `GOOGLE_GENAI_USE_VERTEXAI`
- `GOOGLE_CLOUD_PROJECT`
- `GOOGLE_CLOUD_LOCATION`
- `GEMINI_MODEL`
- `GROQ_MODEL`
- `GROQ_API_KEY`
- `DISTILBERT_MODEL_PATH`
- `LIGHTURLNET_MODEL_PATH`
- `EMBEDDING_MODEL_PATH`
- `DISTILBERT_URL_LABELS`
- `DISTILBERT_ANALYSIS_LABELS`
- `DISTILBERT_MIN_CONFIDENCE`
- `SAFE_BROWSING_API_KEY`
- `RDAP_TIMEOUT_SECONDS`
- `INTEL_CORPUS_PATH`
- `GOOGLE_OAUTH_CLIENT_ID`
- `GOOGLE_OAUTH_CLIENT_SECRET`
- `GOOGLE_OAUTH_REDIRECT_URI`
- `GMAIL_TOKEN_DB_PATH`
- `GMAIL_TOKEN_ENCRYPTION_KEY`
- `GMAIL_FETCH_LIMIT`

## Grounded URL decisions

`decide_url_threat` is the strict decision tool for suspicious links. It extracts the URL/domain, runs local model evidence, enriches with RDAP registration data, checks Google Safe Browsing when `SAFE_BROWSING_API_KEY` is configured, retrieves related guidance from `knowledge/intel_corpus.json`, and returns `verdict`, `risk_score`, `confidence`, `evidence`, `citations`, `recommended_actions`, and `unknowns`.

Missing RDAP or Safe Browsing data is reported in `unknowns`; the agent must not treat a missing or empty source result as proof that a URL is safe.

## Gmail mailbox diagnosis

`analyze_latest_gmail` uses ADK Web authenticated tool flow with the read-only Gmail scope `https://www.googleapis.com/auth/gmail.readonly`. It fetches up to 10 recent emails, classifies email text with the existing DistilBERT tool, checks extracted URLs with the existing URL decision tool, generates embeddings with the existing embedding model, and applies deterministic sender/risk rules.

The mailbox report includes `overall_status`, `security_score`, `risk_level`, per-email evidence, recommendations, `mailbox_diagnosis`, and a fixed-format `MAILGUARD AI SECURITY REPORT`. Scores and findings are derived from model/tool output and deterministic rules; the LLM should summarize them, not invent them.
