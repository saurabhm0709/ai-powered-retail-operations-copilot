# Project rules (all AI assistants and all team members)

Hackathon project: **Retail Operations Copilot** for Punjab Sports Mart (HCLTech AI Engineer hackathon).
Every team member must be able to explain every line to a panel and modify it live. Readability beats
cleverness. Budget for paid APIs is very small (about $5 total).

**`ARCHITECTURE.md` is the build spec.** Follow its module names, data contracts, priority rules and build
order (§10). If the spec is ambiguous or contradicts the data, say so and propose a fix — do not silently
diverge. If a rule in this file conflicts with ARCHITECTURE.md, this file wins; point out the conflict.

## Core design rules (non-negotiable)
1. **LLM interprets and writes; code decides.** Priority, escalation, thresholds, PII masking, citations and
   approval boundaries are Python (`rules.py`, `controller.py`, `safety.py`), never the LLM.
2. **Mask before the LLM.** Every user text passes `safety.check_input` before any LLM call, cache write or
   log write. Never log or cache unmasked text.
3. **No hard-coding of test cases.** Never match on test ids or public test prompt text. Fix failures through
   general rules, prompts or tools. Few-shot examples must not copy public test prompts.
4. **Citations** use `file.md#section-slug` and must come from the allowed set built by the controller.
5. **Numbers** in answers come only from Data Agent `facts`. The LLM never does arithmetic on data.
6. The copilot never claims to approve refunds, submit claims, quarantine stock or declare equipment safe.
7. **Anchor time** for "today / last 24 hours / last 30 days" is `config.ANCHOR` = `2026-10-10 12:00:00` IST.
   Rows timestamped after it are excluded.
8. Provided inputs are read-only: `knowledge_base/`, `data/`, `evaluation/`.

## Project layout
- Pipeline stages, one file each, in `copilot/` (see ARCHITECTURE.md §3):
  `safety.py`, `triage.py`, `rag.py`, `data_agent.py`, `controller.py`, `responder.py`, `workflow.py`,
  `orchestrator.py`; shared: `config.py`, `schemas.py`, `api_utils.py`, `llm.py`, `data_store.py`, `rules.py`.
- Entry points: `scripts/build_index.py` (ingest), `scripts/run_eval.py`, `copilot/cli.py`, `app.py`.
- `config.py` holds every tunable parameter (thresholds, anchor time, top_k, distance cutoff, model names,
  temperature, max_tokens, MAX_CALLS, paths, flags). Never hardcode these inside functions.
- `api_utils.py` holds the shared disk cache and call-limit helpers (see API rules).
- The files listed in ARCHITECTURE.md §3 are pre-approved. Do not create any other files or folders without asking.

## Code style
- Plain Python functions. No classes unless asked. **Exception:** Pydantic models in `schemas.py` (they are
  the data contracts between agents and the LLM structured outputs).
- Each function does one thing and is under ~25 lines.
- Descriptive names (`top_k_chunks`, not `res`).
- Prefer explicit steps over one-liners, chained calls, or comprehensions that do too much.
- Add type hints to function signatures.
- Comments: one short line above each function (what goes in, what comes out). Comment WHY for non-obvious
  choices, never WHAT. A rule or threshold's comment names its policy source
  (e.g. `# 04_vendor_quality_policy.md#short-shipments`).

## Dependencies
- Allowed stack: `streamlit`, `pandas`, `openai`, `chromadb`, `pydantic>=2`, `python-dotenv`, `pytest`.
  Ask before adding anything else.
- Avoid high-level "magic" wrappers (LangChain chains, RetrievalQA, agent executors) when a few explicit lines
  show what is happening.
- Add a new library to `requirements.txt` on its own line, in a separate small commit, to avoid merge conflicts.

## Error handling
- Handle only errors that realistically happen: API failure or invalid LLM JSON (retry, then deterministic
  fallback), empty retrieval (say policy cannot be confirmed), missing file, missing store/SKU in the data
  (ask a follow-up question). No defensive layers for hypothetical cases.

## API cost rules
- Never call a paid API twice for the same input. Cache results to disk in `.cache/` (gitignored).
- Build the cache key from everything that affects the result: masked input text hash, model name,
  temperature, max_tokens, prompt version string, and any other parameter. Changing any of them must
  invalidate the cache.
- Embeddings are free and local: ChromaDB's default embedding function (`all-MiniLM-L6-v2`, runs via
  onnxruntime, no torch). `scripts/build_index.py` embeds and builds the index once into `.chroma/`.
  `rag.py` only loads the saved index. Never re-embed documents at query time.
- Paid API = OpenAI chat only (Triage + Final Response, 2 calls per request). Use `gpt-4o-mini` for
  development and demo unless the team decides otherwise.
- Every LLM call sets `max_tokens`. Retries are limited to 2 with backoff, then the deterministic fallback.
  No unbounded retry loops and no API calls inside unbounded loops.
- Wrap every paid call in the shared call counter from `api_utils.py`. It raises an error when `MAX_CALLS`
  (set in `config.py`) is exceeded.
- `MOCK_LLM=1` runs the full pipeline with zero network calls. Tests use mock mode or saved responses, never
  real API calls.
- Before running anything that calls a paid API, state how many calls it will make and roughly what it will
  cost (≈ $0.0005 per request with gpt-4o-mini). Ask first if it could exceed ~50 uncached calls.
  A full public eval run is 11 requests ≈ 22 calls; cached re-runs cost nothing.

## Secrets
- API keys live only in `.env` (listed in `.gitignore`). Commit `.env.example` with empty values:
  `OPENAI_API_KEY=`, `OPENAI_MODEL=gpt-4o-mini`, `MOCK_LLM=0`, `PRIORITY_MODE=rules`.
- Each member uses their own key with their own spending cap where possible. Never paste a key in code,
  chat, commits or logs.
- Never read, print, log or commit `.env` or any key.

## Git workflow
- Never commit to `main` directly. Work on `feature/<name>` branches and merge through pull requests with one
  reviewer (any teammate; keep reviews quick — this is a time-boxed build).
- Commit small and often. Messages say what changed ("Add short-shipment rule to rules.py"), never "fix" or
  "update". Every member commits their own component — judges look at the history.
- Never use `git push --force` or `git reset --hard` unless the user asks.
- Do not commit `.env`, `.cache/`, `.chroma/`, large files, models or notebook outputs. Do commit small
  sample outputs needed for the demo (`outputs/predictions.csv`, `outputs/metrics.md`).

## Testing
- Each module gets a small pytest file (`tests/test_safety.py`, `test_rules.py`, `test_data_agent.py`,
  `test_pipeline_mock.py`). Run `pytest -q` before saying a step is done.
- After any pipeline change, run `python scripts/run_eval.py` (mock mode first) and report the metrics table.

## Working with the user
- Make small changes. After each change, say in 3-4 lines what changed and why.
- Do not refactor code the user did not ask you to touch.
- If there is a simpler way than what was asked, say so before writing code.
- Do not generate code the user cannot explain. If a solution needs an advanced concept, explain it briefly
  or propose a simpler alternative.

## Commands
```
pip install -r requirements.txt
python scripts/build_index.py            # build ChromaDB index once (--rebuild to force)
MOCK_LLM=1 python -m copilot.cli "All registers in ST003 cannot process cards"
pytest -q
python scripts/run_eval.py [--extra] [--priority-mode llm]
streamlit run app.py
```
