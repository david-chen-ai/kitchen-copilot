# Kitchen Copilot

An AI assistant for restaurant kitchens: upload an invoice photo, track inventory, and ask questions about stock and kitchen procedures.

**Status:** early development (week 1). The invoice extractor turns a photo into JSON; validation, tests and the assistant come next.

**Stack:** Python 3.12, uv, an LLM vision API. Planned: FastAPI, Postgres + pgvector, React.

**Run locally:** `uv sync`, copy `.env.example` to `.env` and add your API key, then `uv run python extract.py path/to/invoice.jpg`.
