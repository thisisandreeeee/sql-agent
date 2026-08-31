# SQL Agent

Minimal project scaffold for a SQL agent built with [LangGraph](https://langchain-ai.github.io/langgraph/).

## Setup

Install `uv` if it is not already installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create the environment and install dependencies:

```bash
uv sync
```

## DeepSeek configuration

Copy the environment template and set your API key:

```bash
cp .env.example .env
```

Set `DEEPSEEK_API_KEY` in `.env`. The application initializes
`deepseek-v4-pro` with thinking disabled. To send a query with the registered
tools available, run:

```bash
uv run python -m sql_agent.main "What tables are in the database?"
```

## Local database

The starter database is Spider's `formula_1` SQLite database. The checked-in
fixture is 2.9 MB, so it is stored directly in Git rather than Git LFS.

Seed the runtime copy and run the smoke query with:

```bash
uv run python scripts/seed_db.py
uv run python scripts/smoke_query.py
```

The source fixture is [Spider](https://yale-lily.github.io/spider), released
under CC BY-SA 4.0. It was downloaded from the pinned
[dataset mirror revision](https://huggingface.co/datasets/prem-research/spider/tree/2abe051bece3132d964271f79dc8589a84e63d06)
and has SHA-256:

```text
fb6dad97c0a4da22f01bdf817a77fe8f6b6559554661ff0120b40cb81b8c3b68
```

## Unit tests

Run pytest:

```bash
uv run pytest -q
```

## Backlog

- Build openevals harness: trajectory, outcome, SQL validity, and run metrics
- Improve workflow: (1) better schema context, (2) deterministic SQL validation, (3) structured query planning
- Compare deterministic workflow vs. ReAct style loop
