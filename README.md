# SQL Agent

A minimal SQL agent built with [LangGraph](https://langchain-ai.github.io/langgraph/).

## Setup

Install [`uv`](https://docs.astral.sh/uv/) if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install the project dependencies:

```bash
uv sync
```

Copy `.env.example` to `.env` and set `DEEPSEEK_API_KEY`:

```bash
cp .env.example .env
```

To run the agent:

```bash
uv run python -m sql_agent.main "What tables are in the database?"
```

## Database

The checked-in fixture is Spider's `formula_1` SQLite database. It contains 13
tables covering races, drivers, constructors, results, standings, qualifying,
pit stops, and lap times. Fixtures in `data/` are immutable; the seed script
creates the disposable runtime copy at `var/sql-agent.sqlite`.

Create or refresh the runtime database and run the smoke query:

```bash
uv run python data/seed_db.py
uv run python data/smoke_query.py
```

For manual exploration:

```bash
sqlite3 -readonly var/sql-agent.sqlite
```

The current fixture was downloaded from [Spider](https://yale-lily.github.io/spider),
which is released under CC BY-SA 4.0. It is pinned to this
[dataset mirror revision](https://huggingface.co/datasets/prem-research/spider/tree/2abe051bece3132d964271f79dc8589a84e63d06).

- Download URL: `https://huggingface.co/datasets/prem-research/spider/resolve/2abe051bece3132d964271f79dc8589a84e63d06/database/formula_1/formula_1.sqlite`
- File size: 2,940,928 bytes
- SHA-256: `fb6dad97c0a4da22f01bdf817a77fe8f6b6559554661ff0120b40cb81b8c3b68`

Verify the fixture with:

```bash
shasum -a 256 data/formula_1.sqlite
```

### Adding a dataset

For each additional database:

1. Store it as `data/<database_id>.sqlite`, or in a dedicated subdirectory if
   it needs multiple files.
2. Pin and record the source URL or release revision, license, file size, and
   SHA-256 checksum.
3. Validate it before committing:

   ```bash
   sqlite3 -readonly data/<database_id>.sqlite 'PRAGMA quick_check;'
   sqlite3 -readonly data/<database_id>.sqlite '.tables'
   ```

4. Update the seed script and add a smoke query for its main joins.

Keep unrelated databases separate. When multiple databases are supported, the
agent should select a `database_id` before inspecting that database's schema.

## Tests

Run the unit tests:

```bash
uv run pytest -q
```

Evaluation cases call the live model and are separate from the unit-test run:

```bash
uv run python -m evals.runner
```

## Backlog

- Improve evaluation harness: add sql validity, run metrics, and other P0 evaluators.
- Create comprehensive benchmark dataset.
- Improve workflow harness: e.g. schema context, deterministic SQL validation, and structured query planning.
- Compare a deterministic workflow with a ReAct-style loop.
