# SQL Agent

A minimal SQL agent built with [LangGraph](https://langchain-ai.github.io/langgraph/).
It includes a deterministic graph agent (`graph`) and a ReAct agent (`react`)
that use the same database tools.

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
uv run python -m sql_agent.main "Who is the fastest driver?"
```

## Database

The checked-in fixture is the official Spider `formula_1` SQLite database. It
contains 13 tables covering races, drivers, constructors, results, standings,
qualifying, pit stops, and lap times. Fixtures in `data/` are immutable; the
seed script creates the disposable runtime copy at `var/sql-agent.sqlite`.

Create or refresh the runtime database and run the smoke query:

```bash
uv run python data/seed_db.py
uv run python data/smoke_query.py
```

For manual exploration:

```bash
sqlite3 -readonly var/sql-agent.sqlite
```

The current fixture was downloaded from the [official Spider project
page](https://yale-lily.github.io/spider), which links to the official
[`spider_data.zip` Google Drive archive](https://drive.google.com/file/d/1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J/view?usp=sharing).
Spider is released under the CC BY-SA 4.0 license. The archive is extracted
using its `database/<database_id>/<database_id>.sqlite` layout; this project
uses `database/formula_1/formula_1.sqlite`.

- Archive name: `spider_data.zip`
- Archive size: 205,800,266 bytes
- Archive SHA-256: `00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b`
- SQLite source: `data/database/formula_1/formula_1.sqlite`
- SQLite file size: 2,940,928 bytes
- SQLite SHA-256: `fb6dad97c0a4da22f01bdf817a77fe8f6b6559554661ff0120b40cb81b8c3b68`

Verify the fixture with:

```bash
shasum -a 256 data/database/formula_1/formula_1.sqlite
```

### Adding a dataset

For each additional database:

1. Store it as `data/database/<database_id>/<database_id>.sqlite`, preserving
   the Spider database layout.
2. Pin and record the source URL or release revision, license, file size, and
   SHA-256 checksum.
3. Validate it before committing:

   ```bash
   sqlite3 -readonly data/database/<database_id>/<database_id>.sqlite 'PRAGMA quick_check;'
   sqlite3 -readonly data/database/<database_id>/<database_id>.sqlite '.tables'
   ```

4. Update the seed script and add a smoke query for its main joins.

Keep unrelated databases separate. When multiple databases are supported, the
agent should select a `database_id` before inspecting that database's schema.

## Tests

Run the unit tests:

```bash
uv run pytest -q
```

Evaluation cases call the live model and are separate from the unit-test run.
The deterministic graph is the default:

```bash
uv run python -m evals.runner --agent-type graph
```

Run the ReAct comparison with:

```bash
uv run python -m evals.runner --agent-type react
```

Each evaluation artifact is written with the selected agent type as its filename
prefix and records `agent_type` at the top level.

## Backlog

- Stop graph recursion and terminate cleanly on empty tables.
- Include evaluation cases that are prone to hallucination.
- Repair the contradictory and nondeterministic evaluation cases.
- Create comprehensive benchmark dataset.
- Improve workflow harness: e.g. schema context, deterministic SQL validation, and structured query planning.
