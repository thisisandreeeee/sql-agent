# SQL Agent

A small, evaluated natural-language-to-SQL experiment built with
[LangGraph](https://langchain-ai.github.io/langgraph/). It answers questions
against a local SQLite database and compares two workflows:

- `graph`: a fixed list-tables → schema → query → answer workflow.
- `react`: a model-directed tool loop using the same database tools.

The goal is to make the full path inspectable: database provenance, generated
SQL, returned rows, final answer, latency, token usage, and evaluation scores.
This is an engineering benchmark, not a production data service.

## How it works

1. The checked-in Spider Formula 1 fixture is validated and copied to the
   disposable runtime database at `var/sql-agent.sqlite`.
2. The agent can list tables, inspect schemas with sample rows, and execute
   SQL.
3. SQLite connections use `mode=ro` plus an authorizer that permits reads only;
   writes, DDL, `ATTACH`/`DETACH`, and connection-changing pragmas are denied.
4. Both workflows bound SQL attempts to five per run. Results are converted to
   a versioned `RunResult` containing the answer, SQL attempts, tool trace, and
   run metrics.

The application uses `deepseek-v4-flash`. The evaluation judges use
`deepseek-v4-pro`, so running evaluations requires a live API key and can be
non-deterministic.

## Example trace

For the question “How many races are in the database?”, an abbreviated trace
looks like this:

```text
User:  How many races are in the database?
Tool:  sql_db_list_tables()
Tool:  sql_db_schema("races")
Tool:  sql_db_query("SELECT COUNT(*) FROM races")
       -> [(997,)]
Agent: There are 997 races in the database.
```

The complete tool trace, SQL attempts, returned rows, and run metrics are
available in the structured result when the CLI is run with `--output`.

## Quick start

Install [uv](https://docs.astral.sh/uv/) if needed, then install dependencies:

```bash
uv sync
cp .env.example .env
# Set DEEPSEEK_API_KEY in .env
```

Create the runtime database and verify it:

```bash
uv run python data/seed_db.py
uv run python data/smoke_query.py
```

Ask a question. The CLI defaults to ReAct; select the fixed graph explicitly
when comparing workflows:

```bash
uv run python -m sql_agent.main "Who is the fastest driver?"
uv run python -m sql_agent.main --agent-type graph "Who is the fastest driver?"
uv run python -m sql_agent.main "How many races are in the database?" --output runs/manual.json
```

For manual database exploration:

```bash
sqlite3 -readonly var/sql-agent.sqlite
```

## Data provenance

The fixture is the official Spider `formula_1` SQLite database. Spider covers
13 tables for races, drivers, constructors, results, standings, qualifying,
pit stops, and lap times. The data ends in 2018 and is not live Formula 1 data.

Source: [Spider project page](https://yale-lily.github.io/spider), which links
to the official [`spider_data.zip` archive](https://drive.google.com/file/d/1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J/view?usp=sharing).
Spider is released under CC BY-SA 4.0.

- SQLite source: `data/database/formula_1/formula_1.sqlite`
- SQLite SHA-256: `fb6dad97c0a4da22f01bdf817a77fe8f6b6559554661ff0120b40cb81b8c3b68`
- Archive SHA-256: `00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b`

Verify the checked-in fixture with:

```bash
shasum -a 256 data/database/formula_1/formula_1.sqlite
```

To add a dataset, preserve the `data/database/<database_id>/` layout, record
its source, license, size, and SHA-256, validate it with `PRAGMA quick_check`,
and add a representative smoke query. Keep source fixtures immutable and use
`var/` for runtime copies.

## Evaluation

The evaluation cases in `evals/cases.yaml` cover database facts, joins, aggregations,
empty-result questions, metadata, out-of-scope requests, and safety behavior.
Run either workflow with:

```bash
uv run python -m evals.runner --agent-type graph
uv run python -m evals.runner --agent-type react
```

Each run writes an ignored, timestamped JSON artifact under `runs/`. For
historical context, a recorded 48-case snapshot from 2026-09-04, before both
the current 62-case suite and the graph loop guard in this review, was:

| Workflow | Passed | Mean latency | Correctness | Groundedness |
| -------- | -----: | -----------: | ----------: | -----------: |
| Graph    |  28/48 |      9.926 s |       0.826 |        0.939 |
| ReAct    |  28/48 |      7.877 s |       0.792 |        0.965 |

These results are directional because both generation and judging use live
models. The main learning is that valid, bounded SQL execution is not enough:
the remaining failures mostly involve interpreting large or empty result sets
and producing an exact answer. The graph was more accurate in this snapshot;
ReAct was faster and slightly better grounded. Re-run the benchmark after code
changes before treating the numbers as a baseline.

## Tests and project status

Run the local checks with:

```bash
uv run pytest -q
uv run python data/seed_db.py
uv run python data/smoke_query.py
```
