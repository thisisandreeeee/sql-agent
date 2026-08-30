# Dataset fixtures

This directory contains the checked-in SQLite files used by the SQL agent.
Each file is an immutable source fixture. The runtime database is generated in
`var/` by the seed script and is ignored by Git.

## Current dataset

`formula_1.sqlite` is the `formula_1` database from the Spider text-to-SQL
benchmark. It contains 13 relational tables covering races, drivers,
constructors, results, standings, qualifying, pit stops, and lap times.

- Source: [Spider](https://yale-lily.github.io/spider)
- License: CC BY-SA 4.0
- Pinned download revision: [`2abe051bece3132d964271f79dc8589a84e63d06`](https://huggingface.co/datasets/prem-research/spider/tree/2abe051bece3132d964271f79dc8589a84e63d06)
- Download URL:
  `https://huggingface.co/datasets/prem-research/spider/resolve/2abe051bece3132d964271f79dc8589a84e63d06/database/formula_1/formula_1.sqlite`
- File size: 2,940,928 bytes
- SHA-256: `fb6dad97c0a4da22f01bdf817a77fe8f6b6559554661ff0120b40cb81b8c3b68`

The file was downloaded directly into this directory and verified with:

```bash
shasum -a 256 data/formula_1.sqlite
```

## Seed and inspect

Create or refresh the disposable runtime copy:

```bash
uv run python scripts/seed_db.py
```

The result is `var/sql-agent.sqlite`. Run the smoke query with:

```bash
uv run python scripts/smoke_query.py
```

For manual exploration:

```bash
sqlite3 -readonly var/sql-agent.sqlite
```

Re-running the seed replaces the runtime copy with the checked-in fixture.

## Adding another dataset

For each additional database:

1. Prefer a complete SQLite database with multiple related tables, keys, and
   realistic data.
2. Store it as `data/<database_id>.sqlite`, or use a subdirectory when a
   dataset needs multiple files.
3. Pin the exact source URL or release revision; do not download an unpinned
   `latest` artifact.
4. Record the source, license, download URL, file size, and SHA-256 checksum
   in this README.
5. Validate it before committing:

   ```bash
   sqlite3 -readonly data/<database_id>.sqlite \
     'PRAGMA quick_check;'
   sqlite3 -readonly data/<database_id>.sqlite '.tables'
   ```

6. Add or update the seed script so the database can be reproduced locally.
7. Add one small smoke query that exercises the new database's main joins.

Keep databases logically separate. Do not merge unrelated datasets into one
flat set of tables, because identical table names can represent different
concepts. When multiple databases are supported, the agent should select a
`database_id` first and then inspect only that database's schema.
