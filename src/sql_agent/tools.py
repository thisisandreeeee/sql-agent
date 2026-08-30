"""LLM-facing tools and the tool registry."""

from langchain_core.tools import tool

from . import db


@tool
def sql_db_list_tables() -> list[str]:
    """Input is an empty string, output is a comma-separated list of tables in the database."""
    return db.list_tables()


@tool
def sql_db_schema(table_names: str) -> str:
    """Input is a comma-separated list of tables; output is their schema and sample rows.

    Call sql_db_list_tables first to confirm that the tables exist.
    Example input: table1, table2, table3
    """
    return db.schema(table_names)


@tool
def sql_db_query(query: str) -> str:
    """Execute a detailed, correct read-only SQL query and return its result.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    If you encounter an issue with Unknown column 'xxxx' in 'field list', use sql_db_schema to query the correct table fields.
    """
    return db.query(query)


TOOLS = [sql_db_list_tables, sql_db_schema, sql_db_query]
