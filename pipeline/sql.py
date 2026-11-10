"""SQL literals for filenames and globs used by DuckDB statements.

Keep the surrounding SQL separate from external path text. DuckDB accepts
parameters for many statements, but view definitions and some COPY clauses
need literals; SQL escapes an apostrophe by doubling it.
"""


def sql_literal(value) -> str:
    return "'" + str(value).replace("'", "''") + "'"
