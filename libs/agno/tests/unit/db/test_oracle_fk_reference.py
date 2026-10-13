"""Foreign-key reference resolution tests; no database connection is needed."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agno.db.oracle import AsyncOracleDb, OracleDb


@pytest.mark.parametrize("adapter", [OracleDb, AsyncOracleDb], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("schema", "reference", "expected", "expected_table"),
    [
        (None, "agno_sessions.session_id", "agno_sessions.session_id", "agno_sessions"),
        (
            "AGNO_TEST_USER",
            "agno_sessions.session_id",
            "AGNO_TEST_USER.agno_sessions.session_id",
            "agno_sessions",
        ),
        (
            "AGNO_TEST_USER",
            "AGNO_TEST_USER.agno_sessions.session_id",
            "AGNO_TEST_USER.agno_sessions.session_id",
            "agno_sessions",
        ),
        (
            None,
            "OTHER_OWNER.agno_sessions.session_id",
            "OTHER_OWNER.agno_sessions.session_id",
            "agno_sessions",
        ),
        (
            "AGNO_TEST_USER",
            "OTHER_OWNER.agno_sessions.session_id",
            "OTHER_OWNER.agno_sessions.session_id",
            "agno_sessions",
        ),
        (None, "sessions.session_id", "custom_sessions.session_id", "sessions"),
        (
            "AGNO_TEST_USER",
            "sessions.session_id",
            "AGNO_TEST_USER.custom_sessions.session_id",
            "sessions",
        ),
        (
            "AGNO_TEST_USER",
            "AGNO_TEST_USER.sessions.session_id",
            "AGNO_TEST_USER.custom_sessions.session_id",
            "sessions",
        ),
        (
            "AGNO_TEST_USER",
            "OTHER_OWNER.sessions.session_id",
            "OTHER_OWNER.custom_sessions.session_id",
            "sessions",
        ),
        ("AGNO_TEST_USER", "session_id", "session_id", None),
    ],
)
def test_fk_reference_qualifies_schema_once(adapter, schema, reference, expected, expected_table):
    def resolve_table(name):
        return {"sessions": "custom_sessions"}.get(name, name)

    table_resolver = Mock(side_effect=resolve_table)
    context = SimpleNamespace(
        db_schema=schema,
        _resolve_table_name=table_resolver,
    )

    result = adapter._resolve_fk_reference(context, reference)

    assert result == expected
    if expected_table is None:
        table_resolver.assert_not_called()
    else:
        table_resolver.assert_called_once_with(expected_table)
