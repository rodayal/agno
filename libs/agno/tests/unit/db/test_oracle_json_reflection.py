"""Tests for restoring JSON converters; no live database is used."""

from unittest.mock import Mock

import pytest
from sqlalchemy import JSON, Column, MetaData, Table, Text
from sqlalchemy.dialects.oracle import CLOB, dialect
from sqlalchemy.engine import Connection
from sqlalchemy.types import NullType

from agno.db.oracle.schemas import OracleClobJSON, OracleNativeJSON
from agno.db.oracle.utils import restore_json_column_types


def make_table(schema=None, payload_type=None):
    if payload_type is None:
        payload_type = CLOB()

    return Table(
        "agno_json_probe",
        MetaData(),
        Column("payload", payload_type),
        Column("notes", CLOB()),
        schema=schema,
    )


def make_connection(rows):
    connection = Mock(spec=Connection)
    connection.dialect = dialect()
    connection.execute.return_value.all.return_value = rows
    return connection


@pytest.mark.parametrize("declared_type", [OracleClobJSON, OracleNativeJSON])
@pytest.mark.parametrize("storage", ["CLOB", "JSON"])
@pytest.mark.parametrize("schema", [None, "agno_test_user"])
def test_restores_converter_from_physical_storage(declared_type, storage, schema):
    reflected_type = CLOB() if storage == "CLOB" else JSON()
    table = make_table(schema, reflected_type)
    original_notes_type = table.c.notes.type
    connection = make_connection([("PAYLOAD", storage), ("NOTES", "CLOB")])

    definition = {
        "payload": {"type": declared_type},
        "notes": {"type": Text},
    }

    result = restore_json_column_types(connection, table, definition)

    expected_type = OracleClobJSON if storage == "CLOB" else OracleNativeJSON
    assert result is table
    assert isinstance(table.c.payload.type, expected_type)
    assert table.c.notes.type is original_notes_type

    connection.execute.assert_called_once()
    query, params = connection.execute.call_args.args
    sql = str(query).lower()

    if schema is None:
        assert "from user_tab_columns" in sql
        assert params == {"table_name": "AGNO_JSON_PROBE"}
    else:
        assert "from all_tab_columns" in sql
        assert "owner = :owner" in sql
        assert params == {
            "table_name": "AGNO_JSON_PROBE",
            "owner": "AGNO_TEST_USER",
        }


@pytest.mark.parametrize("storage", ["CLOB", "JSON"])
def test_unrecognized_reflection_uses_database_metadata(storage):
    table = make_table(payload_type=NullType())
    connection = make_connection([("PAYLOAD", storage)])

    restore_json_column_types(
        connection,
        table,
        {"payload": {"type": OracleNativeJSON}},
    )

    expected_type = OracleClobJSON if storage == "CLOB" else OracleNativeJSON
    assert isinstance(table.c.payload.type, expected_type)


@pytest.mark.parametrize("storage", [None, "VARCHAR2"], ids=["missing", "unexpected"])
def test_missing_or_unexpected_storage_is_rejected(storage):
    table = make_table()
    rows = [] if storage is None else [("PAYLOAD", storage)]
    connection = make_connection(rows)

    with pytest.raises(ValueError, match="Unsupported physical type"):
        restore_json_column_types(
            connection,
            table,
            {"payload": {"type": OracleNativeJSON}},
        )


@pytest.mark.parametrize(
    "definition",
    [{}, {"notes": {"type": Text}}],
    ids=["no-definition", "ordinary-text"],
)
def test_no_declared_json_leaves_table_unchanged(definition):
    table = make_table()
    original_payload_type = table.c.payload.type
    original_notes_type = table.c.notes.type
    connection = make_connection([])

    result = restore_json_column_types(connection, table, definition)

    assert result is table
    assert table.c.payload.type is original_payload_type
    assert table.c.notes.type is original_notes_type
    connection.execute.assert_not_called()
