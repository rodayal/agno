"""Oracle regression tests grouped by behavior; no cases are discarded."""

import importlib
import inspect as python_inspect
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import agno.db.oracle.async_oracle as async_module
import agno.db.oracle.oracle as sync_module
import pytest
from agno.db.migrations.versions import v2_3_0, v2_5_0, v2_5_6, v3_0_0
from agno.db.oracle import AsyncOracleDb, OracleDb, utils
from agno.db.oracle import utils as storage
from agno.db.oracle.schemas import OracleClobJSON, OracleNativeJSON
from agno.db.oracle.utils import (
    DEFAULT_MERGE_RETRY_ATTEMPTS,
    restore_json_column_types,
    to_db_user_id,
    truncate_identifier,
)
from agno.knowledge.document import Document
from agno.session import AgentSession
from agno.vectordb.oracle import oracle as vector
from sqlalchemy import JSON, BigInteger, Column, MetaData, String, Table, Text, select
from sqlalchemy.dialects.oracle import CLOB, dialect
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError
from sqlalchemy.sql.dml import Insert
from sqlalchemy.sql.elements import TextClause
from sqlalchemy.sql.selectable import Select
from sqlalchemy.types import NullType
from tests.integration.db.oracle import test_differential_async_oracle_postgres as differential_async_module
from tests.integration.db.oracle import test_differential_oracle_postgres as differential_sync_module

# Migration dispatch

pytest.importorskip("sqlalchemy")


def _make_db(class_name: str):
    """A fake db instance whose type().__name__ is class_name, matching the
    dispatch pattern test_migration_v2_5_0_dispatch.py already established.
    """
    mock = MagicMock()

    class FakeDb:
        def __getattr__(self, name):
            return getattr(mock, name)

    FakeDb.__name__ = class_name
    FakeDb.__qualname__ = class_name
    return FakeDb()


NOT_IMPLEMENTED_PHRASES = ("not implemented", "nosql/document store")


def _assert_no_fallback_phrase(caplog):
    lowered = caplog.text.lower()
    for phrase in NOT_IMPLEMENTED_PHRASES:
        assert phrase not in lowered, (
            f"Oracle dispatch fell through to the generic fallback ({phrase!r}): {caplog.text}"
        )


class TestV2_3_0OracleDispatch:
    def test_up_is_explicit_no_op(self, caplog):
        with caplog.at_level("INFO"):
            assert v2_3_0.up(_make_db("OracleDb"), "memories", "agno_memories") is False
        assert "already present since table creation" in caplog.text
        _assert_no_fallback_phrase(caplog)

    @pytest.mark.asyncio
    async def test_async_up_is_explicit_no_op(self, caplog):
        with caplog.at_level("INFO"):
            assert await v2_3_0.async_up(_make_db("AsyncOracleDb"), "memories", "agno_memories") is False
        assert "already present since table creation" in caplog.text
        _assert_no_fallback_phrase(caplog)

    def test_down_is_explicit_no_op(self, caplog):
        with caplog.at_level("INFO"):
            assert v2_3_0.down(_make_db("OracleDb"), "memories", "agno_memories") is False
        assert "no pre-v2.3.0 oracle schema exists" in caplog.text.lower()
        _assert_no_fallback_phrase(caplog)

    @pytest.mark.asyncio
    async def test_async_down_is_explicit_no_op(self, caplog):
        with caplog.at_level("INFO"):
            assert await v2_3_0.async_down(_make_db("AsyncOracleDb"), "memories", "agno_memories") is False
        assert "no pre-v2.3.0 oracle schema exists" in caplog.text.lower()
        _assert_no_fallback_phrase(caplog)


class TestV2_5_0OracleDispatch:
    def test_up_is_explicit_no_op(self):
        # v2.5.0's fallback message is identical for the SqliteDb branch and
        # the generic else, so this pins behavior (False, no exception) --
        # the module-level comment records the decision.
        assert v2_5_0.up(_make_db("OracleDb"), "sessions", "agno_sessions") is False

    @pytest.mark.asyncio
    async def test_async_up_is_explicit_no_op(self):
        assert await v2_5_0.async_up(_make_db("AsyncOracleDb"), "sessions", "agno_sessions") is False

    def test_down_is_explicit_no_op(self):
        assert v2_5_0.down(_make_db("OracleDb"), "sessions", "agno_sessions") is False

    @pytest.mark.asyncio
    async def test_async_down_is_explicit_no_op(self):
        assert await v2_5_0.async_down(_make_db("AsyncOracleDb"), "sessions", "agno_sessions") is False


class TestV2_5_6OracleDispatch:
    def test_up_is_explicit_no_op(self):
        assert v2_5_6.up(_make_db("OracleDb"), "approvals", "agno_approvals") is False

    @pytest.mark.asyncio
    async def test_async_up_is_explicit_no_op(self):
        assert await v2_5_6.async_up(_make_db("AsyncOracleDb"), "approvals", "agno_approvals") is False

    def test_down_is_explicit_no_op(self):
        assert v2_5_6.down(_make_db("OracleDb"), "approvals", "agno_approvals") is False

    @pytest.mark.asyncio
    async def test_async_down_is_explicit_no_op(self):
        assert await v2_5_6.async_down(_make_db("AsyncOracleDb"), "approvals", "agno_approvals") is False


class TestV3_0_0OracleDispatch:
    def test_up_is_explicit_no_op(self, caplog):
        with caplog.at_level("INFO"):
            assert v3_0_0.up(_make_db("OracleDb"), "sessions", "agno_sessions") is False
        assert "already present since table creation" in caplog.text
        _assert_no_fallback_phrase(caplog)

    @pytest.mark.asyncio
    async def test_async_up_is_explicit_no_op(self, caplog):
        with caplog.at_level("INFO"):
            assert await v3_0_0.async_up(_make_db("AsyncOracleDb"), "sessions", "agno_sessions") is False
        assert "already present since table creation" in caplog.text
        _assert_no_fallback_phrase(caplog)

    def test_down_is_explicit_no_op(self, caplog):
        with caplog.at_level("INFO"):
            assert v3_0_0.down(_make_db("OracleDb"), "sessions", "agno_sessions") is False
        assert "no pre-v3.0.0 oracle schema exists" in caplog.text.lower()
        _assert_no_fallback_phrase(caplog)

    @pytest.mark.asyncio
    async def test_async_down_is_explicit_no_op(self, caplog):
        with caplog.at_level("INFO"):
            assert await v3_0_0.async_down(_make_db("AsyncOracleDb"), "sessions", "agno_sessions") is False
        assert "no pre-v3.0.0 oracle schema exists" in caplog.text.lower()
        _assert_no_fallback_phrase(caplog)

    def test_learnings_content_rekey_takes_real_path_not_oracle_branch(self, monkeypatch):
        """The learnings rekey dispatches on table_type, before the db_type
        check -- it must reach _rekey_learnings even for OracleDb, never the
        Oracle no-op branch (which would silently skip a real content move).
        """
        called = {}

        def _fake_rekey(db, table_name):
            called["hit"] = True
            return True

        monkeypatch.setattr(v3_0_0, "_rekey_learnings", _fake_rekey)
        result = v3_0_0.up(_make_db("OracleDb"), "learnings", "agno_learnings")
        assert called.get("hit") is True
        assert result is True

    @pytest.mark.asyncio
    async def test_async_learnings_content_rekey_takes_real_path(self, monkeypatch):
        called = {}

        async def _fake_arekey(db, table_name):
            called["hit"] = True
            return True

        monkeypatch.setattr(v3_0_0, "_arekey_learnings", _fake_arekey)
        result = await v3_0_0.async_up(_make_db("AsyncOracleDb"), "learnings", "agno_learnings")
        assert called.get("hit") is True
        assert result is True


# Foreign-key reference resolution


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


# Reflected JSON converters


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


# Generated identifier names


@pytest.mark.parametrize("name", ["agno_sessions", "x" * 128, "é" * 64])
def test_short_identifiers_are_unchanged(name):
    assert truncate_identifier(name) == name


@pytest.mark.parametrize("limit", [30, 128])
@pytest.mark.parametrize("name", ["x" * 200, "資料é🙂" * 40])
def test_long_identifiers_have_a_sha256_suffix_within_the_byte_limit(name, limit):
    encoded = name.encode("utf-8")
    suffix = sha256(encoded).hexdigest()[:8]
    prefix = encoded[: limit - len(suffix) - 1].decode("utf-8", errors="ignore")
    result = truncate_identifier(name, max_length=limit)
    assert result == f"{prefix}_{suffix}"
    assert len(result.encode("utf-8")) <= limit
    assert result == truncate_identifier(name, max_length=limit)


def test_long_names_with_the_same_prefix_remain_distinct():
    prefix = "same_prefix_" * 20
    assert truncate_identifier(prefix + "first") != truncate_identifier(prefix + "second")


# Session ownership guards


@pytest.fixture
def owner_table():
    return Table(
        "unit_sessions",
        MetaData(),
        Column("session_id", String(128), primary_key=True),
        Column("user_id", String(128)),
        Column("session_data", OracleNativeJSON),
    )


@pytest.fixture
def owner_values():
    return {
        "session_id": "shared-session",
        "user_id": "alice",
        "session_data": {"marker": "unit-test"},
    }


@pytest.fixture(params=[False, True], ids=["sync", "async"])
def owner_merge_runner(request, owner_table, owner_values):
    connection = MagicMock()
    if request.param:
        connection.execute = AsyncMock()

    async def run(**kwargs):
        if request.param:
            return await utils.amerge_upsert(
                connection,
                owner_table,
                key_columns=["session_id"],
                values=owner_values,
                **kwargs,
            )
        return utils.merge_upsert(
            connection,
            owner_table,
            key_columns=["session_id"],
            values=owner_values,
            **kwargs,
        )

    return connection, run


def test_owner_guard_is_in_update_and_preserves_typed_binds(owner_table, owner_values):
    statement = utils.build_merge_statement(
        owner_table,
        ["session_id"],
        list(owner_values),
        owner_column="user_id",
    )
    sql = str(statement)

    assert "ON (t.session_id = s.session_id)" in sql
    guard = "WHERE (t.user_id IS NULL OR t.user_id = s.user_id)"
    assert guard in sql
    assert sql.index("WHEN MATCHED") < sql.index(guard)
    assert sql.index(guard) < sql.index("WHEN NOT MATCHED")

    assert "alice" not in sql
    assert "unit-test" not in sql
    assert statement._bindparams["session_data"].type is owner_table.c.session_data.type


def test_unguarded_sql_remains_unguarded(owner_table, owner_values):
    statement = utils.build_merge_statement(owner_table, ["session_id"], list(owner_values))
    assert " WHERE " not in str(statement)


@pytest.mark.parametrize("owner_column", ["missing_column", "session_id"])
def test_invalid_owner_guard_is_rejected(owner_table, owner_values, owner_column):
    with pytest.raises(ValueError, match="non-key column"):
        utils.build_merge_statement(
            owner_table,
            ["session_id"],
            list(owner_values),
            owner_column=owner_column,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("rowcount", [0, 1])
async def test_guarded_helper_returns_write_count(owner_merge_runner, rowcount):
    connection, run = owner_merge_runner
    connection.execute.return_value = SimpleNamespace(rowcount=rowcount)

    result = await run(owner_column="user_id")

    assert result == rowcount
    connection.execute.assert_called_once()
    connection.commit.assert_not_called()


@pytest.mark.asyncio
async def test_unguarded_helper_keeps_none_return(owner_merge_runner):
    connection, run = owner_merge_runner
    connection.execute.return_value = SimpleNamespace(rowcount=None)

    assert await run() is None
    connection.commit.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("rowcount", [-1, None, 2])
async def test_unknown_guarded_write_count_raises(owner_merge_runner, rowcount):
    connection, run = owner_merge_runner
    connection.execute.return_value = SimpleNamespace(rowcount=rowcount)

    with pytest.raises(RuntimeError, match="reliable row count"):
        await run(owner_column="user_id")


@pytest.mark.asyncio
async def test_retry_keeps_owner_guard(owner_merge_runner):
    connection, run = owner_merge_runner
    duplicate_key = Exception(SimpleNamespace(code=1))
    connection.execute.side_effect = [
        duplicate_key,
        SimpleNamespace(rowcount=0),
    ]

    assert await run(owner_column="user_id") == 0
    assert connection.execute.call_count == 2

    for call in connection.execute.call_args_list:
        sql = str(call.args[0])
        assert "WHERE (t.user_id IS NULL OR t.user_id = s.user_id)" in sql


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
async def test_denied_session_write_does_not_read_other_users_data(monkeypatch, owner_table, async_mode):
    session = MagicMock()
    session.__enter__.return_value = session
    session.__aenter__.return_value = session

    # Any session-data read after rejection must make this test fail.
    unexpected_read = AssertionError("Unexpected read of another user's session")

    if async_mode:
        db = AsyncOracleDb.__new__(AsyncOracleDb)
        db._get_table = AsyncMock(return_value=owner_table)
        db.async_session_factory = MagicMock(return_value=session)
        session.execute = AsyncMock(side_effect=unexpected_read)
        helper = AsyncMock(return_value=0)
        monkeypatch.setattr(async_module, "amerge_upsert", helper)
    else:
        db = OracleDb.__new__(OracleDb)
        db._get_table = MagicMock(return_value=owner_table)
        db.Session = MagicMock(return_value=session)
        session.execute = MagicMock(side_effect=unexpected_read)
        helper = MagicMock(return_value=0)
        monkeypatch.setattr(sync_module, "merge_upsert", helper)

    incoming = AgentSession(
        session_id="shared-session",
        user_id="bob",
        session_data={"message": "Bob's data"},
    )

    if async_mode:
        result = await db.upsert_session(incoming, deserialize=False)
    else:
        result = db.upsert_session(incoming, deserialize=False)

    assert result is None
    helper.assert_called_once()
    assert helper.call_args.kwargs["owner_column"] == "user_id"
    session.execute.assert_not_called()


# Run ordering


@pytest.fixture
def run_table():
    return Table(
        "unit_runs",
        MetaData(),
        Column("run_id", String(128), primary_key=True),
        Column("user_id", String(128)),
        Column("run_index", BigInteger),
        Column("created_at", BigInteger),
        Column("status", String(32)),
        Column("run_data", OracleNativeJSON),
        Column("date", BigInteger),
    )


@pytest.fixture
def run_values():
    return {
        "run_id": "unit-run",
        "user_id": "alice",
        "run_index": 99,
        "created_at": 1700000000,
        "status": "COMPLETED",
        "run_data": {"content": "updated"},
    }


def assert_run_policy(statement):
    sql = str(statement)
    update = sql.split("WHEN MATCHED THEN UPDATE SET ", 1)[1].split("WHEN NOT MATCHED", 1)[0]
    assert "t.run_index = COALESCE(t.run_index, s.run_index)" in update
    assert "t.created_at =" not in update
    assert "t.status = s.status" in update
    insert = sql.split("WHEN NOT MATCHED THEN INSERT", 1)[1]
    assert "created_at" in insert
    assert "run_index" in insert


def test_fill_null_sql_preserves_created_at_and_typed_binds(run_table, run_values):
    statement = utils.build_merge_statement(
        run_table,
        ["run_id"],
        list(run_values),
        preserve_on_conflict=["created_at"],
        fill_null_on_conflict=["run_index"],
    )
    assert_run_policy(statement)
    for name in run_values:
        assert statement._bindparams[name].type is run_table.c[name].type
    assert "alice" not in str(statement)
    assert "COMPLETED" not in str(statement)


def test_default_merge_policy_is_unchanged(run_table, run_values):
    statement = utils.build_merge_statement(run_table, ["run_id"], list(run_values))
    sql = str(statement)
    assert "COALESCE" not in sql
    assert "t.run_index = s.run_index" in sql
    assert "t.created_at = s.created_at" in sql


def test_fill_null_policy_keeps_reserved_column_quoting(run_table):
    statement = utils.build_merge_statement(run_table, ["run_id"], ["run_id", "date"], fill_null_on_conflict=["date"])
    assert 't."date" = COALESCE(t."date", s."date")' in str(statement)


@pytest.mark.parametrize("column", ["missing", "run_id", "created_at"])
def test_invalid_fill_null_policy_is_rejected(run_table, run_values, column):
    with pytest.raises(ValueError, match="fill_null_on_conflict"):
        utils.build_merge_statement(
            run_table,
            ["run_id"],
            list(run_values),
            preserve_on_conflict=["created_at"],
            fill_null_on_conflict=[column],
        )


@pytest.fixture(params=["sync", "async"])
def run_merge_runner(request, run_table, run_values):
    connection = MagicMock()
    if request.param == "async":
        connection.execute = AsyncMock()

    async def run(**kwargs):
        args = (connection, run_table)
        options = dict(key_columns=["run_id"], values=run_values, **kwargs)
        if request.param == "async":
            return await utils.amerge_upsert(*args, **options)
        return utils.merge_upsert(*args, **options)

    return connection, run


@pytest.mark.asyncio
@pytest.mark.parametrize("guarded", [False, True])
async def test_retry_keeps_fill_null_policy_and_owner_guard(run_merge_runner, guarded):
    connection, run = run_merge_runner
    connection.execute.side_effect = [Exception(SimpleNamespace(code=1)), SimpleNamespace(rowcount=1)]
    result = await run(
        preserve_on_conflict=["created_at"],
        fill_null_on_conflict=["run_index"],
        owner_column="user_id" if guarded else None,
    )
    assert result == (1 if guarded else None)
    assert connection.execute.call_count == 2
    for call in connection.execute.call_args_list:
        assert_run_policy(call.args[0])
        guard = "WHERE (t.user_id IS NULL OR t.user_id = s.user_id)"
        assert (guard in str(call.args[0])) is guarded
        assert call.args[1]["run_index"] == 99
    connection.commit.assert_not_called()
    connection.rollback.assert_not_called()


@pytest.mark.asyncio
async def test_invalid_policy_is_rejected_before_execution(run_merge_runner):
    connection, run = run_merge_runner
    with pytest.raises(ValueError, match="fill_null_on_conflict"):
        await run(fill_null_on_conflict=["run_id"])
    connection.execute.assert_not_called()


@pytest.mark.asyncio
async def test_other_database_errors_propagate_without_retry(run_merge_runner):
    connection, run = run_merge_runner
    error = RuntimeError("Deliberate non-unique failure")
    connection.execute.side_effect = error
    with pytest.raises(RuntimeError) as caught:
        await run(preserve_on_conflict=["created_at"], fill_null_on_conflict=["run_index"])
    assert caught.value is error
    assert connection.execute.call_count == 1
    connection.commit.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_upsert_run_enables_the_policy_without_connecting(monkeypatch, run_table, mode):
    session = MagicMock()
    session.__enter__.return_value = session
    session.__aenter__.return_value = session
    if mode == "async":
        database = AsyncOracleDb.__new__(AsyncOracleDb)
        database._get_table = AsyncMock(return_value=run_table)
        database.async_session_factory = MagicMock(return_value=session)
        helper = AsyncMock()
        monkeypatch.setattr(async_module, "amerge_upsert", helper)
        session.execute = AsyncMock(side_effect=AssertionError("Unexpected database operation"))
    else:
        database = OracleDb.__new__(OracleDb)
        database._get_table = MagicMock(return_value=run_table)
        database.Session = MagicMock(return_value=session)
        helper = MagicMock()
        monkeypatch.setattr(sync_module, "merge_upsert", helper)
        session.execute = MagicMock(side_effect=AssertionError("Unexpected database operation"))

    run = {"run_id": "unit-run", "agent_id": "unit-agent", "content": "updated", "created_at": 1700000000}
    if mode == "async":
        await database.upsert_run(run, session_id="unit-session", user_id="alice", run_index=99)
    else:
        database.upsert_run(run, session_id="unit-session", user_id="alice", run_index=99)

    helper.assert_called_once()
    assert helper.call_args.kwargs["preserve_on_conflict"] == ["created_at"]
    assert helper.call_args.kwargs["fill_null_on_conflict"] == ["run_index"]
    assert helper.call_args.kwargs["values"]["run_index"] == 99
    session.execute.assert_not_called()


# Tool payload cleanup


@pytest.fixture(params=["sync", "async"])
def cleanup_case(request, monkeypatch):
    mode = request.param
    module = async_module if mode == "async" else sync_module
    table = Table(
        "unit_tool_results",
        MetaData(),
        Column("result_id", String, primary_key=True),
        Column("session_id", String),
        Column("namespace", String),
        Column("path", String),
    )
    state = SimpleNamespace(
        rows=[("result-1", "tool-results/s1", "results/report.txt")],
        found=True,
        payload_calls=[],
        index_calls=[],
        payload_error=None,
        warnings=[],
    )

    def execute(statement, parameters=None):
        if getattr(statement, "is_select", False):
            return SimpleNamespace(fetchall=lambda: list(state.rows))
        if isinstance(statement, TextClause):
            state.payload_calls.append((str(statement), parameters))
            if state.payload_error is not None:
                raise state.payload_error
            return SimpleNamespace(rowcount=1 if state.found else 0)
        if getattr(statement, "is_delete", False):
            state.index_calls.append(statement)
            return SimpleNamespace(rowcount=len(state.rows))
        raise AssertionError("Unexpected SQL operation")

    session = MagicMock()
    session.__enter__.return_value = session
    session.__aenter__.return_value = session
    if mode == "async":
        database = AsyncOracleDb.__new__(AsyncOracleDb)
        database._get_table = AsyncMock(return_value=table)
        database.async_session_factory = MagicMock(return_value=session)
        session.execute = AsyncMock(side_effect=execute)
        exists = AsyncMock(return_value=True)
        monkeypatch.setattr(module, "ais_table_available", exists)
    else:
        database = OracleDb.__new__(OracleDb)
        database._get_table = MagicMock(return_value=table)
        database.Session = MagicMock(return_value=session)
        session.execute = MagicMock(side_effect=execute)
        exists = MagicMock(return_value=True)
        monkeypatch.setattr(module, "is_table_available", exists)
    database.db_schema = "OTHER_SESSION_OWNER"
    database.tool_result_filesystems = []
    monkeypatch.setattr(module, "log_warning", state.warnings.append)

    async def run(ids=None):
        if ids is None:
            ids = ["s1"]
        if mode == "async":
            return await database._cascade_tool_results(ids)
        return database._cascade_tool_results(ids)

    state.database = database
    state.exists = exists
    state.session = session
    state.run = run
    return state


@pytest.mark.parametrize("adapter", [OracleDb, AsyncOracleDb], ids=["sync", "async"])
def test_default_payload_table_matches_oracle_filesystem_default(adapter):
    assert adapter._default_payload_table() == "agno_fs"


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["empty-ids", "missing-index", "empty-index"])
async def test_empty_cascade_does_not_access_payloads(cleanup_case, reason):
    case = cleanup_case
    ids = [] if reason == "empty-ids" else ["s1"]
    if reason == "missing-index":
        case.database._get_table.return_value = None
    if reason == "empty-index":
        case.rows = []
    await case.run(ids)
    case.exists.assert_not_called()
    assert case.payload_calls == []
    assert case.index_calls == []
    assert case.warnings == []
    if reason == "empty-ids":
        case.database._get_table.assert_not_called()


@pytest.mark.asyncio
async def test_fresh_adapter_deletes_by_bound_namespace_and_path(cleanup_case):
    case = cleanup_case
    namespace = "tool-results/s1'--"
    path = "results/report's.txt"
    case.rows = [("result-1", namespace, path)]
    await case.run()
    assert len(case.payload_calls) == 1
    sql, parameters = case.payload_calls[0]
    assert sql == "DELETE FROM agno_fs WHERE namespace = :namespace AND path = :path"
    assert namespace not in sql and path not in sql
    assert parameters == {"namespace": namespace, "path": path}
    # The filesystem default is not qualified with the session-table schema.
    assert case.exists.call_args.args == (case.session, "agno_fs")
    assert len(case.index_calls) == 1
    assert case.warnings == []


@pytest.mark.asyncio
async def test_missing_default_table_is_not_created(cleanup_case):
    case = cleanup_case
    case.exists.return_value = False
    await case.run()
    assert case.payload_calls == []
    assert len(case.index_calls) == 1
    assert any("no store for" in warning for warning in case.warnings)


def register_filesystem(case, monkeypatch, *, fails=False):
    import agno.fs

    calls = []

    def file_system(*, backend, namespace):
        def delete(path):
            calls.append((namespace, path))
            if fails:
                raise RuntimeError("Deliberate registered-filesystem failure")
            return True

        return SimpleNamespace(delete=delete)

    case.database.tool_result_filesystems = [SimpleNamespace(backend=object())]
    monkeypatch.setattr(agno.fs, "FileSystem", file_system)
    return calls


@pytest.mark.asyncio
async def test_registered_filesystem_success_is_not_deleted_twice(cleanup_case, monkeypatch):
    case = cleanup_case
    calls = register_filesystem(case, monkeypatch)
    await case.run()
    assert calls == [(case.rows[0][1], case.rows[0][2])]
    assert case.payload_calls == []
    assert len(case.index_calls) == 1
    assert case.warnings == []


@pytest.mark.asyncio
async def test_registered_filesystem_failure_still_tries_default(cleanup_case, monkeypatch):
    case = cleanup_case
    calls = register_filesystem(case, monkeypatch, fails=True)
    await case.run()
    assert len(calls) == 1
    assert len(case.payload_calls) == 1
    assert len(case.index_calls) == 1
    assert any("Deliberate registered-filesystem failure" in warning for warning in case.warnings)
    assert not any("no store for" in warning for warning in case.warnings)


@pytest.mark.asyncio
async def test_unreachable_payload_is_still_reported(cleanup_case):
    case = cleanup_case
    case.found = False
    await case.run()
    assert len(case.payload_calls) == 1
    assert len(case.index_calls) == 1
    assert any("no store for" in warning for warning in case.warnings)


@pytest.mark.asyncio
async def test_default_delete_error_is_reported_before_index_deletion(cleanup_case):
    case = cleanup_case
    case.payload_error = RuntimeError("Deliberate default-payload deletion failure")
    assert await case.run() is None
    assert len(case.payload_calls) == 1
    assert case.index_calls == []
    assert any("Deliberate default-payload deletion failure" in warning for warning in case.warnings)


@pytest.mark.asyncio
async def test_duplicate_locator_is_deleted_once(cleanup_case):
    case = cleanup_case
    case.rows.append(("result-2", case.rows[0][1], case.rows[0][2]))
    await case.run()
    assert len(case.payload_calls) == 1
    assert len(case.index_calls) == 1
    assert case.warnings == []


# Concurrent trace writes

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)
CONTEXT_FIELDS = ("run_id", "session_id", "user_id", "agent_id", "team_id", "workflow_id")


def trace_values(role="root"):
    root = role == "root"
    return {
        "trace_id": "shared-trace",
        "name": "Workflow.run" if root else "fetch_data",
        "status": "OK" if root else "ERROR",
        "start_time": (BASE + timedelta(seconds=0 if root else 2)).isoformat(),
        "end_time": (BASE + timedelta(seconds=4 if root else 6)).isoformat(),
        "duration_ms": 4000,
        "run_id": "root-run" if root else None,
        "session_id": "root-session" if root else None,
        "user_id": "alice" if root else None,
        "agent_id": None if root else "agent-1",
        "team_id": None,
        "workflow_id": "workflow-1" if root else None,
        "created_at": (BASE + timedelta(seconds=-10 if root else 2)).isoformat(),
        "total_spans": 99,
        "error_count": 77,
    }


def stored_values(values):
    result = {key: value for key, value in values.items() if key not in {"total_spans", "error_count"}}
    result["user_id"] = to_db_user_id(result["user_id"])
    return result


def expected_merge(existing, incoming):
    result = stored_values(incoming)
    result.update(
        name="Workflow.run",
        start_time=BASE.isoformat(),
        end_time=(BASE + timedelta(seconds=6)).isoformat(),
        duration_ms=6000,
        created_at=existing["created_at"],
    )
    for field in CONTEXT_FIELDS:
        result[field] = existing.get(field) or result.get(field)
    return result


def oracle_integrity_error(code):
    driver_error = Exception(SimpleNamespace(code=code))
    return IntegrityError("test statement", {}, driver_error)


@pytest.fixture(params=["sync", "async"])
def trace_case(request, monkeypatch):
    asynchronous = request.param == "async"
    adapter = AsyncOracleDb if asynchronous else OracleDb
    module = importlib.import_module(f"agno.db.oracle.{'async_oracle' if asynchronous else 'oracle'}")
    database = adapter.__new__(adapter)
    names = (
        "trace_id",
        "name",
        "status",
        "start_time",
        "end_time",
        "run_id",
        "session_id",
        "user_id",
        "agent_id",
        "team_id",
        "workflow_id",
        "created_at",
    )
    table = Table(
        "unit_traces",
        MetaData(),
        *(Column(name, String(255)) for name in names),
        Column("duration_ms", BigInteger),
    )
    state = SimpleNamespace(reads=[], insert_errors=[], selects=[], inserts=[], errors=[])
    session = MagicMock()
    session.__enter__.return_value = session
    session.__aenter__ = AsyncMock(return_value=session)

    def execute(statement):
        if isinstance(statement, Select):
            assert statement._for_update_arg is not None, "Existing trace reads must lock the row"
            state.selects.append(statement)
            response = state.reads.pop(0)
            if isinstance(response, Exception):
                raise response
            row = None if response is None else SimpleNamespace(_mapping=response)
            return SimpleNamespace(fetchone=lambda: row)
        if isinstance(statement, Insert):
            state.inserts.append(statement.compile().params)
            failure = state.insert_errors.pop(0) if state.insert_errors else None
            if failure is not None:
                raise failure
            return SimpleNamespace(rowcount=1)
        raise AssertionError(f"Unexpected statement type: {type(statement).__name__}")

    session.execute = AsyncMock(side_effect=execute) if asynchronous else MagicMock(side_effect=execute)
    factory = MagicMock(return_value=session)
    if asynchronous:
        database.async_session_factory = factory
        database._get_table = AsyncMock(return_value=table)
    else:
        database.Session = factory
        database._get_table = MagicMock(return_value=table)
    merge = AsyncMock() if asynchronous else MagicMock()
    monkeypatch.setattr(module, "amerge_upsert" if asynchronous else "merge_upsert", merge)
    monkeypatch.setattr(module, "log_error", state.errors.append)

    async def run(values):
        trace = SimpleNamespace(to_dict=lambda: dict(values))
        if asynchronous:
            return await database.upsert_trace(trace)
        return database.upsert_trace(trace)

    state.run, state.merge = run, merge
    state.session, state.factory = session, factory
    return state


def assert_one_transaction(case):
    case.factory.assert_called_once_with()
    case.session.begin.assert_called_once_with()
    case.session.commit.assert_not_called()
    case.session.rollback.assert_not_called()


def assert_merge(case, existing, incoming):
    assert not case.errors
    case.merge.assert_called_once()
    assert case.merge.call_args.kwargs["key_columns"] == ["trace_id"]
    assert case.merge.call_args.kwargs["preserve_on_conflict"] == ["created_at"]
    assert case.merge.call_args.kwargs["values"] == expected_merge(existing, incoming)
    assert_one_transaction(case)


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", [None, "", "alice"])
async def test_new_trace_uses_insert_and_preserves_owner_encoding(trace_case, owner):
    case = trace_case
    incoming = trace_values()
    incoming["user_id"] = owner
    case.reads = [None]
    assert await case.run(incoming) is None
    assert case.inserts == [stored_values(incoming)]
    case.merge.assert_not_called()
    assert not case.errors
    assert_one_transaction(case)


@pytest.mark.asyncio
@pytest.mark.parametrize("first_role", ["root", "child"])
async def test_existing_trace_combines_window_name_and_context(trace_case, first_role):
    case = trace_case
    existing = stored_values(trace_values(first_role))
    incoming = trace_values("child" if first_role == "root" else "root")
    case.reads = [existing]
    await case.run(incoming)
    assert not case.inserts
    assert_merge(case, existing, incoming)


@pytest.mark.asyncio
async def test_duplicate_insert_reloads_winner_before_merging(trace_case):
    case = trace_case
    existing, incoming = stored_values(trace_values()), trace_values("child")
    case.reads = [None, existing]
    case.insert_errors = [oracle_integrity_error(1)]
    await case.run(incoming)
    assert len(case.selects) == 2
    assert len(case.inserts) == 1
    assert_merge(case, existing, incoming)


@pytest.mark.asyncio
async def test_disappearing_winner_can_retry_again_within_bound(trace_case):
    case = trace_case
    assert DEFAULT_MERGE_RETRY_ATTEMPTS >= 3
    existing, incoming = stored_values(trace_values()), trace_values("child")
    case.reads = [None, None, existing]
    case.insert_errors = [oracle_integrity_error(1), oracle_integrity_error(1)]
    await case.run(incoming)
    assert len(case.selects) == 3
    assert len(case.inserts) == 2
    assert_merge(case, existing, incoming)


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [1400, 2290])
async def test_other_integrity_errors_are_logged_without_retry(trace_case, code):
    case = trace_case
    case.reads = [None]
    case.insert_errors = [oracle_integrity_error(code)]
    assert await case.run(trace_values()) is None
    assert len(case.selects) == len(case.inserts) == 1
    assert len(case.errors) == 1
    assert case.errors[0].startswith("Error creating trace:")
    case.merge.assert_not_called()
    assert_one_transaction(case)


@pytest.mark.asyncio
async def test_unique_conflicts_stop_at_retry_limit_and_are_logged(trace_case):
    case = trace_case
    case.reads = [None] * DEFAULT_MERGE_RETRY_ATTEMPTS
    case.insert_errors = [oracle_integrity_error(1) for _ in case.reads]
    assert await case.run(trace_values()) is None
    assert len(case.selects) == len(case.inserts) == DEFAULT_MERGE_RETRY_ATTEMPTS
    assert len(case.errors) == 1
    case.merge.assert_not_called()
    assert_one_transaction(case)


@pytest.mark.asyncio
async def test_read_error_is_logged_without_insert_or_merge(trace_case):
    case = trace_case
    case.reads = [RuntimeError("deliberate read failure")]
    assert await case.run(trace_values()) is None
    assert len(case.errors) == 1
    assert "deliberate read failure" in case.errors[0]
    assert not case.inserts
    case.merge.assert_not_called()


@pytest.mark.asyncio
async def test_merge_error_is_logged_without_breaking_application(trace_case):
    case = trace_case
    case.reads = [stored_values(trace_values())]
    case.merge.side_effect = RuntimeError("deliberate merge failure")
    assert await case.run(trace_values("child")) is None
    assert len(case.errors) == 1
    assert "deliberate merge failure" in case.errors[0]
    assert not case.inserts


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ["", "alice"])
async def test_existing_owner_and_context_survive_conflicting_child(trace_case, owner):
    case = trace_case
    existing = stored_values(trace_values())
    existing["user_id"] = to_db_user_id(owner)
    incoming = trace_values("child")
    incoming.update(user_id="bob", session_id="other-session", run_id="other-run")
    case.reads = [None, existing]
    case.insert_errors = [oracle_integrity_error(1)]
    await case.run(incoming)
    assert_merge(case, existing, incoming)


# Reserved owner identifiers


@pytest.fixture(params=["storage", "vector"])
def owner_codec(request):
    if request.param == "storage":
        return SimpleNamespace(
            encode=storage.to_db_user_id,
            decode=storage.from_db_user_id,
            marker=storage.UNOWNED_USER_ID,
            other_marker=vector._UNOWNED_SENTINEL,
        )
    return SimpleNamespace(
        encode=vector._to_db_user_id,
        decode=vector._from_db_user_id,
        marker=vector._UNOWNED_SENTINEL,
        other_marker=storage.UNOWNED_USER_ID,
    )


@pytest.mark.parametrize("owner", [None, "", "alice", "~alice", "__agno_unowned___", "é" * 128])
def test_valid_owners_keep_the_existing_representation_and_round_trip(owner_codec, owner):
    encoded = owner_codec.encode(owner)
    expected = owner_codec.marker if owner == "" else owner
    assert encoded == expected
    assert owner_codec.decode(encoded) == owner


def test_only_this_adapters_own_marker_is_reserved(owner_codec):
    assert owner_codec.encode(owner_codec.other_marker) == owner_codec.other_marker
    assert owner_codec.decode(owner_codec.other_marker) == owner_codec.other_marker


def test_literal_marker_is_rejected_as_caller_input(owner_codec):
    with pytest.raises(ValueError, match="reserved"):
        owner_codec.encode(owner_codec.marker)


def test_stored_marker_still_decodes_to_a_legitimate_empty_owner(owner_codec):
    assert owner_codec.decode(owner_codec.marker) == ""
    assert owner_codec.encode("") == owner_codec.marker


def test_none_empty_and_named_owners_have_distinct_stored_values(owner_codec):
    assert len({owner_codec.encode(owner) for owner in (None, "", "alice")}) == 3


@pytest.fixture
def vector_store(monkeypatch):
    database = vector.OracleVector.__new__(vector.OracleVector)
    database.table = Table(
        "offline_vectors",
        MetaData(),
        Column("id", String),
        Column("user_id", String),
        Column("content_hash", String),
    )
    database._user_id_column_exists = MagicMock(return_value=True)
    database.Session = MagicMock(side_effect=AssertionError("Unexpected database session"))
    database.embedder = MagicMock()
    embed = MagicMock(side_effect=AssertionError("Embedding must not precede owner validation"))
    aembed = AsyncMock(side_effect=AssertionError("Embedding must not precede owner validation"))
    monkeypatch.setattr(vector, "embed_before_replace", embed)
    monkeypatch.setattr(vector, "aembed_before_replace", aembed)
    database._async_embed_documents = AsyncMock(side_effect=AssertionError("Unexpected embedding"))
    return SimpleNamespace(db=database, embed=embed, aembed=aembed)


def assert_no_database_or_embedding(store):
    store.db._user_id_column_exists.assert_not_called()
    store.db.Session.assert_not_called()
    store.db.embedder.get_embedding_and_usage.assert_not_called()
    store.db._async_embed_documents.assert_not_called()
    store.embed.assert_not_called()
    store.aembed.assert_not_called()


def test_vector_guard_rejects_before_metadata_inspection(vector_store):
    with pytest.raises(ValueError, match="reserved"):
        vector_store.db._require_owner_column(vector._UNOWNED_SENTINEL)
    assert_no_database_or_embedding(vector_store)


@pytest.mark.parametrize(
    "method", ["insert", "upsert", "search", "content_hash_exists", "delete_by_content_id", "_delete_by_content_hash"]
)
def test_sync_vector_entry_points_block_reserved_owner_before_work(vector_store, method):
    database = vector_store.db
    options = {"user_id": vector._UNOWNED_SENTINEL}
    if method in {"insert", "upsert"}:
        options.update(content_hash="protected-hash", documents=[Document(content="Must not be embedded")])
    elif method == "search":
        options["query"] = "Must not be embedded"
    elif method == "delete_by_content_id":
        options["content_id"] = "protected-content"
    else:
        options["content_hash"] = "protected-hash"
    with pytest.raises(ValueError, match="reserved"):
        getattr(database, method)(**options)
    assert_no_database_or_embedding(vector_store)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["async_insert", "async_upsert"])
@pytest.mark.parametrize("empty", [False, True], ids=["with-document", "empty-list"])
async def test_async_ingestion_blocks_reserved_owner_before_work(vector_store, method, empty):
    documents = [] if empty else [Document(content="Must not be embedded")]
    with pytest.raises(ValueError, match="reserved"):
        await getattr(vector_store.db, method)(
            content_hash="protected-hash", documents=documents, user_id=vector._UNOWNED_SENTINEL
        )
    assert_no_database_or_embedding(vector_store)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["async_insert", "async_upsert"])
@pytest.mark.parametrize("size", [0, -1])
async def test_existing_async_batch_size_validation_still_runs(vector_store, method, size):
    with pytest.raises(ValueError, match="batch_size"):
        await getattr(vector_store.db, method)(content_hash="h", documents=[], user_id="alice", batch_size=size)
    assert_no_database_or_embedding(vector_store)


@pytest.mark.parametrize("method", ["_insert_prepared_batch", "_replace_prepared_documents"])
def test_worker_helpers_reject_reserved_owner_and_still_remove_scoped_session(vector_store, method):
    with pytest.raises(ValueError, match="reserved"):
        getattr(vector_store.db, method)(content_hash="protected-hash", documents=[], user_id=vector._UNOWNED_SENTINEL)
    assert_no_database_or_embedding(vector_store)
    vector_store.db.Session.remove.assert_called_once_with()


def test_search_scope_rejects_literal_marker(vector_store):
    statement = select(vector_store.db.table.c.id)
    with pytest.raises(ValueError, match="reserved"):
        vector_store.db._apply_user_scope(statement, vector._UNOWNED_SENTINEL)
    assert_no_database_or_embedding(vector_store)


def test_none_search_scope_is_unchanged(vector_store):
    statement = select(vector_store.db.table.c.id)
    assert vector_store.db._apply_user_scope(statement, None) is statement
    assert_no_database_or_embedding(vector_store)


@pytest.mark.parametrize("owner", ["", "alice"])
def test_valid_search_scopes_keep_encoded_owner_and_shared_null_bucket(vector_store, owner):
    statement = vector_store.db._apply_user_scope(select(vector_store.db.table.c.id), owner)
    compiled = statement.compile()
    expected = vector._UNOWNED_SENTINEL if owner == "" else owner
    assert list(compiled.params.values()) == [expected]
    assert "IS NULL" in str(compiled)
    vector_store.db._user_id_column_exists.assert_called_once_with()
    vector_store.db.Session.assert_not_called()


# Differential fixture isolation


class FixtureHarness:
    def __init__(self, variant, monkeypatch):
        self.asynchronous = variant == "async"
        self.module = differential_async_module if self.asynchronous else differential_sync_module
        self.protected = {
            "agno_tool_results",
            "agno_schema_versions",
            "agno_spans",
            "unrelated_table",
        }
        self.catalog = set(self.protected)
        self.instances = []
        self.connections = []
        self.drop_attempts = []
        self.dropped = []
        self.reject_existing = False
        self.metadata_error = False
        self.cleanup_error = False

        constructor = "AsyncOracleDb" if self.asynchronous else "OracleDb"
        monkeypatch.setattr(self.module, constructor, self.make_database)
        monkeypatch.setattr(self.module, "inspect", self.inspect_connection)
        monkeypatch.setattr(
            self.module,
            "create_engine",
            Mock(side_effect=AssertionError("Unexpected real engine construction")),
        )

    def make_database(self, **kwargs):
        engine = MagicMock()
        connection = MagicMock()
        connection.__enter__.return_value = connection
        connection.__aenter__.return_value = connection
        engine.connect.return_value = connection
        engine.begin.return_value = connection
        connection.run_sync = AsyncMock(side_effect=lambda callback: callback(connection))
        executor = AsyncMock if self.asynchronous else Mock
        connection.execute = executor(side_effect=self.execute_drop)
        database = SimpleNamespace(
            db_engine=engine,
            Session=MagicMock(),
            close=AsyncMock(),
            tool_results_table_name="agno_tool_results",
            table_options={key: value for key, value in kwargs.items() if key.endswith("_table")},
        )
        self.instances.append(database)
        self.connections.append(connection)
        if self.reject_existing:
            self.catalog.add(kwargs["session_table"])
        return database

    def inspect_connection(self, connection):
        assert any(connection is known for known in self.connections)
        return SimpleNamespace(has_table=self.has_table)

    def has_table(self, name):
        if self.metadata_error:
            raise RuntimeError("Deliberate metadata failure")
        return name in self.catalog

    def execute_drop(self, statement):
        parts = str(statement).split()
        assert parts[:2] == ["DROP", "TABLE"]
        name = parts[2]
        assert name not in self.protected, f"Cleanup targeted protected table {name}"
        assert name in self.catalog
        self.drop_attempts.append(name)
        if self.cleanup_error:
            raise RuntimeError("Deliberate cleanup failure")
        self.catalog.remove(name)
        self.dropped.append(name)

    def owned_names(self, database):
        return {*database.table_options.values(), database.tool_results_table_name}

    async def start(self):
        # Bypass pytest's fixture invocation guard, not the fixture's own body.
        # The server-gating dependency is intentionally not requested: all
        # adapters, engines, connections and inspection calls are fakes.
        body = python_inspect.unwrap(self.module.oracle_db)
        if self.asynchronous:
            assert python_inspect.isasyncgenfunction(body), "Unexpected async fixture wrapper"
        else:
            assert python_inspect.isgeneratorfunction(body), "Unexpected sync fixture wrapper"
        generator = body(_servers_up=None)
        if self.asynchronous:
            database = await generator.__anext__()
        else:
            database = next(generator)
        return generator, database

    async def finish(self, generator):
        if self.asynchronous:
            await generator.aclose()
        else:
            generator.close()

    async def fail_body(self, generator):
        error = RuntimeError("Deliberate test-body failure")
        if self.asynchronous:
            await generator.athrow(error)
        else:
            generator.throw(error)

    def assert_closed(self, database):
        if self.asynchronous:
            database.close.assert_awaited_once()
        else:
            database.Session.remove.assert_called_once()
            database.db_engine.dispose.assert_called_once()


@pytest.fixture(params=["sync", "async"])
def harness(request, monkeypatch):
    return FixtureHarness(request.param, monkeypatch)


@pytest.mark.asyncio
async def test_cleanup_uses_captured_names_and_preserves_shared_tables(harness):
    generator, database = await harness.start()
    names = harness.owned_names(database)
    try:
        assert "spans_table" in database.table_options
        assert "versions_table" in database.table_options
        assert names.isdisjoint(harness.protected)
        assert database.tool_results_table_name.startswith(("diff_tool_", "adiff_tool_"))
        harness.catalog.update(names)
        # Even a later attribute change must not redirect the saved cleanup list.
        database.tool_results_table_name = "agno_tool_results"
    finally:
        await harness.finish(generator)
    assert set(harness.dropped) == names
    assert harness.catalog == harness.protected
    harness.assert_closed(database)


@pytest.mark.asyncio
async def test_preexisting_name_is_rejected_without_deleting_anything(harness):
    harness.reject_existing = True
    with pytest.raises(RuntimeError, match="Refusing to reuse existing test tables"):
        await harness.start()
    database = harness.instances[-1]
    collision = database.table_options["session_table"]
    assert harness.catalog == harness.protected | {collision}
    assert harness.drop_attempts == []
    harness.assert_closed(database)


@pytest.mark.asyncio
async def test_test_body_failure_still_cleans_up_owned_tables(harness):
    generator, database = await harness.start()
    names = harness.owned_names(database)
    harness.catalog.update(names)
    try:
        with pytest.raises(RuntimeError, match="Deliberate test-body failure"):
            await harness.fail_body(generator)
    finally:
        await harness.finish(generator)
    assert set(harness.dropped) == names
    assert harness.catalog == harness.protected
    harness.assert_closed(database)


@pytest.mark.asyncio
async def test_two_fixtures_do_not_remove_each_others_tables(harness):
    first_generator, first_database = await harness.start()
    second_generator = None
    first_names = harness.owned_names(first_database)
    harness.catalog.update(first_names)
    try:
        second_generator, second_database = await harness.start()
        second_names = harness.owned_names(second_database)
        assert first_names.isdisjoint(second_names)
        harness.catalog.update(second_names)
        await harness.finish(first_generator)
        assert harness.catalog == harness.protected | second_names
        harness.assert_closed(first_database)
    finally:
        await harness.finish(first_generator)
        if second_generator is not None:
            await harness.finish(second_generator)
    assert harness.catalog == harness.protected
    harness.assert_closed(second_database)


@pytest.mark.asyncio
async def test_failed_metadata_preflight_never_runs_destructive_cleanup(harness):
    harness.metadata_error = True
    with pytest.raises(RuntimeError, match="Deliberate metadata failure"):
        await harness.start()
    assert harness.catalog == harness.protected
    assert harness.drop_attempts == []
    harness.assert_closed(harness.instances[-1])


@pytest.mark.asyncio
async def test_cleanup_failure_still_closes_the_database(harness):
    generator, database = await harness.start()
    names = harness.owned_names(database)
    harness.catalog.update(names)
    harness.cleanup_error = True
    try:
        with pytest.raises(RuntimeError, match="Deliberate cleanup failure"):
            await harness.finish(generator)
    finally:
        await harness.finish(generator)
    assert harness.protected.issubset(harness.catalog)
    assert len(harness.drop_attempts) == 1
    assert set(harness.drop_attempts).issubset(names)
    harness.assert_closed(database)
