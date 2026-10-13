"""Offline regression tests for Oracle session ownership protection."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import Column, MetaData, String, Table

import agno.db.oracle.async_oracle as async_module
import agno.db.oracle.oracle as sync_module
from agno.db.oracle import AsyncOracleDb, OracleDb, utils
from agno.db.oracle.schemas import OracleNativeJSON
from agno.session import AgentSession


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
def merge_runner(request, owner_table, owner_values):
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
async def test_guarded_helper_returns_write_count(merge_runner, rowcount):
    connection, run = merge_runner
    connection.execute.return_value = SimpleNamespace(rowcount=rowcount)

    result = await run(owner_column="user_id")

    assert result == rowcount
    connection.execute.assert_called_once()
    connection.commit.assert_not_called()


@pytest.mark.asyncio
async def test_unguarded_helper_keeps_none_return(merge_runner):
    connection, run = merge_runner
    connection.execute.return_value = SimpleNamespace(rowcount=None)

    assert await run() is None
    connection.commit.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("rowcount", [-1, None, 2])
async def test_unknown_guarded_write_count_raises(merge_runner, rowcount):
    connection, run = merge_runner
    connection.execute.return_value = SimpleNamespace(rowcount=rowcount)

    with pytest.raises(RuntimeError, match="reliable row count"):
        await run(owner_column="user_id")


@pytest.mark.asyncio
async def test_retry_keeps_owner_guard(merge_runner):
    connection, run = merge_runner
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
