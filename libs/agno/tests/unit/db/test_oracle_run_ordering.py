"""Offline checks for opt-in Oracle MERGE run-order preservation."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import BigInteger, Column, MetaData, String, Table

import agno.db.oracle.async_oracle as async_module
import agno.db.oracle.oracle as sync_module
from agno.db.oracle import AsyncOracleDb, OracleDb, utils
from agno.db.oracle.schemas import OracleNativeJSON


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
def merge_runner(request, run_table, run_values):
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
async def test_retry_keeps_fill_null_policy_and_owner_guard(merge_runner, guarded):
    connection, run = merge_runner
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
async def test_invalid_policy_is_rejected_before_execution(merge_runner):
    connection, run = merge_runner
    with pytest.raises(ValueError, match="fill_null_on_conflict"):
        await run(fill_null_on_conflict=["run_id"])
    connection.execute.assert_not_called()


@pytest.mark.asyncio
async def test_other_database_errors_propagate_without_retry(merge_runner):
    connection, run = merge_runner
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
