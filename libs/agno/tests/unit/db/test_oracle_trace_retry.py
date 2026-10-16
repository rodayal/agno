"""Offline trace-specific retry tests; no database URL or connection is used."""

import importlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import BigInteger, Column, MetaData, String, Table
from sqlalchemy.exc import IntegrityError
from sqlalchemy.sql.dml import Insert
from sqlalchemy.sql.selectable import Select

from agno.db.oracle import AsyncOracleDb, OracleDb
from agno.db.oracle.utils import DEFAULT_MERGE_RETRY_ATTEMPTS, to_db_user_id

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
