"""Offline regressions for Oracle's default tool-payload cleanup fallback."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import Column, MetaData, String, Table
from sqlalchemy.sql.elements import TextClause

import agno.db.oracle.async_oracle as async_module
import agno.db.oracle.oracle as sync_module
from agno.db.oracle import AsyncOracleDb, OracleDb


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
