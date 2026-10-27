"""Opt-in Oracle large-text writes and lossless tool-result offloading."""

import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import MetaData, Table, create_engine, event, func, inspect, select, text, update
from sqlalchemy.engine import make_url

from agno.db.oracle import OracleDb
from agno.db.oracle.utils import is_unique_violation
from agno.fs import FileSystem
from agno.fs.db import DbFileSystem
from agno.fs.errors import QuotaExceededError, VersionConflictError
from agno.offload import ResultStore


@pytest.fixture
def storage():
    value = os.getenv("AGNO_TEST_ORACLE_URL")
    if not value:
        pytest.skip("Set AGNO_TEST_ORACLE_URL to opt in to isolated Oracle tests")
    url = make_url(value)
    assert url.drivername == "oracle+oracledb"
    engine = create_engine(url, hide_parameters=True, pool_pre_ping=True)
    token = "lob_" + uuid4().hex[:12]
    names = {"payload": token + "_p", "index": token + "_i", "versions": token + "_v"}
    database = None
    preflight = False
    try:
        with engine.connect() as connection:
            user, schema = connection.execute(
                text("SELECT USER, SYS_CONTEXT('USERENV','CURRENT_SCHEMA') FROM dual")
            ).one()
            assert user == schema == url.username.upper()
        inspector = inspect(engine)
        assert not any(inspector.has_table(name) for name in names.values())
        preflight = True
        database = OracleDb(
            db_engine=engine, versions_table=names["versions"], json_storage="clob", create_schema=False
        )
        database.tool_results_table_name = names["index"]
        database._get_table("tool_results", create_table_if_not_found=True)
        backend = DbFileSystem(db_engine=engine, table_name=names["payload"], db_schema=None)
        backend._ensure_table()
        filesystem = FileSystem(backend=backend, namespace="tool-results", max_file_bytes=8_000_000)
        store = ResultStore(db=database, fs=filesystem, threshold_chars=16000)
        yield SimpleNamespace(engine=engine, database=database, backend=backend, store=store, names=names)
    finally:
        errors = []
        try:
            if database is not None:
                database.Session.remove()
            if preflight:
                for name in (names["index"], names["payload"], names["versions"]):
                    try:
                        Table(name, MetaData()).drop(engine, checkfirst=True)
                    except Exception as error:
                        errors.append((name, type(error).__name__))
                assert not errors, f"Owned-table cleanup failed: {errors}"
                assert not any(inspect(engine).has_table(name) for name in names.values())
        finally:
            engine.dispose()


@pytest.mark.parametrize(
    "content",
    ["", "small", "a" * 22702, "b" * 32767, "c" * 32768, "d" * 68115, "café🙂漢字\n" * 6000],
    ids=["empty", "small", "22kb", "32767", "32768", "66kb", "unicode"],
)
def test_large_file_insert_replace_reopen_and_namespace_isolation(storage, content):
    backend = storage.backend
    backend.write("bob", "report.txt", "Bob's control")
    first = backend.write("alice", "report.txt", content)
    assert first.version == 1 and first.size_bytes == len(content.encode("utf-8"))
    assert backend.read("alice", "report.txt") == content
    with storage.engine.begin() as connection:
        connection.execute(update(backend.table).where(backend.table.c.namespace == "alice").values(created_at=1234567))
        created_at = connection.execute(
            select(backend.table.c.created_at).where(backend.table.c.namespace == "alice")
        ).scalar_one()
    replacement = content[::-1] + "\nupdated"
    second = backend.write("alice", "report.txt", replacement)
    reopened = DbFileSystem(db_engine=storage.engine, table_name=backend.table_name, db_schema=None)
    assert reopened.read("alice", "report.txt").encode("utf-8") == replacement.encode("utf-8")
    assert reopened.read("bob", "report.txt") == "Bob's control"
    assert second.version == 2 and second.size_bytes == len(replacement.encode("utf-8"))
    with storage.engine.connect() as connection:
        assert connection.execute(select(func.count()).select_from(backend.table)).scalar_one() == 2
        assert (
            connection.execute(
                select(backend.table.c.created_at).where(backend.table.c.namespace == "alice")
            ).scalar_one()
            == created_at
        )


@pytest.mark.parametrize("commit", [False, True], ids=["rollback", "commit"])
def test_large_write_keeps_the_callers_transaction(storage, commit):
    backend = storage.backend
    backend.write("alice", "report.txt", "original")
    content = "café🙂" * 10000
    with storage.engine.connect() as connection:
        transaction = connection.begin()
        backend._write_on(connection, "alice", "marker.txt", "same transaction")
        backend._write_on(connection, "alice", "report.txt", content)
        assert transaction.is_active
        if commit:
            transaction.commit()
        else:
            transaction.rollback()
    assert backend.read("alice", "marker.txt") == ("same transaction" if commit else None)
    assert backend.read("alice", "report.txt") == (content if commit else "original")


def test_concurrent_first_large_writes_retry_without_losing_a_version(storage):
    backend = storage.backend
    barrier = threading.Barrier(2)
    state = threading.local()
    reached = set()
    conflicts = []
    lock = threading.Lock()

    def meet_after_first_update(connection, cursor, statement, parameters, context, executemany):
        if statement.startswith("UPDATE " + backend.table_name) and not getattr(state, "waited", False):
            state.waited = True
            with lock:
                reached.add(threading.get_ident())
            barrier.wait(timeout=15)

    def observe_conflict(context):
        if is_unique_violation(context.sqlalchemy_exception):
            with lock:
                conflicts.append(True)

    event.listen(storage.engine, "after_cursor_execute", meet_after_first_update)
    event.listen(storage.engine, "handle_error", observe_conflict)
    contents = ["A" * 68115, "café🙂" * 10000]
    try:
        with ThreadPoolExecutor(max_workers=2) as workers:
            futures = [workers.submit(backend.write, "alice", "race.txt", content) for content in contents]
            versions = [future.result(timeout=30).version for future in futures]
        assert sorted(versions) == [1, 2]
        assert len(reached) == 2 and len(conflicts) == 1
        assert backend.read("alice", "race.txt") in contents
        assert backend._stat("alice", "race.txt").version == 2
    finally:
        event.remove(storage.engine, "after_cursor_execute", meet_after_first_update)
        event.remove(storage.engine, "handle_error", observe_conflict)


def test_large_conditional_write_append_and_quota_are_unchanged(storage):
    backend = storage.backend
    backend.write("alice", "report.txt", "small")
    content = "café🙂\n" * 7000
    changed = backend.write("alice", "report.txt", content, expected_version=1)
    assert changed.version == 2 and backend.read("alice", "report.txt") == content
    with pytest.raises(VersionConflictError):
        backend.write("alice", "report.txt", "stale", expected_version=1)
    assert backend.read("alice", "report.txt") == content
    backend.append("alice", "report.txt", "end marker", max_file_bytes=100000)
    expected = content + "end marker\n"
    assert backend.read("alice", "report.txt") == expected
    with pytest.raises(QuotaExceededError):
        backend.append("alice", "report.txt", "refused", max_file_bytes=len(expected.encode("utf-8")))
    assert backend.read("alice", "report.txt") == expected


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async-bridge"])
def test_large_result_round_trip_and_session_scoped_cleanup(storage, asynchronous):
    lines = [f"row {i}: café🙂漢字" for i in range(1800)]
    lines[900] += " middle-marker-outside-preview"
    payload = "\n".join(lines)
    options = {
        "session_id": "S1",
        "run_id": "R1",
        "tool_call_id": "C1",
        "tool_name": "report",
        "tool_args": {},
        "output": payload,
        "user_id": "alice",
    }
    store = storage.store
    if asynchronous:
        reference = asyncio.run(store.aoffload(**options))
    else:
        reference = store.offload(**options)
    control = store.offload(**{**options, "session_id": "S2", "user_id": "bob"})
    row = store.get_row(reference.result_id)
    assert row["size_bytes"] == len(payload.encode("utf-8"))
    assert row["namespace"] != store.get_row(control.result_id)["namespace"]
    fresh_backend = DbFileSystem(db_engine=storage.engine, table_name=storage.names["payload"], db_schema=None)
    fresh = ResultStore(db=storage.database, fs=FileSystem(backend=fresh_backend, namespace="tool-results"))
    recovered = asyncio.run(fresh.apayload(reference.result_id)) if asynchronous else fresh.payload(reference.result_id)
    assert recovered.encode("utf-8") == payload.encode("utf-8")
    middle = (
        asyncio.run(fresh.aread(reference.result_id, start_line=901))
        if asynchronous
        else fresh.read(reference.result_id, start_line=901)
    )
    assert "middle-marker-outside-preview" in middle.text
    assert len(fresh.read(reference.result_id).text) <= 16000
    found = (
        asyncio.run(fresh.asearch(reference.result_id, "middle-marker-outside-preview"))
        if asynchronous
        else fresh.search(reference.result_id, "middle-marker-outside-preview")
    )
    assert found
    removed = asyncio.run(fresh.adelete_for_sessions(["S1"])) if asynchronous else fresh.delete_for_sessions(["S1"])
    assert removed == 1
    assert fresh.get_row(reference.result_id) is None
    assert fresh_backend.read(row["namespace"], row["path"]) is None
    assert fresh.payload(control.result_id) == payload
    assert fresh.delete_for_sessions(["S2"]) == 1


def test_failed_index_write_removes_the_new_large_payload(storage, monkeypatch):
    def fail_index(*args, **kwargs):
        raise RuntimeError("Deliberate index failure")

    monkeypatch.setattr(storage.database, "upsert_tool_result", fail_index)
    with pytest.raises(RuntimeError, match="Deliberate index failure"):
        storage.store.offload(
            session_id="S1", run_id="R1", tool_call_id="C1", tool_name="report", tool_args={}, output="x" * 68115
        )
    with storage.engine.connect() as connection:
        assert connection.execute(select(func.count()).select_from(storage.backend.table)).scalar_one() == 0
