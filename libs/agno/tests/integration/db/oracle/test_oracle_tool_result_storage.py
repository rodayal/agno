"""Opt-in Oracle large-text writes and lossless tool-result offloading."""

import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import Column, MetaData, Table, create_engine, event, func, inspect, select, text, update
from sqlalchemy.engine import make_url

from agno.db.oracle import OracleDb
from agno.db.oracle.utils import is_unique_violation
from agno.fs import FileSystem
from agno.fs.db import AsyncDbFileSystem, DbFileSystem
from agno.fs.errors import QuotaExceededError, SchemaOutdatedError, VersionConflictError
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
    names = {"payload": token + "_p", "index": token + "_i", "versions": token + "_v", "legacy": token + "_old"}
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
                for name in (names["index"], names["payload"], names["versions"], names["legacy"]):
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


@pytest.mark.parametrize("user_id", ["alice", "bob", "s:", "u:alice", ""])
def test_user_partitions_keep_all_filesystem_operations_isolated(storage, user_id):
    backend = storage.backend
    other = "control" if user_id != "control" else "other"
    backend.write("unit", "report.txt", "private control", user_id=other)
    meta = backend.write("unit", "report.txt", "café🙂" * 10000, user_id=user_id, expected_version=0)
    assert meta.version == 1 and meta.user_id == (user_id or None)
    with pytest.raises(VersionConflictError):
        backend.write("unit", "report.txt", "duplicate", user_id=user_id, expected_version=0)
    assert backend.read("unit", "report.txt", user_id=other) == "private control"
    backend.write("unit", "report.txt", "", user_id=user_id, expected_version=1)
    assert backend.read_with_meta("unit", "report.txt", user_id=user_id).content == ""
    backend.append("unit", "report.txt", "owned marker", user_id=user_id)
    assert backend.contains("unit", ["owned marker", "private control"], user_id=user_id) == {"owned marker"}
    assert len(backend.search("unit", "owned", user_id=user_id)) == 1
    assert backend.search("unit", "private control", user_id=user_id) == []
    assert backend.usage("unit", user_id=user_id).file_count == 1
    assert backend.list("unit", user_id=user_id)[0].user_id == (user_id or None)
    assert set(backend.partitions("unit")) == ({user_id, other} - {""})
    backend.move("unit", "report.txt", "moved.txt", user_id=user_id)
    assert backend.read("unit", "report.txt", user_id=user_id) is None
    assert backend.delete("unit", "moved.txt", user_id=user_id)
    assert backend.read("unit", "report.txt", user_id=other) == "private control"


@pytest.mark.asyncio
async def test_native_async_filesystem_keeps_partitions_and_large_content(storage):
    from agno.db.oracle.engine import create_async_oracle_engine

    engine = create_async_oracle_engine(
        storage.engine.url.set(drivername="oracle+oracledb_async"), hide_parameters=True
    )
    backend = AsyncDbFileSystem(SimpleNamespace(db_engine=engine), table_name=storage.names["payload"], db_schema=None)
    try:
        await backend.awrite("unit", "report.txt", "Bob", user_id="bob")
        content = "café🙂" * 10000
        first = await backend.awrite("unit", "report.txt", content, user_id="alice", expected_version=0)
        assert first.version == 1 and first.user_id == "alice"
        assert await backend.aread("unit", "report.txt", user_id="alice") == content
        await backend.aappend("unit", "report.txt", "end", user_id="alice")
        with pytest.raises(QuotaExceededError):
            await backend.aappend("unit", "report.txt", "too much", user_id="alice", max_file_bytes=1)
        assert await backend.aread("unit", "report.txt", user_id="alice") == content + "\nend\n"
        await backend.amove("unit", "report.txt", "moved.txt", user_id="alice")
        assert (await backend.aread_with_meta("unit", "moved.txt", user_id="alice")).metadata.user_id == "alice"
        assert await backend.adelete("unit", "moved.txt", user_id="alice")
        assert await backend.aread("unit", "report.txt", user_id="bob") == "Bob"
        await backend.awrite("unit", "empty.txt", "")
        assert (await backend.aread_with_meta("unit", "empty.txt")).content == ""
    finally:
        await engine.dispose()


def test_explicit_legacy_upgrade_preserves_shared_files(storage):
    legacy = Table(
        storage.names["legacy"],
        MetaData(),
        *[
            Column(column.name, column.type, nullable=column.nullable, primary_key=column.name in ("namespace", "path"))
            for column in storage.backend.table.columns
            if column.name != "user_id"
        ],
    )
    legacy.create(storage.engine)
    with storage.engine.begin() as connection:
        connection.execute(
            legacy.insert().values(
                namespace="unit",
                path="report.txt",
                content="old",
                size_bytes=3,
                version=7,
                created_at=123,
                updated_at=124,
            )
        )
    backend = DbFileSystem(db_engine=storage.engine, table_name=legacy.name, db_schema=None)
    with pytest.raises(SchemaOutdatedError):
        backend.read("unit", "report.txt")
    assert backend.upgrade_schema() is True
    assert backend.read("unit", "report.txt") == "old"
    assert backend._stat("unit", "report.txt").version == 7
    backend.write("unit", "report.txt", "Alice", user_id="alice")
    assert backend.read("unit", "report.txt") == "old"
    assert backend.read("unit", "report.txt", user_id="alice") == "Alice"
    assert backend.upgrade_schema() is False


def test_session_cleanup_preserves_named_partition_with_the_same_path(storage, monkeypatch):
    ref = storage.store.offload(
        session_id="S1", run_id="R1", tool_call_id="C1", tool_name="report", tool_args={}, output="x" * 68115
    )
    row = storage.store.get_row(ref.result_id)
    storage.backend.write(row["namespace"], row["path"], "named control", user_id="bob")
    monkeypatch.setattr(storage.database, "_default_payload_table", lambda: storage.names["payload"])
    storage.database.tool_result_filesystems = []
    storage.database._cascade_tool_results(["S1"])
    assert storage.backend.read(row["namespace"], row["path"]) is None
    assert storage.backend.read(row["namespace"], row["path"], user_id="bob") == "named control"
    assert storage.store.get_row(ref.result_id) is None
