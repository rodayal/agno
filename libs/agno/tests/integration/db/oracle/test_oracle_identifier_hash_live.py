"""Long generated names and reopened Oracle tables work with SHA-256 naming."""

import os
from hashlib import sha256
from inspect import isawaitable
from uuid import uuid4

import pytest
import pytest_asyncio
from agno.db.base import SessionType
from agno.db.oracle import AsyncOracleDb, OracleDb
from agno.db.oracle import async_oracle as async_adapter
from agno.db.oracle import oracle as sync_adapter
from agno.db.oracle.utils import truncate_identifier
from agno.session import AgentSession
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.schema import CreateTable


@pytest_asyncio.fixture
async def temporary_tables():
    db_url = os.getenv("AGNO_TEST_ORACLE_URL", "")
    if not db_url:
        pytest.skip("AGNO_TEST_ORACLE_URL is not configured")
    url = make_url(db_url)
    if url.drivername != "oracle+oracledb":
        pytest.fail("A synchronous Oracle connection URL is required")
    engine = create_engine(url, hide_parameters=True)
    prefix = "test_ohash_" + uuid4().hex[:12]
    names = {
        "session_table": prefix + "_" + "x" * (127 - len(prefix) - 1),
        "runs_table": prefix + "_runs",
        "versions_table": prefix + "_versions",
    }
    owned = set(names.values())
    created = []
    databases = []

    def track_create(conn, cursor, statement, parameters, context, executemany):
        ddl = getattr(getattr(context, "compiled", None), "statement", None)
        if conn.dialect.name == "oracle" and isinstance(ddl, CreateTable) and ddl.element.name in owned:
            created.append(ddl.element.name)

    try:
        for name in owned:
            if inspect(engine).has_table(name):
                pytest.fail("Refusing to reuse an existing table")
        event.listen(Engine, "after_cursor_execute", track_create)

        def database(asynchronous):
            adapter = AsyncOracleDb if asynchronous else OracleDb
            connection_url = url.set(drivername="oracle+oracledb_async") if asynchronous else url
            instance = adapter(
                db_url=connection_url.render_as_string(hide_password=False),
                json_storage="clob",
                **names,
            )
            databases.append(instance)
            return instance

        yield database, engine, names
    finally:
        if event.contains(Engine, "after_cursor_execute", track_create):
            event.remove(Engine, "after_cursor_execute", track_create)
        for instance in databases:
            if isinstance(instance, AsyncOracleDb):
                await instance.db_engine.dispose()
            else:
                instance.Session.remove()
                instance.db_engine.dispose()
        try:
            for name in dict.fromkeys(reversed(created)):
                if inspect(engine).has_table(name):
                    stored_name = engine.dialect.denormalize_name(name)
                    quoted = engine.dialect.identifier_preparer.quote_identifier(stored_name)
                    with engine.begin() as conn:
                        conn.exec_driver_sql(f"DROP TABLE {quoted}")
            for name in owned:
                assert not inspect(engine).has_table(name), "Temporary table cleanup failed"
        finally:
            engine.dispose()


async def invoke(function, *args, **kwargs):
    result = function(*args, **kwargs)
    return await result if isawaitable(result) else result


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("existing_names", [False, True], ids=["sha256-names", "different-existing-names"])
async def test_long_table_names_can_be_written_and_reopened(
    temporary_tables, monkeypatch, asynchronous, existing_names
):
    factory, engine, names = temporary_tables
    first = factory(asynchronous)

    def previous_naming(name, max_length=128):
        # Existing physical names need not match the current naming helper.
        return name if len(name.encode("utf-8")) <= max_length else "existing_" + sha256(name.encode()).hexdigest()[:24]

    with monkeypatch.context() as naming:
        if existing_names:
            naming.setattr(sync_adapter, "truncate_identifier", previous_naming)
            naming.setattr(async_adapter, "truncate_identifier", previous_naming)
        saved = await invoke(
            first.upsert_session,
            AgentSession(session_id="hash-session", agent_id="hash-agent", user_id="alice", session_data={"count": 1}),
        )
        assert saved is not None

    with engine.connect() as connection:
        index_names = set(
            connection.execute(
                text("SELECT INDEX_NAME FROM USER_INDEXES WHERE TABLE_NAME = :name"),
                {"name": names["session_table"].upper()},
            ).scalars()
        )
    for column in ("session_type", "created_at"):
        index_name = "idx_" + names["session_table"] + "_" + column
        expected = previous_naming(index_name) if existing_names else truncate_identifier(index_name)
        assert expected.upper() in index_names

    reopened = factory(asynchronous)
    restored = await invoke(
        reopened.get_session, "hash-session", session_type=SessionType.AGENT, user_id="alice", deserialize=False
    )
    assert restored["session_data"]["count"] == 1
    await invoke(
        reopened.upsert_session,
        AgentSession(session_id="hash-session", agent_id="hash-agent", user_id="alice", session_data={"count": 2}),
    )
    restored = await invoke(
        first.get_session, "hash-session", session_type=SessionType.AGENT, user_id="alice", deserialize=False
    )
    assert restored["session_data"]["count"] == 2
