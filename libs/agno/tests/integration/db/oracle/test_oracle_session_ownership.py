"""Live Oracle ownership tests; requires an explicitly configured test schema."""

import asyncio
import os
import threading
import uuid
from types import SimpleNamespace

import agno.db.oracle.async_oracle as async_module
import agno.db.oracle.oracle as sync_module
import pytest
import pytest_asyncio
from agno.db.oracle import AsyncOracleDb, OracleDb
from agno.db.oracle.engine import (
    create_async_oracle_engine,
    create_oracle_engine,
)
from agno.db.oracle.utils import to_db_user_id
from agno.session import AgentSession
from sqlalchemy import inspect, select, text
from sqlalchemy.engine import make_url


async def _save(store, session_id, owner, message):
    incoming = AgentSession(
        session_id=session_id,
        agent_id="ownership-test",
        user_id=owner,
        session_data={"message": message},
    )
    if store.mode == "async":
        return await store.db.upsert_session(incoming, deserialize=False)

    def write():
        try:
            return store.db.upsert_session(incoming, deserialize=False)
        finally:
            store.db.Session.remove()

    return await asyncio.to_thread(write)


async def _snapshot(store, session_id):
    def read():
        with store.engine.connect() as connection:
            return dict(
                connection.execute(
                    select(store.table.c.user_id, store.table.c.session_data).where(
                        store.table.c.session_id == session_id
                    )
                )
                .mappings()
                .one()
            )

    return await asyncio.to_thread(read)


@pytest_asyncio.fixture(params=["sync", "async"])
async def oracle_store(request):
    raw_url = os.environ.get("AGNO_TEST_ORACLE_URL")
    if not raw_url:
        pytest.skip("AGNO_TEST_ORACLE_URL is not set")

    url = make_url(raw_url)
    if url.get_backend_name() != "oracle":
        pytest.fail("AGNO_TEST_ORACLE_URL must identify an Oracle database")

    engine = create_oracle_engine(url.set(drivername="oracle+oracledb"))
    async_engine = None
    cleanup_allowed = False
    suffix = uuid.uuid4().hex[:12]
    names = (f"owner_it_s_{suffix}", f"owner_it_v_{suffix}")

    def check_destination():
        with engine.connect() as connection:
            identity = tuple(
                connection.execute(
                    text(
                        "SELECT "
                        "SYS_CONTEXT('USERENV', 'SESSION_USER'), "
                        "SYS_CONTEXT('USERENV', 'CON_NAME'), "
                        "SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA') FROM dual"
                    )
                ).one()
            )
        user, container, schema = identity
        if user.upper() in {"SYS", "SYSTEM"} or container == "CDB$ROOT":
            raise RuntimeError("Refusing to test as SYS/SYSTEM or in CDB root")
        if user != schema:
            raise RuntimeError("Tests require the login user's own schema")
        for name in names:
            if inspect(engine).has_table(name):
                raise RuntimeError("A proposed temporary table already exists")

    def drop_temporary_tables():
        for name in names:
            if inspect(engine).has_table(name):
                with engine.begin() as connection:
                    connection.execute(text(f"DROP TABLE {name} CASCADE CONSTRAINTS"))

    try:
        await asyncio.to_thread(check_destination)
        cleanup_allowed = True

        if request.param == "async":
            async_engine = create_async_oracle_engine(url.set(drivername="oracle+oracledb_async"))
            db = AsyncOracleDb(
                db_engine=async_engine,
                session_table=names[0],
                versions_table=names[1],
            )
            table = await db._get_table(table_type="sessions", create_table_if_not_found=True)
        else:

            def prepare_sync():
                db = OracleDb(
                    db_engine=engine,
                    session_table=names[0],
                    versions_table=names[1],
                )
                try:
                    table = db._get_table(table_type="sessions", create_table_if_not_found=True)
                    return db, table
                finally:
                    db.Session.remove()

            db, table = await asyncio.to_thread(prepare_sync)

        assert table is not None
        yield SimpleNamespace(
            mode=request.param,
            db=db,
            table=table,
            engine=engine,
        )
    finally:
        try:
            if cleanup_allowed:
                await asyncio.to_thread(drop_temporary_tables)
        finally:
            try:
                if async_engine is not None:
                    await async_engine.dispose()
            finally:
                await asyncio.to_thread(engine.dispose)


@pytest.mark.asyncio
async def test_delayed_write_cannot_take_another_owners_session(oracle_store, monkeypatch):
    store = oracle_store
    session_id = "shared-session"

    if store.mode == "async":
        module = async_module
        helper_name = "amerge_upsert"
        ready = asyncio.Event()
        release = asyncio.Event()
        original = module.amerge_upsert

        async def hook(connection, table, key_columns, values, *args, **kwargs):
            if table.name == store.table.name and values.get("user_id") == "bob":
                ready.set()
                await asyncio.wait_for(release.wait(), timeout=30)
            return await original(connection, table, key_columns, values, *args, **kwargs)

        async def wait_for_bob():
            await asyncio.wait_for(ready.wait(), timeout=20)
    else:
        module = sync_module
        helper_name = "merge_upsert"
        ready = threading.Event()
        release = threading.Event()
        original = module.merge_upsert

        def hook(connection, table, key_columns, values, *args, **kwargs):
            if table.name == store.table.name and values.get("user_id") == "bob":
                ready.set()
                if not release.wait(timeout=30):
                    raise TimeoutError("Bob was not released")
            return original(connection, table, key_columns, values, *args, **kwargs)

        async def wait_for_bob():
            assert await asyncio.to_thread(ready.wait, 20), "Bob did not reach the write step"

    with monkeypatch.context() as context:
        context.setattr(module, helper_name, hook)
        bob_task = asyncio.create_task(_save(store, session_id, "bob", "Bob's data"))
        ready_task = asyncio.create_task(wait_for_bob())

        try:
            done, _ = await asyncio.wait(
                {ready_task, bob_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if bob_task in done:
                bob_task.result()  # Surface Bob's actual exception.
                pytest.fail("Bob finished before reaching the pause point")

            await ready_task
            alice_result = await _save(store, session_id, "alice", "Alice's data")
            assert alice_result is not None
        finally:
            ready.set()  # Release any remaining readiness waiter.
            release.set()
            await asyncio.gather(ready_task, bob_task, return_exceptions=True)

        bob_result = bob_task.result()

    stored = await _snapshot(store, session_id)
    assert bob_result is None
    assert stored["user_id"] == "alice"
    assert stored["session_data"]["message"] == "Alice's data"


@pytest.mark.asyncio
async def test_owner_state_rules_are_preserved(oracle_store):
    store = oracle_store
    cases = [
        ("alice", "alice", True),
        ("alice", "bob", False),
        ("alice", None, False),
        ("alice", "", False),
        (None, "alice", True),
        (None, None, True),
        (None, "", True),
        ("", "", True),
        ("", "alice", False),
        ("", None, False),
    ]

    for number, (existing_owner, incoming_owner, allowed) in enumerate(cases):
        session_id = f"owner-case-{number}"
        initial = await _save(store, session_id, existing_owner, "original")
        assert initial is not None

        result = await _save(store, session_id, incoming_owner, "replacement")
        stored = await _snapshot(store, session_id)

        assert (result is not None) == allowed, (existing_owner, incoming_owner, allowed)
        expected_owner = incoming_owner if allowed else existing_owner
        expected_message = "replacement" if allowed else "original"

        assert stored["user_id"] == to_db_user_id(expected_owner)
        assert stored["session_data"]["message"] == expected_message
        if allowed:
            assert result["user_id"] == incoming_owner


@pytest.mark.asyncio
@pytest.mark.parametrize("seed_null_owner", [False, True], ids=["new-session", "null-owned-session"])
async def test_concurrent_different_owners_have_one_winner(oracle_store, monkeypatch, seed_null_owner):
    store = oracle_store
    module = async_module if store.mode == "async" else sync_module
    helper_name = "amerge_upsert" if store.mode == "async" else "merge_upsert"
    original = getattr(module, helper_name)

    for attempt in range(5):
        session_id = f"concurrent-{attempt}"
        if seed_null_owner:
            initial = await _save(store, session_id, None, "unclaimed")
            assert initial is not None

        if store.mode == "async":
            start_gate = asyncio.Event()
            arrivals = 0

            async def hook(connection, table, key_columns, values, *args, **kwargs):
                nonlocal arrivals
                if table.name == store.table.name and values.get("session_id") == session_id:
                    arrivals += 1
                    if arrivals == 2:
                        start_gate.set()
                    await asyncio.wait_for(start_gate.wait(), timeout=15)

                return await original(connection, table, key_columns, values, *args, **kwargs)
        else:
            start_gate = threading.Barrier(2)

            def hook(connection, table, key_columns, values, *args, **kwargs):
                if table.name == store.table.name and values.get("session_id") == session_id:
                    start_gate.wait(timeout=15)

                return original(connection, table, key_columns, values, *args, **kwargs)

        with monkeypatch.context() as context:
            context.setattr(module, helper_name, hook)
            owners = ("alice", "bob")
            outcomes = await asyncio.gather(
                *[_save(store, session_id, owner, f"{owner} race data") for owner in owners],
                return_exceptions=True,
            )

            # Wait for both operations before propagating an error,
            # so fixture cleanup does not race with an active writer.
            for outcome in outcomes:
                if isinstance(outcome, BaseException):
                    raise outcome

        results = dict(zip(owners, outcomes))
        winners = [owner for owner, result in results.items() if result is not None]
        assert len(winners) == 1, (session_id, winners)

        winner = winners[0]
        stored = await _snapshot(store, session_id)
        assert stored["user_id"] == winner
        assert stored["session_data"]["message"] == f"{winner} race data"
        assert results[winner]["user_id"] == winner
