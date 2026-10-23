"""Oracle-only live checks for differential-fixture isolation.

Requires AGNO_TEST_ORACLE_URL pointing to an approved disposable schema.
No PostgreSQL connection or model API is used. The original fixture bodies
are invoked directly, bypassing only their PostgreSQL availability dependency.
"""

import inspect as python_inspect
import os
import uuid
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url
from tests.integration.db.oracle import test_differential_async_oracle_postgres as async_module
from tests.integration.db.oracle import test_differential_oracle_postgres as sync_module


@pytest.fixture
def oracle_engine():
    raw_url = os.getenv("AGNO_TEST_ORACLE_URL", "")
    if not raw_url:
        pytest.skip("AGNO_TEST_ORACLE_URL is not set")
    url = make_url(raw_url)
    if url.drivername != "oracle+oracledb":
        raise RuntimeError("Supply the synchronous oracle+oracledb test URL")
    engine = create_engine(url)
    try:
        with engine.connect() as connection:
            user, container, schema = connection.execute(
                text(
                    "SELECT USER, SYS_CONTEXT('USERENV', 'CON_NAME'), "
                    "SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA') FROM dual"
                )
            ).one()
        if user.upper() in {"SYS", "SYSTEM"} or container == "CDB$ROOT":
            raise RuntimeError("Refusing to test as SYS/SYSTEM or in the root container")
        if user != schema:
            raise RuntimeError("Fixture checks require the login user's own schema")
        yield engine
    finally:
        engine.dispose()


def table_names(engine):
    # Ignore recycled objects; these checks do not require purging the recycle bin.
    return {name for name in inspect(engine).get_table_names() if not name.upper().startswith("BIN$")}


@contextmanager
def marker_table(engine, name):
    # Callers generate these names locally; no application table is reused.
    if inspect(engine).has_table(name):
        raise RuntimeError("Proposed control-table name already exists")
    created = False
    marker = uuid.uuid4().hex
    try:
        with engine.begin() as connection:
            connection.execute(text(f"CREATE TABLE {name} (marker VARCHAR2(64) NOT NULL)"))
            created = True
            connection.execute(text(f"INSERT INTO {name} (marker) VALUES (:marker)"), {"marker": marker})
        yield marker
    finally:
        if created and inspect(engine).has_table(name):
            with engine.begin() as connection:
                connection.execute(text(f"DROP TABLE {name} CASCADE CONSTRAINTS"))


def assert_marker(engine, name, expected):
    assert inspect(engine).has_table(name), "Fixture removed the preexisting control table"
    with engine.connect() as connection:
        actual = connection.execute(text(f"SELECT marker FROM {name}")).scalars().all()
    assert actual == [expected], "Fixture changed the preexisting control data"


def fixture_generator(variant, engine):
    module = async_module if variant == "async" else sync_module
    configured = module.ORACLE_SYNC_URL if variant == "async" else module.ORACLE_URL
    assert make_url(configured) == engine.url, "Fixture URL differs from the checked test target"
    body = python_inspect.unwrap(module.oracle_db)
    return body(_servers_up=None)


async def advance(generator, variant):
    if variant == "async":
        return await generator.__anext__()
    return next(generator)


async def finish(generator, variant):
    if variant == "async":
        await generator.aclose()
    else:
        generator.close()


async def fail_body(generator, variant):
    error = RuntimeError("Deliberate live test-body failure")
    if variant == "async":
        await generator.athrow(error)
    else:
        generator.throw(error)


async def invoke(database, variant, method, **kwargs):
    result = getattr(database, method)(**kwargs)
    if variant == "async":
        return await result
    return result


def probe_names(database):
    return {
        database.session_table_name,
        database.runs_table_name,
        database.trace_table_name,
        database.span_table_name,
        database.versions_table_name,
        database.tool_results_table_name,
    }


async def exercise_fixture(database, variant):
    for table_type in ("sessions", "runs", "spans", "tool_results"):
        table = await invoke(database, variant, "_get_table", table_type=table_type, create_table_if_not_found=True)
        assert table is not None
    row = {
        "result_id": "fixture-safety-result",
        "namespace": "fixture-safety",
        "path": "/probe",
        "session_id": "fixture-safety-session",
        "run_id": "fixture-safety-run",
        "tool_call_id": "fixture-safety-call",
        "tool_name": "fixture-safety-tool",
        "args_hash": "fixture-safety-hash",
        "content_type": "text/plain",
        "size_bytes": 5,
        "line_count": 1,
        "preview": "probe",
        "user_id": None,
        "created_at": 1700000000,
        "expires_at": None,
    }
    await invoke(database, variant, "upsert_tool_result", row=row)
    saved = await invoke(database, variant, "get_tool_result", result_id=row["result_id"])
    assert saved is not None and saved["preview"] == "probe"


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", ["sync", "async"])
@pytest.mark.parametrize("body_fails", [False, True], ids=["normal-exit", "body-failure"])
async def test_live_cleanup_preserves_preexisting_tables(oracle_engine, variant, body_fails):
    control = f"fixture_guard_{uuid.uuid4().hex[:12]}"
    before_control = table_names(oracle_engine)
    with marker_table(oracle_engine, control) as marker:
        baseline = table_names(oracle_engine)
        generator = fixture_generator(variant, oracle_engine)
        try:
            database = await advance(generator, variant)
            names = probe_names(database)
            prefix = "adiff_" if variant == "async" else "diff_"
            assert all(name.startswith(prefix) for name in names)
            assert names.isdisjoint(baseline)
            await exercise_fixture(database, variant)
            assert names.issubset(table_names(oracle_engine))
            if body_fails:
                with pytest.raises(RuntimeError, match="Deliberate live test-body failure"):
                    await fail_body(generator, variant)
        finally:
            await finish(generator, variant)
        assert table_names(oracle_engine) == baseline
        assert_marker(oracle_engine, control, marker)
    assert table_names(oracle_engine) == before_control


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", ["sync", "async"])
async def test_live_existing_name_is_refused_without_deletion(oracle_engine, variant, monkeypatch):
    module = async_module if variant == "async" else sync_module
    token = uuid.uuid4().hex
    prefix = "adiff_sess_" if variant == "async" else "diff_sess_"
    collision = f"{prefix}{token[:12]}"
    before_control = table_names(oracle_engine)
    with marker_table(oracle_engine, collision) as marker:
        baseline = table_names(oracle_engine)
        # Replace only this test module's uuid reference, not global uuid.uuid4.
        monkeypatch.setattr(module, "uuid", SimpleNamespace(uuid4=lambda: SimpleNamespace(hex=token)))
        generator = fixture_generator(variant, oracle_engine)
        try:
            with pytest.raises(RuntimeError, match="Refusing to reuse existing test tables"):
                await advance(generator, variant)
        finally:
            await finish(generator, variant)
        assert table_names(oracle_engine) == baseline
        assert_marker(oracle_engine, collision, marker)
    assert table_names(oracle_engine) == before_control
