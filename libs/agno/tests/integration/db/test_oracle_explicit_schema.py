"""Live explicit-schema foreign-key tests using isolated temporary tables."""

import asyncio
import os
import uuid

import pytest
from sqlalchemy import MetaData, Table, inspect, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError

from agno.db.oracle import AsyncOracleDb, OracleDb
from agno.db.oracle.engine import create_async_oracle_engine, create_oracle_engine


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_explicit_schema_run_foreign_key(mode):
    raw_url = os.getenv("AGNO_TEST_ORACLE_URL", "")
    if not raw_url:
        pytest.skip("AGNO_TEST_ORACLE_URL is not set")

    url = make_url(raw_url)
    assert url.get_backend_name() == "oracle"

    engine = create_oracle_engine(url.set(drivername="oracle+oracledb"))
    async_engine = None
    cleanup_allowed = False
    owner = ""

    suffix = uuid.uuid4().hex[:12]
    session_name = f"agno_fk_s_{suffix}"
    run_name = f"agno_fk_r_{suffix}"
    version_name = f"agno_fk_v_{suffix}"
    names = (run_name, session_name, version_name)

    def check_destination():
        with engine.connect() as connection:
            identity = tuple(
                connection.execute(
                    text(
                        "SELECT SYS_CONTEXT('USERENV', 'SESSION_USER'), "
                        "SYS_CONTEXT('USERENV', 'CON_NAME'), "
                        "SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA') FROM dual"
                    )
                ).one()
            )

        login, container, current_schema = identity
        assert login.upper() not in {"SYS", "SYSTEM"}
        assert container.upper() != "CDB$ROOT"
        assert current_schema == login

        inspector = inspect(engine)
        for name in names:
            assert not inspector.has_table(name, schema=login), "Temporary name already exists"

        return login

    def cleanup():
        for name in names:
            Table(name, MetaData(), schema=owner).drop(engine, checkfirst=True)

        inspector = inspect(engine)
        assert not any(inspector.has_table(name, schema=owner) for name in names)

    try:
        owner = await asyncio.to_thread(check_destination)
        cleanup_allowed = True

        settings = {
            "db_schema": owner,
            "session_table": session_name,
            "runs_table": run_name,
            "versions_table": version_name,
            "create_schema": False,
        }

        if mode == "async":
            async_engine = create_async_oracle_engine(url.set(drivername="oracle+oracledb_async"))
            db = AsyncOracleDb(db_engine=async_engine, **settings)
            runs = await db._get_table(table_type="runs", create_table_if_not_found=True)
            sessions = await db._get_table(table_type="sessions", create_table_if_not_found=True)
        else:

            def prepare_sync():
                db = OracleDb(db_engine=engine, **settings)
                try:
                    runs = db._get_table(table_type="runs", create_table_if_not_found=True)
                    sessions = db._get_table(table_type="sessions", create_table_if_not_found=True)
                    return sessions, runs
                finally:
                    db.Session.remove()

            sessions, runs = await asyncio.to_thread(prepare_sync)

        assert sessions is not None
        assert runs is not None
        foreign_keys = list(runs.c.session_id.foreign_keys)
        assert len(foreign_keys) == 1
        assert foreign_keys[0].target_fullname == f"{owner}.{session_name}.session_id"

        def check_enforcement():
            with engine.begin() as connection:
                connection.execute(
                    sessions.insert(),
                    {
                        "session_id": "S1",
                        "session_type": "agent",
                        "created_at": 1700000000,
                    },
                )

            valid_run = {
                "run_id": "valid-run",
                "session_id": "S1",
                "run_type": "agent",
                "run_data": {"message": "Valid session reference"},
                "created_at": 1700000000,
            }

            with engine.begin() as connection:
                connection.execute(runs.insert(), valid_run)

            invalid_run = {
                **valid_run,
                "run_id": "invalid-run",
                "session_id": "S99",
            }

            with pytest.raises(IntegrityError) as failure:
                with engine.begin() as connection:
                    connection.execute(runs.insert(), invalid_run)

            detail = failure.value.orig.args[0]
            assert getattr(detail, "code", None) == 2291

            with engine.connect() as connection:
                saved_runs = connection.execute(select(runs.c.run_id)).scalars().all()

            assert saved_runs == ["valid-run"]

        await asyncio.to_thread(check_enforcement)
    finally:
        try:
            if cleanup_allowed:
                await asyncio.to_thread(cleanup)
        finally:
            try:
                if async_engine is not None:
                    await async_engine.dispose()
            finally:
                await asyncio.to_thread(engine.dispose)
