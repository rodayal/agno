"""Live JSON reopening tests using isolated memory and version tables."""

import asyncio
import os
import uuid

import pytest
from sqlalchemy import MetaData, Table, inspect, select, text
from sqlalchemy.engine import make_url

from agno.db.oracle import AsyncOracleDb, OracleDb
from agno.db.oracle._version import detect_capabilities
from agno.db.oracle.engine import create_async_oracle_engine, create_oracle_engine
from agno.db.oracle.schemas import OracleClobJSON, OracleNativeJSON


def assert_json_value(actual, expected):
    assert type(actual) is type(expected), (actual, expected)

    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_json_value(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            assert_json_value(actual_item, expected_item)
    else:
        assert actual == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("storage", ["clob", "native"])
@pytest.mark.parametrize("explicit_schema", [False, True], ids=["default-schema", "explicit-schema"])
async def test_json_reads_and_writes_after_reopening(mode, storage, explicit_schema):
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
    memory_name = f"agno_jr_m_{suffix}"
    version_name = f"agno_jr_v_{suffix}"
    names = (memory_name, version_name)
    plain_text = "Meeting at 10 — this is ordinary text, not JSON."

    def preflight():
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

        if storage == "native" and not detect_capabilities(engine).native_json:
            pytest.skip("Native JSON is unavailable on this Oracle server")

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
        owner = await asyncio.to_thread(preflight)
        cleanup_allowed = True

        settings = {
            "db_schema": owner if explicit_schema else None,
            "memory_table": memory_name,
            "versions_table": version_name,
            "create_schema": False,
        }

        if mode == "async":
            async_engine = create_async_oracle_engine(url.set(drivername="oracle+oracledb_async"))

        async def open_table(override, create=False):
            if mode == "async":
                db = AsyncOracleDb(db_engine=async_engine, json_storage=override, **settings)
                return await db._get_table(table_type="memories", create_table_if_not_found=create)

            def open_sync():
                db = OracleDb(db_engine=engine, json_storage=override, **settings)
                try:
                    return db._get_table(table_type="memories", create_table_if_not_found=create)
                finally:
                    db.Session.remove()

            return await asyncio.to_thread(open_sync)

        async def execute(statement, parameters=None, read=False):
            parameters = {} if parameters is None else parameters

            if mode == "async":
                async with async_engine.begin() as connection:
                    result = await connection.execute(statement, parameters)
                    return result.mappings().all() if read else None

            def execute_sync():
                with engine.begin() as connection:
                    result = connection.execute(statement, parameters)
                    return result.mappings().all() if read else None

            return await asyncio.to_thread(execute_sync)

        fresh = await open_table(storage, create=True)
        assert fresh is not None

        initial = {"message": "Before reopening", "count": 2, "tags": [1, None]}
        await execute(
            fresh.insert(),
            {
                "memory_id": "initial",
                "memory": initial,
                "input": plain_text,
                "topics": [1, None],
                "created_at": 1700000000,
            },
        )

        fresh_rows = await execute(select(fresh.c.memory), read=True)
        assert_json_value(fresh_rows[0]["memory"], initial)

        opposite_storage = "native" if storage == "clob" else "clob"
        reopened = await open_table(opposite_storage)
        assert reopened is not None

        expected_type = OracleClobJSON if storage == "clob" else OracleNativeJSON
        assert isinstance(reopened.c.memory.type, expected_type)
        assert isinstance(reopened.c.topics.type, expected_type)
        assert not isinstance(reopened.c.input.type, (OracleClobJSON, OracleNativeJSON))

        existing = await execute(select(reopened.c.memory), read=True)
        assert_json_value(existing[0]["memory"], initial)

        replacement = {"message": "After reopening", "count": 4, "enabled": True}
        await execute(
            reopened.update().where(reopened.c.memory_id == "initial").values(memory=replacement, topics=None)
        )

        values = [
            {"nested": {"count": 3, "ratio": 1.25, "enabled": True}, "tags": [1, None]},
            ["first", 2, False, None],
            "Remember this",
            "",
            7,
            1.25,
            True,
        ]

        records = []
        expected = {"initial": (replacement, None)}

        for number, value in enumerate(values):
            memory_id = f"value-{number}"
            topics = None if number == 0 else [number, None]
            records.append(
                {
                    "memory_id": memory_id,
                    "memory": value,
                    "input": plain_text,
                    "topics": topics,
                    "created_at": 1700000000,
                }
            )
            expected[memory_id] = (value, topics)

        await execute(reopened.insert(), records)

        rows = await execute(
            select(reopened.c.memory_id, reopened.c.memory, reopened.c.input, reopened.c.topics),
            read=True,
        )

        assert len(rows) == len(expected)
        assert {row["memory_id"] for row in rows} == set(expected)

        for row in rows:
            expected_memory, expected_topics = expected[row["memory_id"]]
            assert_json_value(row["memory"], expected_memory)
            assert_json_value(row["topics"], expected_topics)
            assert row["input"] == plain_text

        def check_physical_storage():
            with engine.connect() as connection:
                result = connection.execute(
                    text(
                        "SELECT data_type FROM user_tab_columns WHERE table_name = :table_name "
                        "AND UPPER(column_name) IN ('MEMORY', 'TOPICS')"
                    ),
                    {"table_name": memory_name.upper()},
                )
                physical_types = [row[0] for row in result.all()]

            expected_storage = "CLOB" if storage == "clob" else "JSON"
            assert len(physical_types) == 2
            assert all(value == expected_storage for value in physical_types)

        await asyncio.to_thread(check_physical_storage)
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
