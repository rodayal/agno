"""Test the differential fixtures' cleanup logic without live databases.

The fixture bodies are exercised directly with fake adapters and a fake table
catalog. No PostgreSQL or Oracle connection is opened. These tests establish
control-flow safety, not real Oracle DDL behavior or adapter parity.
"""

import inspect as python_inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from tests.integration.db import test_differential_async_oracle_postgres as async_module
from tests.integration.db import test_differential_oracle_postgres as sync_module


class FixtureHarness:
    def __init__(self, variant, monkeypatch):
        self.asynchronous = variant == "async"
        self.module = async_module if self.asynchronous else sync_module
        self.protected = {
            "agno_tool_results",
            "agno_schema_versions",
            "agno_spans",
            "unrelated_table",
        }
        self.catalog = set(self.protected)
        self.instances = []
        self.connections = []
        self.drop_attempts = []
        self.dropped = []
        self.reject_existing = False
        self.metadata_error = False
        self.cleanup_error = False

        constructor = "AsyncOracleDb" if self.asynchronous else "OracleDb"
        monkeypatch.setattr(self.module, constructor, self.make_database)
        monkeypatch.setattr(self.module, "inspect", self.inspect_connection)
        monkeypatch.setattr(
            self.module,
            "create_engine",
            Mock(side_effect=AssertionError("Unexpected real engine construction")),
        )

    def make_database(self, **kwargs):
        engine = MagicMock()
        connection = MagicMock()
        connection.__enter__.return_value = connection
        connection.__aenter__.return_value = connection
        engine.connect.return_value = connection
        engine.begin.return_value = connection
        connection.run_sync = AsyncMock(side_effect=lambda callback: callback(connection))
        executor = AsyncMock if self.asynchronous else Mock
        connection.execute = executor(side_effect=self.execute_drop)
        database = SimpleNamespace(
            db_engine=engine,
            Session=MagicMock(),
            close=AsyncMock(),
            tool_results_table_name="agno_tool_results",
            table_options={key: value for key, value in kwargs.items() if key.endswith("_table")},
        )
        self.instances.append(database)
        self.connections.append(connection)
        if self.reject_existing:
            self.catalog.add(kwargs["session_table"])
        return database

    def inspect_connection(self, connection):
        assert any(connection is known for known in self.connections)
        return SimpleNamespace(has_table=self.has_table)

    def has_table(self, name):
        if self.metadata_error:
            raise RuntimeError("Deliberate metadata failure")
        return name in self.catalog

    def execute_drop(self, statement):
        parts = str(statement).split()
        assert parts[:2] == ["DROP", "TABLE"]
        name = parts[2]
        assert name not in self.protected, f"Cleanup targeted protected table {name}"
        assert name in self.catalog
        self.drop_attempts.append(name)
        if self.cleanup_error:
            raise RuntimeError("Deliberate cleanup failure")
        self.catalog.remove(name)
        self.dropped.append(name)

    def owned_names(self, database):
        return {*database.table_options.values(), database.tool_results_table_name}

    async def start(self):
        # Bypass pytest's fixture invocation guard, not the fixture's own body.
        # The server-gating dependency is intentionally not requested: all
        # adapters, engines, connections and inspection calls are fakes.
        body = python_inspect.unwrap(self.module.oracle_db)
        if self.asynchronous:
            assert python_inspect.isasyncgenfunction(body), "Unexpected async fixture wrapper"
        else:
            assert python_inspect.isgeneratorfunction(body), "Unexpected sync fixture wrapper"
        generator = body(_servers_up=None)
        if self.asynchronous:
            database = await generator.__anext__()
        else:
            database = next(generator)
        return generator, database

    async def finish(self, generator):
        if self.asynchronous:
            await generator.aclose()
        else:
            generator.close()

    async def fail_body(self, generator):
        error = RuntimeError("Deliberate test-body failure")
        if self.asynchronous:
            await generator.athrow(error)
        else:
            generator.throw(error)

    def assert_closed(self, database):
        if self.asynchronous:
            database.close.assert_awaited_once()
        else:
            database.Session.remove.assert_called_once()
            database.db_engine.dispose.assert_called_once()


@pytest.fixture(params=["sync", "async"])
def harness(request, monkeypatch):
    return FixtureHarness(request.param, monkeypatch)


@pytest.mark.asyncio
async def test_cleanup_uses_captured_names_and_preserves_shared_tables(harness):
    generator, database = await harness.start()
    names = harness.owned_names(database)
    try:
        assert "spans_table" in database.table_options
        assert "versions_table" in database.table_options
        assert names.isdisjoint(harness.protected)
        assert database.tool_results_table_name.startswith(("diff_tool_", "adiff_tool_"))
        harness.catalog.update(names)
        # Even a later attribute change must not redirect the saved cleanup list.
        database.tool_results_table_name = "agno_tool_results"
    finally:
        await harness.finish(generator)
    assert set(harness.dropped) == names
    assert harness.catalog == harness.protected
    harness.assert_closed(database)


@pytest.mark.asyncio
async def test_preexisting_name_is_rejected_without_deleting_anything(harness):
    harness.reject_existing = True
    with pytest.raises(RuntimeError, match="Refusing to reuse existing test tables"):
        await harness.start()
    database = harness.instances[-1]
    collision = database.table_options["session_table"]
    assert harness.catalog == harness.protected | {collision}
    assert harness.drop_attempts == []
    harness.assert_closed(database)


@pytest.mark.asyncio
async def test_test_body_failure_still_cleans_up_owned_tables(harness):
    generator, database = await harness.start()
    names = harness.owned_names(database)
    harness.catalog.update(names)
    try:
        with pytest.raises(RuntimeError, match="Deliberate test-body failure"):
            await harness.fail_body(generator)
    finally:
        await harness.finish(generator)
    assert set(harness.dropped) == names
    assert harness.catalog == harness.protected
    harness.assert_closed(database)


@pytest.mark.asyncio
async def test_two_fixtures_do_not_remove_each_others_tables(harness):
    first_generator, first_database = await harness.start()
    second_generator = None
    first_names = harness.owned_names(first_database)
    harness.catalog.update(first_names)
    try:
        second_generator, second_database = await harness.start()
        second_names = harness.owned_names(second_database)
        assert first_names.isdisjoint(second_names)
        harness.catalog.update(second_names)
        await harness.finish(first_generator)
        assert harness.catalog == harness.protected | second_names
        harness.assert_closed(first_database)
    finally:
        await harness.finish(first_generator)
        if second_generator is not None:
            await harness.finish(second_generator)
    assert harness.catalog == harness.protected
    harness.assert_closed(second_database)


@pytest.mark.asyncio
async def test_failed_metadata_preflight_never_runs_destructive_cleanup(harness):
    harness.metadata_error = True
    with pytest.raises(RuntimeError, match="Deliberate metadata failure"):
        await harness.start()
    assert harness.catalog == harness.protected
    assert harness.drop_attempts == []
    harness.assert_closed(harness.instances[-1])


@pytest.mark.asyncio
async def test_cleanup_failure_still_closes_the_database(harness):
    generator, database = await harness.start()
    names = harness.owned_names(database)
    harness.catalog.update(names)
    harness.cleanup_error = True
    try:
        with pytest.raises(RuntimeError, match="Deliberate cleanup failure"):
            await harness.finish(generator)
    finally:
        await harness.finish(generator)
    assert harness.protected.issubset(harness.catalog)
    assert len(harness.drop_attempts) == 1
    assert set(harness.drop_attempts).issubset(names)
    harness.assert_closed(database)
