"""Offline Oracle filesystem retry checks, plus in-memory SQLite regressions."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import agno.fs.db as fs_module
import pytest
from agno.db.oracle.utils import DEFAULT_MERGE_RETRY_ATTEMPTS
from agno.fs.db import DbFileSystem
from agno.fs.errors import VersionConflictError
from agno.fs.types import FileMeta
from sqlalchemy import BigInteger, Column, MetaData, String, Table, Text, create_engine
from sqlalchemy.dialects import oracle
from sqlalchemy.exc import IntegrityError, OperationalError, ProgrammingError


def oracle_integrity_error(code):
    return IntegrityError("unit statement", {}, Exception(SimpleNamespace(code=code)))


@pytest.fixture
def backend_case(monkeypatch):
    backend = DbFileSystem.__new__(DbFileSystem)
    backend.dialect = "oracle"
    backend.table = Table(
        "unit_files",
        MetaData(),
        Column("namespace", String(255), primary_key=True),
        Column("path", String(2000), primary_key=True),
        Column("content", Text),
        Column("size_bytes", BigInteger),
        Column("version", BigInteger),
        Column("created_at", BigInteger),
        Column("updated_at", BigInteger),
    )
    backend._ensure_table = MagicMock()
    backend.db_engine = MagicMock()
    connection = MagicMock()
    connection.dialect = oracle.dialect()
    backend.db_engine.begin.return_value.__enter__.return_value = connection
    monkeypatch.setattr(fs_module, "time", SimpleNamespace(time=lambda: 1700000000))
    return SimpleNamespace(backend=backend, connection=connection)


def readback(version, content):
    return SimpleNamespace(one=lambda: (version, len(content.encode("utf-8"))))


def assert_no_transaction_management(case):
    case.connection.commit.assert_not_called()
    case.connection.rollback.assert_not_called()
    case.connection.close.assert_not_called()
    case.connection.begin.assert_not_called()
    case.connection.begin_nested.assert_not_called()
    case.backend.db_engine.begin.assert_not_called()
    case.backend._ensure_table.assert_not_called()


@pytest.mark.parametrize("content", ["hello", "", "café🙂"])
def test_success_reads_metadata_once_and_preserves_bound_values(backend_case, content):
    case = backend_case
    case.connection.execute.side_effect = [object(), readback(1, content)]
    meta = case.backend._write_on(case.connection, "shared-work", "report.txt", content)
    assert meta == FileMeta(
        path="report.txt", version=1, size_bytes=len(content.encode("utf-8")), updated_at=1700000000
    )
    calls = case.connection.execute.call_args_list
    assert len(calls) == 2
    assert str(calls[0].args[0]).startswith("MERGE INTO unit_files")
    assert "d.version = d.version + 1" in str(calls[0].args[0])
    assert calls[0].args[1] == {
        "ns": "shared-work",
        "p": "report.txt",
        "content": content,
        "size_bytes": len(content.encode("utf-8")),
        "now": 1700000000,
    }
    assert calls[1].args[0].is_select
    assert_no_transaction_management(case)


def test_unique_conflict_retries_same_statement_on_same_connection(backend_case):
    case = backend_case
    case.connection.execute.side_effect = [oracle_integrity_error(1), object(), readback(2, "Draft B")]
    meta = case.backend._write_on(case.connection, "shared-work", "report.txt", "Draft B")
    assert meta.version == 2
    assert meta.size_bytes == 7
    calls = case.connection.execute.call_args_list
    assert len(calls) == 3
    assert calls[0].args[0] is calls[1].args[0]
    assert calls[0].args[1] is calls[1].args[1]
    assert calls[2].args[0].is_select
    assert_no_transaction_management(case)


def test_final_allowed_attempt_can_succeed(backend_case):
    case = backend_case
    errors = [oracle_integrity_error(1) for _ in range(DEFAULT_MERGE_RETRY_ATTEMPTS - 1)]
    case.connection.execute.side_effect = [*errors, object(), readback(2, "Draft B")]
    assert case.backend._write_on(case.connection, "shared-work", "report.txt", "Draft B").version == 2
    assert case.connection.execute.call_count == DEFAULT_MERGE_RETRY_ATTEMPTS + 1
    assert_no_transaction_management(case)


def test_exhausted_retries_raise_original_error_without_readback(backend_case):
    case = backend_case
    error = oracle_integrity_error(1)
    case.connection.execute.side_effect = error
    with pytest.raises(IntegrityError) as caught:
        case.backend._write_on(case.connection, "shared-work", "report.txt", "Draft B")
    assert caught.value is error
    assert case.connection.execute.call_count == DEFAULT_MERGE_RETRY_ATTEMPTS
    assert all(str(call.args[0]).startswith("MERGE INTO") for call in case.connection.execute.call_args_list)
    assert_no_transaction_management(case)


@pytest.mark.parametrize("code", [1400, 2290, 2291])
def test_other_integrity_errors_are_not_retried(backend_case, code):
    case = backend_case
    error = oracle_integrity_error(code)
    case.connection.execute.side_effect = error
    with pytest.raises(IntegrityError) as caught:
        case.backend._write_on(case.connection, "shared-work", "report.txt", "Draft B")
    assert caught.value is error
    assert case.connection.execute.call_count == 1
    assert_no_transaction_management(case)


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("Deliberate application failure"),
        OperationalError("unit", {}, Exception(SimpleNamespace(code=12541))),
        ProgrammingError("unit", {}, Exception(SimpleNamespace(code=942))),
    ],
    ids=["application", "connection", "invalid-sql"],
)
def test_non_integrity_errors_are_not_retried(backend_case, error):
    case = backend_case
    case.connection.execute.side_effect = error
    with pytest.raises(type(error)) as caught:
        case.backend._write_on(case.connection, "shared-work", "report.txt", "Draft B")
    assert caught.value is error
    assert case.connection.execute.call_count == 1
    assert_no_transaction_management(case)


def test_metadata_read_error_does_not_repeat_the_write(backend_case):
    case = backend_case
    error = oracle_integrity_error(1)
    case.connection.execute.side_effect = [object(), error]
    with pytest.raises(IntegrityError) as caught:
        case.backend._write_on(case.connection, "shared-work", "report.txt", "Draft B")
    assert caught.value is error
    assert case.connection.execute.call_count == 2
    assert case.connection.execute.call_args_list[1].args[0].is_select
    assert_no_transaction_management(case)


def test_public_oracle_write_delegates_inside_its_transaction(backend_case):
    case = backend_case
    expected = FileMeta(path="report.txt", version=2, size_bytes=7, updated_at=1700000000)
    case.backend._write_on = MagicMock(return_value=expected)
    assert case.backend.write("shared-work", "report.txt", "Draft B") is expected
    case.backend._ensure_table.assert_called_once()
    case.backend.db_engine.begin.assert_called_once()
    case.backend._write_on.assert_called_once_with(case.connection, "shared-work", "report.txt", "Draft B")
    case.connection.execute.assert_not_called()


@pytest.mark.parametrize("stale", [False, True], ids=["matching-version", "stale-version"])
def test_conditional_write_does_not_use_unconditional_retry(backend_case, stale):
    case = backend_case
    case.backend._write_on = MagicMock(side_effect=AssertionError("Conditional write was delegated"))
    if stale:
        case.connection.execute.side_effect = [SimpleNamespace(first=lambda: None), SimpleNamespace(scalar=lambda: 2)]
        with pytest.raises(VersionConflictError) as caught:
            case.backend.write("shared-work", "report.txt", "Draft B", expected_version=1)
        assert caught.value.expected == 1 and caught.value.actual == 2
    else:
        case.connection.execute.return_value.first.return_value = (2, 7)
        meta = case.backend.write("shared-work", "report.txt", "Draft B", expected_version=1)
        assert meta.version == 2 and meta.size_bytes == 7
    case.backend._write_on.assert_not_called()
    assert case.connection.execute.call_args_list[0].args[0].is_update


@pytest.mark.parametrize("commit", [False, True], ids=["rollback", "commit"])
def test_sqlite_caller_owned_transaction_is_unchanged(commit):
    import sqlite3

    if sqlite3.sqlite_version_info < (3, 35, 0):
        pytest.skip(f"SQLite >= 3.35.0 required; found {sqlite3.sqlite_version}")
    engine = create_engine("sqlite:///:memory:")
    backend = DbFileSystem(db_engine=engine)
    try:
        backend._ensure_table()
        with engine.connect() as connection:
            transaction = connection.begin()
            backend._write_on(connection, "unit", "marker.txt", "earlier work")
            meta = backend._write_on(connection, "unit", "report.txt", "Draft B")
            assert meta.version == 1
            assert transaction.is_active
            if commit:
                transaction.commit()
            else:
                transaction.rollback()
        assert backend.read("unit", "marker.txt") == ("earlier work" if commit else None)
        assert backend.read("unit", "report.txt") == ("Draft B" if commit else None)
    finally:
        engine.dispose()
