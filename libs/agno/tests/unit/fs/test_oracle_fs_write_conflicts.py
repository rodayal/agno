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
        Column("user_id", String(257), primary_key=True),
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


def update_result(version=None):
    return SimpleNamespace(first=lambda: None if version is None else (version,))


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
    case.connection.execute.side_effect = [update_result(), object(), readback(1, content)]
    meta = case.backend._write_on(case.connection, "shared-work", "report.txt", content)
    assert meta == FileMeta(
        path="report.txt", version=1, size_bytes=len(content.encode("utf-8")), updated_at=1700000000
    )
    calls = case.connection.execute.call_args_list
    assert len(calls) == 3
    assert calls[0].args[0].is_update
    assert calls[1].args[0].is_insert
    assert "version=(unit_files.version +" in str(calls[0].args[0])
    assert calls[1].args[0].compile().params == {
        "namespace": "shared-work",
        "user_id": "",
        "path": "report.txt",
        "content": content,
        "size_bytes": len(content.encode("utf-8")),
        "version": 1,
        "created_at": 1700000000,
        "updated_at": 1700000000,
    }
    assert calls[2].args[0].is_select
    assert_no_transaction_management(case)


@pytest.mark.parametrize(
    "content",
    ["", "x" * 22702, "x" * 32767, "x" * 32768, "x" * 68115, "café🙂" * 7000],
    ids=["empty", "22kb", "32767", "32768", "66kb", "unicode"],
)
def test_content_writes_keep_column_type_and_do_not_use_raw_sql(backend_case, content):
    case = backend_case

    def execute(statement, *args):
        if getattr(statement, "is_select", False):
            return readback(1, content)
        return update_result() if getattr(statement, "is_update", False) else object()

    case.connection.execute.side_effect = execute
    case.backend._write_on(case.connection, "unit", "large.txt", content)
    for call in case.connection.execute.call_args_list[:2]:
        compiled = call.args[0].compile(dialect=oracle.dialect())
        assert not compiled.isplaintext
        assert isinstance(compiled.binds["content"].type, Text)
        assert compiled.params["content"] == content
        assert content not in str(compiled) or not content
    assert_no_transaction_management(case)


def test_existing_file_updates_without_attempting_insert(backend_case):
    case = backend_case
    case.connection.execute.side_effect = [update_result(2), readback(2, "Draft B")]
    meta = case.backend._write_on(case.connection, "unit", "report.txt", "Draft B")
    assert meta.version == 2
    assert [call.args[0].is_update for call in case.connection.execute.call_args_list] == [True, False]
    assert_no_transaction_management(case)


def test_unique_conflict_retries_same_statement_on_same_connection(backend_case):
    case = backend_case
    case.connection.execute.side_effect = [
        update_result(),
        oracle_integrity_error(1),
        update_result(2),
        readback(2, "Draft B"),
    ]
    meta = case.backend._write_on(case.connection, "shared-work", "report.txt", "Draft B")
    assert meta.version == 2
    assert meta.size_bytes == 7
    calls = case.connection.execute.call_args_list
    assert len(calls) == 4
    assert calls[0].args[0] is calls[2].args[0]
    assert calls[1].args[0].is_insert
    assert calls[3].args[0].is_select
    assert_no_transaction_management(case)


def test_final_allowed_attempt_can_succeed(backend_case):
    case = backend_case
    attempts = [
        item for _ in range(DEFAULT_MERGE_RETRY_ATTEMPTS - 1) for item in (update_result(), oracle_integrity_error(1))
    ]
    case.connection.execute.side_effect = [*attempts, update_result(), object(), readback(1, "Draft B")]
    assert case.backend._write_on(case.connection, "shared-work", "report.txt", "Draft B").version == 1
    assert case.connection.execute.call_count == DEFAULT_MERGE_RETRY_ATTEMPTS * 2 + 1
    assert_no_transaction_management(case)


def test_exhausted_retries_raise_original_error_without_readback(backend_case):
    case = backend_case
    error = oracle_integrity_error(1)
    case.connection.execute.side_effect = [
        item for _ in range(DEFAULT_MERGE_RETRY_ATTEMPTS) for item in (update_result(), error)
    ]
    with pytest.raises(IntegrityError) as caught:
        case.backend._write_on(case.connection, "shared-work", "report.txt", "Draft B")
    assert caught.value is error
    assert case.connection.execute.call_count == DEFAULT_MERGE_RETRY_ATTEMPTS * 2
    assert not any(call.args[0].is_select for call in case.connection.execute.call_args_list)
    assert_no_transaction_management(case)


@pytest.mark.parametrize("code", [1400, 2290, 2291])
def test_other_integrity_errors_are_not_retried(backend_case, code):
    case = backend_case
    error = oracle_integrity_error(code)
    case.connection.execute.side_effect = [update_result(), error]
    with pytest.raises(IntegrityError) as caught:
        case.backend._write_on(case.connection, "shared-work", "report.txt", "Draft B")
    assert caught.value is error
    assert case.connection.execute.call_count == 2
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
@pytest.mark.parametrize("stage", ["update", "insert"])
def test_non_integrity_errors_are_not_retried(backend_case, error, stage):
    case = backend_case
    case.connection.execute.side_effect = error if stage == "update" else [update_result(), error]
    with pytest.raises(type(error)) as caught:
        case.backend._write_on(case.connection, "shared-work", "report.txt", "Draft B")
    assert caught.value is error
    assert case.connection.execute.call_count == (1 if stage == "update" else 2)
    assert_no_transaction_management(case)


def test_metadata_read_error_does_not_repeat_the_write(backend_case):
    case = backend_case
    error = oracle_integrity_error(1)
    case.connection.execute.side_effect = [update_result(2), error]
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
    case.backend._write_on.assert_called_once_with(case.connection, "shared-work", "report.txt", "Draft B", user_id="")
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


@pytest.mark.parametrize("user_id", ["", "alice", "bob", "s:", "u:alice", "__agno_unowned__", "é🙂"])
def test_oracle_partition_encoding_round_trips_without_reserved_usernames(user_id):
    column_type = fs_module._OracleFileUserId()
    dialect = oracle.dialect()
    stored = column_type.process_bind_param(user_id, dialect)
    assert stored
    assert column_type.process_result_value(stored, dialect) == user_id
    if user_id:
        assert stored != column_type.process_bind_param("", dialect)


@pytest.mark.parametrize("user_id", ["", "alice", "bob"])
def test_write_and_metadata_read_are_scoped_to_the_user(backend_case, user_id):
    case = backend_case
    case.connection.execute.side_effect = [update_result(), object(), readback(1, "content")]
    meta = case.backend._write_on(case.connection, "unit", "report.txt", "content", user_id=user_id)
    assert meta.user_id == (user_id or None)
    for call in case.connection.execute.call_args_list:
        compiled = call.args[0].compile()
        assert user_id in compiled.params.values()
        assert "user_id" in str(compiled)
    assert_no_transaction_management(case)
