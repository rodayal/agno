"""Oracle regression tests grouped by behavior; no cases are discarded."""

import asyncio
import json
import threading
from copy import deepcopy
from decimal import Decimal
from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import MagicMock

import agno.vectordb.oracle.oracle as oracle_module
import pytest
from agno.exceptions import EmbeddingError
from agno.knowledge.document import Document
from agno.vectordb.oracle import OracleVector
from agno.vectordb.oracle.oracle import OracleNativeJSON

# Asynchronous ingestion


@pytest.fixture
def vector_store(monkeypatch):
    loop_thread = threading.get_ident()
    events = []

    def worker_event(name):
        assert threading.get_ident() != loop_thread, f"{name} ran on the event-loop thread"
        events.append(name)

    class TestEmbedder:
        enable_batch = False

        async def async_get_embedding_and_usage(self, text):
            assert threading.get_ident() == loop_thread
            events.append("embedding")
            return [1.0, 0.0], {"total_tokens": 1}

        def get_embedding_and_usage(self, text):
            raise AssertionError("Unexpected synchronous re-embedding")

    # Bypass construction: no engine or database connection is needed.
    store = OracleVector.__new__(OracleVector)
    store.embedder = TestEmbedder()
    store.table = SimpleNamespace(insert=lambda: "fake insert")

    def owner_column_exists():
        worker_event("metadata")
        return True

    monkeypatch.setattr(store, "_user_id_column_exists", owner_column_exists)

    session = MagicMock()
    session.__enter__.return_value = session

    def open_session():
        worker_event("session")
        return session

    store.Session = MagicMock(side_effect=open_session, return_value=session)
    store.Session.remove.side_effect = lambda: worker_event("cleanup")

    def check_records(records):
        assert len(records) == 1
        assert records[0]["embedding"] == [1.0, 0.0]
        assert records[0]["user_id"] == "alice"

    def execute(statement, records):
        worker_event("insert")
        check_records(records)

    session.execute.side_effect = execute

    def content_exists(content_hash, user_id=None):
        worker_event("exists")
        assert user_id == "alice"
        return True

    def delete_content(content_hash, user_id=None):
        worker_event("delete")
        assert user_id == "alice"

    monkeypatch.setattr(store, "content_hash_exists", content_exists)
    monkeypatch.setattr(store, "_delete_by_content_hash", delete_content)

    def merge(sess, table, keys, records, preserve_on_conflict):
        worker_event("merge")
        assert sess is session
        assert keys == ["id"]
        assert preserve_on_conflict == ["created_at"]
        check_records(records)

    monkeypatch.setattr(oracle_module, "merge_upsert_many", merge)
    return store, events


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["async_insert", "async_upsert"])
async def test_async_ingestion_moves_database_work_to_worker(vector_store, operation):
    store, events = vector_store

    await getattr(store, operation)(
        content_hash="test-hash",
        documents=[Document(name="test-doc", content="Test document")],
        user_id="alice",
    )

    assert events[0] == "embedding"
    assert events.count("embedding") == 1
    assert "metadata" in events
    assert "session" in events
    assert events[-1] == "cleanup"

    if operation == "async_insert":
        assert events.count("insert") == 1
        assert "delete" not in events
    else:
        assert events.count("merge") == 1
        assert events.index("embedding") < events.index("delete")
        assert events.index("exists") < events.index("delete")
        assert events.index("delete") < events.index("merge")


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["async_insert", "async_upsert"])
async def test_embedding_failure_prevents_database_work(vector_store, monkeypatch, operation):
    store, events = vector_store

    async def fail_embedding(text):
        raise RuntimeError("Embedding failed deliberately")

    monkeypatch.setattr(
        store.embedder,
        "async_get_embedding_and_usage",
        fail_embedding,
    )

    with pytest.raises(RuntimeError, match="Embedding failed deliberately"):
        await getattr(store, operation)(
            content_hash="test-hash",
            documents=[Document(content="Test document")],
            user_id="alice",
        )

    assert events == []
    store.Session.assert_not_called()
    store.Session.remove.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["async_insert", "async_upsert"])
async def test_event_loop_progresses_during_database_work(vector_store, monkeypatch, operation):
    store, events = vector_store
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    release_worker = threading.Event()

    def heartbeat():
        assert threading.get_ident() == loop_thread
        events.append("heartbeat")
        release_worker.set()

    if operation == "async_insert":
        original_write = store.Session.return_value.execute.side_effect
    else:
        original_write = oracle_module.merge_upsert_many

    def waiting_write(*args, **kwargs):
        original_write(*args, **kwargs)
        loop.call_soon_threadsafe(heartbeat)
        assert release_worker.wait(timeout=5), "The event loop could not progress during database work"
        events.append("worker-resumed")

    if operation == "async_insert":
        store.Session.return_value.execute.side_effect = waiting_write
    else:
        monkeypatch.setattr(oracle_module, "merge_upsert_many", waiting_write)

    await getattr(store, operation)(
        content_hash="test-hash",
        documents=[Document(content="Test document")],
        user_id="alice",
    )

    write_event = "insert" if operation == "async_insert" else "merge"
    assert events.index(write_event) < events.index("heartbeat")
    assert events.index("heartbeat") < events.index("worker-resumed")
    assert events[-1] == "cleanup"


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["async_insert", "async_upsert"])
async def test_database_error_propagates_and_cleans_up(vector_store, monkeypatch, operation):
    store, events = vector_store

    if operation == "async_insert":
        original_write = store.Session.return_value.execute.side_effect
    else:
        original_write = oracle_module.merge_upsert_many

    def failing_write(*args, **kwargs):
        original_write(*args, **kwargs)
        raise RuntimeError("Database write failed deliberately")

    if operation == "async_insert":
        store.Session.return_value.execute.side_effect = failing_write
    else:
        monkeypatch.setattr(oracle_module, "merge_upsert_many", failing_write)

    with pytest.raises(RuntimeError, match="Database write failed deliberately"):
        await getattr(store, operation)(
            content_hash="test-hash",
            documents=[Document(content="Test document")],
            user_id="alice",
        )

    assert events[-1] == "cleanup"
    store.Session.remove.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["async_insert", "async_upsert"])
@pytest.mark.parametrize("batch_size", [0, -1])
async def test_invalid_batch_size_prevents_work(vector_store, operation, batch_size):
    store, events = vector_store

    with pytest.raises(ValueError, match="batch_size must be greater than zero"):
        await getattr(store, operation)(
            content_hash="test-hash",
            documents=[Document(content="Test document")],
            batch_size=batch_size,
            user_id="alice",
        )

    assert events == []
    store.Session.assert_not_called()


@pytest.mark.asyncio
async def test_batch_rate_limit_logs_correct_message(monkeypatch):
    error = EmbeddingError("Test rate limit", status_code=429)

    async def fail_batch(_texts):
        raise error

    store = OracleVector.__new__(OracleVector)
    store.embedder = SimpleNamespace(
        enable_batch=True,
        async_get_embeddings_batch_and_usage=fail_batch,
    )

    messages = []
    monkeypatch.setattr(oracle_module, "log_error", messages.append)

    with pytest.raises(EmbeddingError) as caught:
        await store._async_embed_documents([Document(content="Test document")])

    assert caught.value is error
    assert messages == [f"Rate limit detected during batch embedding: {error}"]


# Native JSON conversion


def assert_same_json_types(actual, expected):
    """Numeric equality alone would incorrectly accept Decimal as int/float."""
    assert type(actual) is type(expected)
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_same_json_types(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for item, expected_item in zip(actual, expected):
            assert_same_json_types(item, expected_item)
    else:
        assert actual == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        (True, True),
        (False, False),
        ("hello", "hello"),
        ("null", "null"),
        ("123", "123"),
        ('{"count": 2}', '{"count": 2}'),
        (Decimal("2"), 2),
        (Decimal("1.25"), 1.25),
        (Decimal("9007199254740993"), 9007199254740993),
        (
            {"count": Decimal("2"), "nested": {"prices": [Decimal("1.25"), None, True]}},
            {"count": 2, "nested": {"prices": [1.25, None, True]}},
        ),
        ([Decimal("2"), {"price": Decimal("1.25")}, "123", 3, 0.5], [2, {"price": 1.25}, "123", 3, 0.5]),
    ],
)
def test_vector_json_normalizes_numbers_and_preserves_scalar_strings(raw, expected):
    converter = OracleNativeJSON()
    decoded = converter.process_result_value(raw, None)
    assert_same_json_types(decoded, expected)

    encoded = converter.process_bind_param(decoded, None)
    if expected is None:
        assert encoded is None
    else:
        assert_same_json_types(json.loads(encoded), expected)


def test_vector_json_normalization_does_not_mutate_driver_value():
    raw = {"nested": [{"count": Decimal("2"), "price": Decimal("1.25")}], "label": "123"}
    original = deepcopy(raw)

    decoded = OracleNativeJSON().process_result_value(raw, None)

    assert_same_json_types(raw, original)
    assert_same_json_types(decoded, {"nested": [{"count": 2, "price": 1.25}], "label": "123"})


# Document identity


@pytest.fixture
def store():
    # This helper needs no engine, table, embedder, or live connection.
    return OracleVector.__new__(OracleVector)


@pytest.mark.parametrize("owner", [None, "", "alice", "bob"])
@pytest.mark.parametrize(
    "left,right",
    [
        (("a_b", "c"), ("a", "b_c")),
        (("_", ""), ("", "_")),
        (("a__b", "c"), ("a", "_b_c")),
        (("é_資料", "🙂"), ("é", "資料_🙂")),
    ],
)
def test_base_id_and_content_hash_boundaries_do_not_collide(store, owner, left, right):
    assert store._scoped_record_id(*left, owner) != store._scoped_record_id(*right, owner)


@pytest.mark.parametrize(
    "base_id,content_hash,owner",
    [
        ("document", "hash", None),
        ("document", "hash", ""),
        ("document", "hash", "alice"),
        ("", "", None),
        ("_", "_", ""),
        ('doc"\\[]', 'hash"_\\', 'owner"\\'),
        ("é_資料", "🙂", "用户"),
        ("null", "[]", "null"),
    ],
)
def test_record_ids_remain_deterministic_64_character_hex(store, base_id, content_hash, owner):
    first = store._scoped_record_id(base_id, content_hash, owner)
    second = store._scoped_record_id(base_id, content_hash, owner)
    assert first == second
    assert len(first) == 64
    assert set(first) <= set("0123456789abcdef")


@pytest.mark.parametrize("owner", [None, "", "alice", "bob"])
def test_record_identity_uses_sha256_in_both_stages(store, owner):
    identity = json.dumps(["chunk_1", "content_hash"], separators=(",", ":"))
    first_stage = sha256(identity.encode("utf-8")).hexdigest()
    expected = first_stage if owner is None else sha256(f"{first_stage}_{owner}".encode()).hexdigest()
    assert store._scoped_record_id("chunk_1", "content_hash", owner) == expected


@pytest.mark.parametrize("document_id", [None, "", "caller-supplied-id"])
@pytest.mark.parametrize("owner", [None, "", "alice"])
def test_content_derived_base_id_uses_sha256_and_supplied_ids_are_preserved(store, monkeypatch, document_id, owner):
    document = SimpleNamespace(
        id=document_id,
        content="Hello\x00資料",
        embedding=[1.0, 0.0, 0.0],
        name="report",
        meta_data={},
        usage={},
        content_id="source-1",
    )
    captured = []

    def record_id(base_id, content_hash, user_id):
        captured.append((base_id, content_hash, user_id))
        return "prepared-record-id"

    monkeypatch.setattr(store, "_scoped_record_id", record_id)
    monkeypatch.setattr(store, "_user_id_column_exists", lambda: False)
    record = store._get_document_record(document, content_hash="upload-1", user_id=owner, prepared=True)
    cleaned = store._clean_content(document.content)
    expected_base_id = document_id or sha256(cleaned.encode()).hexdigest()
    assert captured == [(expected_base_id, "upload-1", owner)]
    assert record["content"] == cleaned
    assert record["id"] == "prepared-record-id"


def test_shared_empty_and_named_owners_have_separate_ids(store):
    ids = [store._scoped_record_id("report-chunk-1", "H123", owner) for owner in (None, "", "alice", "bob")]
    assert len(set(ids)) == 4


@pytest.mark.parametrize(
    "left,right",
    [
        (("doc_1", "hash", "alice"), ("doc", "hash", "1_alice")),
        (("doc", "hash_alice", "bob"), ("doc", "hash", "alice_bob")),
    ],
)
def test_owner_boundaries_remain_separate(store, left, right):
    assert store._scoped_record_id(*left) != store._scoped_record_id(*right)


def test_base_id_and_content_hash_order_matters(store):
    assert store._scoped_record_id("document", "upload", "alice") != store._scoped_record_id(
        "upload", "document", "alice"
    )


def test_different_content_hashes_keep_separate_records(store):
    assert store._scoped_record_id("document", "hash-1", "alice") != store._scoped_record_id(
        "document", "hash-2", "alice"
    )


def test_different_base_ids_keep_separate_records(store):
    assert store._scoped_record_id("document-1", "hash", "alice") != store._scoped_record_id(
        "document-2", "hash", "alice"
    )
