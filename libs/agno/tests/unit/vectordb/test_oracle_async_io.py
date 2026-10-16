"""OracleVector async ingestion tests without a live database."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import agno.vectordb.oracle.oracle as oracle_module
from agno.knowledge.document import Document
from agno.vectordb.oracle import OracleVector
from agno.exceptions import EmbeddingError


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
