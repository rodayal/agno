"""Reserved Oracle owner IDs: offline correctness tests, no connections."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import Column, MetaData, String, Table, select

from agno.db.oracle import utils as storage
from agno.knowledge.document import Document
from agno.vectordb.oracle import oracle as vector


@pytest.fixture(params=["storage", "vector"])
def owner_codec(request):
    if request.param == "storage":
        return SimpleNamespace(
            encode=storage.to_db_user_id,
            decode=storage.from_db_user_id,
            marker=storage.UNOWNED_USER_ID,
            other_marker=vector._UNOWNED_SENTINEL,
        )
    return SimpleNamespace(
        encode=vector._to_db_user_id,
        decode=vector._from_db_user_id,
        marker=vector._UNOWNED_SENTINEL,
        other_marker=storage.UNOWNED_USER_ID,
    )


@pytest.mark.parametrize("owner", [None, "", "alice", "~alice", "__agno_unowned___", "é" * 128])
def test_valid_owners_keep_the_existing_representation_and_round_trip(owner_codec, owner):
    encoded = owner_codec.encode(owner)
    expected = owner_codec.marker if owner == "" else owner
    assert encoded == expected
    assert owner_codec.decode(encoded) == owner


def test_only_this_adapters_own_marker_is_reserved(owner_codec):
    assert owner_codec.encode(owner_codec.other_marker) == owner_codec.other_marker
    assert owner_codec.decode(owner_codec.other_marker) == owner_codec.other_marker


def test_literal_marker_is_rejected_as_caller_input(owner_codec):
    with pytest.raises(ValueError, match="reserved"):
        owner_codec.encode(owner_codec.marker)


def test_stored_marker_still_decodes_to_a_legitimate_empty_owner(owner_codec):
    assert owner_codec.decode(owner_codec.marker) == ""
    assert owner_codec.encode("") == owner_codec.marker


def test_none_empty_and_named_owners_have_distinct_stored_values(owner_codec):
    assert len({owner_codec.encode(owner) for owner in (None, "", "alice")}) == 3


@pytest.fixture
def vector_store(monkeypatch):
    database = vector.OracleVector.__new__(vector.OracleVector)
    database.table = Table(
        "offline_vectors",
        MetaData(),
        Column("id", String),
        Column("user_id", String),
        Column("content_hash", String),
    )
    database._user_id_column_exists = MagicMock(return_value=True)
    database.Session = MagicMock(side_effect=AssertionError("Unexpected database session"))
    database.embedder = MagicMock()
    embed = MagicMock(side_effect=AssertionError("Embedding must not precede owner validation"))
    aembed = AsyncMock(side_effect=AssertionError("Embedding must not precede owner validation"))
    monkeypatch.setattr(vector, "embed_before_replace", embed)
    monkeypatch.setattr(vector, "aembed_before_replace", aembed)
    database._async_embed_documents = AsyncMock(side_effect=AssertionError("Unexpected embedding"))
    return SimpleNamespace(db=database, embed=embed, aembed=aembed)


def assert_no_database_or_embedding(store):
    store.db._user_id_column_exists.assert_not_called()
    store.db.Session.assert_not_called()
    store.db.embedder.get_embedding_and_usage.assert_not_called()
    store.db._async_embed_documents.assert_not_called()
    store.embed.assert_not_called()
    store.aembed.assert_not_called()


def test_vector_guard_rejects_before_metadata_inspection(vector_store):
    with pytest.raises(ValueError, match="reserved"):
        vector_store.db._require_owner_column(vector._UNOWNED_SENTINEL)
    assert_no_database_or_embedding(vector_store)


@pytest.mark.parametrize(
    "method", ["insert", "upsert", "search", "content_hash_exists", "delete_by_content_id", "_delete_by_content_hash"]
)
def test_sync_vector_entry_points_block_reserved_owner_before_work(vector_store, method):
    database = vector_store.db
    options = {"user_id": vector._UNOWNED_SENTINEL}
    if method in {"insert", "upsert"}:
        options.update(content_hash="protected-hash", documents=[Document(content="Must not be embedded")])
    elif method == "search":
        options["query"] = "Must not be embedded"
    elif method == "delete_by_content_id":
        options["content_id"] = "protected-content"
    else:
        options["content_hash"] = "protected-hash"
    with pytest.raises(ValueError, match="reserved"):
        getattr(database, method)(**options)
    assert_no_database_or_embedding(vector_store)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["async_insert", "async_upsert"])
@pytest.mark.parametrize("empty", [False, True], ids=["with-document", "empty-list"])
async def test_async_ingestion_blocks_reserved_owner_before_work(vector_store, method, empty):
    documents = [] if empty else [Document(content="Must not be embedded")]
    with pytest.raises(ValueError, match="reserved"):
        await getattr(vector_store.db, method)(
            content_hash="protected-hash", documents=documents, user_id=vector._UNOWNED_SENTINEL
        )
    assert_no_database_or_embedding(vector_store)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["async_insert", "async_upsert"])
@pytest.mark.parametrize("size", [0, -1])
async def test_existing_async_batch_size_validation_still_runs(vector_store, method, size):
    with pytest.raises(ValueError, match="batch_size"):
        await getattr(vector_store.db, method)(content_hash="h", documents=[], user_id="alice", batch_size=size)
    assert_no_database_or_embedding(vector_store)


@pytest.mark.parametrize("method", ["_insert_prepared_batch", "_replace_prepared_documents"])
def test_worker_helpers_reject_reserved_owner_and_still_remove_scoped_session(vector_store, method):
    with pytest.raises(ValueError, match="reserved"):
        getattr(vector_store.db, method)(content_hash="protected-hash", documents=[], user_id=vector._UNOWNED_SENTINEL)
    assert_no_database_or_embedding(vector_store)
    vector_store.db.Session.remove.assert_called_once_with()


def test_search_scope_rejects_literal_marker(vector_store):
    statement = select(vector_store.db.table.c.id)
    with pytest.raises(ValueError, match="reserved"):
        vector_store.db._apply_user_scope(statement, vector._UNOWNED_SENTINEL)
    assert_no_database_or_embedding(vector_store)


def test_none_search_scope_is_unchanged(vector_store):
    statement = select(vector_store.db.table.c.id)
    assert vector_store.db._apply_user_scope(statement, None) is statement
    assert_no_database_or_embedding(vector_store)


@pytest.mark.parametrize("owner", ["", "alice"])
def test_valid_search_scopes_keep_encoded_owner_and_shared_null_bucket(vector_store, owner):
    statement = vector_store.db._apply_user_scope(select(vector_store.db.table.c.id), owner)
    compiled = statement.compile()
    expected = vector._UNOWNED_SENTINEL if owner == "" else owner
    assert list(compiled.params.values()) == [expected]
    assert "IS NULL" in str(compiled)
    vector_store.db._user_id_column_exists.assert_called_once_with()
    vector_store.db.Session.assert_not_called()
