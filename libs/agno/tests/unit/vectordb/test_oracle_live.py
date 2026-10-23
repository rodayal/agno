"""Live Oracle regression tests grouped by behavior; no cases are discarded."""

import os
import uuid
from hashlib import sha256

import pytest
from agno.filters import AND, EQ, GT, IN, LT, NOT, OR, STARTSWITH
from agno.knowledge.document import Document
from agno.vectordb.oracle.oracle import OracleVector, _to_db_user_id
from sqlalchemy import create_engine, text

from .conftest import DeterministicEmbedder

# Search, filters and metadata

DB_URL = os.getenv("AGNO_TEST_ORACLE_URL", "")


def _server_reachable() -> bool:
    if not DB_URL:
        return False
    engine = create_engine(DB_URL)
    try:
        with engine.connect() as conn:
            conn.execute(text("select 1 from dual"))
        return True
    except Exception:
        return False
    finally:
        engine.dispose()


@pytest.fixture(scope="module")
def _oracle_server():
    if not _server_reachable():
        pytest.skip("AGNO_TEST_ORACLE_URL not set or Oracle is unavailable")


@pytest.fixture
def search_db(_oracle_server):
    table_name = f"test_filt_{uuid.uuid4().hex[:8]}"
    database = OracleVector(table_name=table_name, db_url=DB_URL, embedder=DeterministicEmbedder())
    database.create()
    database.insert(
        content_hash="h1",
        documents=[
            Document(name="a", content="alpha document", meta_data={"views": 9}),
            Document(name="b", content="beta document", meta_data={"views": 10}),
            Document(name="c", content="gamma document", meta_data={"views": 100}),
        ],
    )
    yield database
    database.drop()


class TestNumericFilterComparesAsNumber:
    """The specific defect this ticket calls out as a known mistake in an
    existing third-party Oracle vector store: without RETURNING NUMBER,
    JSON_VALUE returns a string and nine sorts above ten lexicographically.
    """

    def test_gt_excludes_lexicographically_larger_but_numerically_smaller(self, search_db):
        # "10" > "9" numerically is False if compared as strings ("10" < "9" lexically).
        results = search_db.vector_search("document", limit=10, filters=[GT("views", 9)])
        assert {r.name for r in results} == {"b", "c"}

    def test_lt_includes_only_numerically_smaller_values(self, search_db):
        results = search_db.vector_search("document", limit=10, filters=[LT("views", 10)])
        assert {r.name for r in results} == {"a"}

    def test_gt_100_way_boundary(self, search_db):
        """A lexicographic comparison would put "100" between "10" and "9"'s
        digit-by-digit prefix in confusing ways; a numeric comparison must not.
        """
        results = search_db.vector_search("document", limit=10, filters=[GT("views", 50)])
        assert {r.name for r in results} == {"c"}


class TestSupportedFilterOperatorSet:
    """Exactly PgVector's own operator set: EQ, IN, GT, LT, NOT, AND, OR.
    NEQ, GTE, LTE, CONTAINS, STARTSWITH are part of the broader FilterExpr
    language but not supported by the reference store, so they raise here too.
    """

    def test_eq_works(self, search_db):
        assert {r.name for r in search_db.vector_search("document", limit=10, filters=[EQ("views", 9)])} == {"a"}

    def test_in_works(self, search_db):
        assert {r.name for r in search_db.vector_search("document", limit=10, filters=[IN("views", [9, 10])])} == {
            "a",
            "b",
        }

    def test_and_works(self, search_db):
        results = search_db.vector_search("document", limit=10, filters=[AND(GT("views", 5), LT("views", 50))])
        assert {r.name for r in results} == {"a", "b"}

    def test_or_works(self, search_db):
        results = search_db.vector_search("document", limit=10, filters=[OR(EQ("views", 9), EQ("views", 100))])
        assert {r.name for r in results} == {"a", "c"}

    def test_not_works(self, search_db):
        results = search_db.vector_search("document", limit=10, filters=[NOT(EQ("views", 9))])
        assert {r.name for r in results} == {"b", "c"}

    def test_startswith_raises(self, search_db):
        with pytest.raises(ValueError):
            search_db.vector_search("document", limit=10, filters=[STARTSWITH("name", "a")])


class TestHybridWeightBoundary:
    def test_full_vector_weight_equals_pure_vector_search(self, search_db):
        # hybrid_search always needs the CTXSYS.CONTEXT index, even at
        # vector_score_weight=1.0: the text-scoring subquery is part of the
        # query regardless of the weight applied to its result.
        search_db.optimize()
        search_db.vector_score_weight = 1.0
        try:
            hybrid = search_db.hybrid_search("document", limit=3)
            vector = search_db.vector_search("document", limit=3)
            assert [r.id for r in hybrid] == [r.id for r in vector]
        finally:
            search_db.vector_score_weight = 0.5

    def test_out_of_range_weight_raises(self, search_db):
        search_db.vector_score_weight = 2.0
        try:
            with pytest.raises(ValueError):
                search_db.hybrid_search("document", limit=3)
        finally:
            search_db.vector_score_weight = 0.5


class TestTextSearchUnavailableIsActionable:
    def test_keyword_search_without_optimize_raises_actionable_error(self, search_db):
        """No CTXSYS.CONTEXT index has been created (optimize() was never
        called) -- must fail with a message naming the fix, not a raw
        DRG-*/ORA-29902 driver error.
        """
        with pytest.raises(ValueError, match="optimize"):
            search_db.keyword_search("document")

    def test_keyword_search_works_after_optimize(self, search_db):
        search_db.optimize()
        results = search_db.keyword_search("alpha")
        assert any(r.name == "a" for r in results)


class TestTextIndexFreshness:
    def test_keyword_search_sees_document_inserted_after_optimize(self, search_db):
        search_db.optimize()
        search_db.insert(
            content_hash="post-index-insert",
            documents=[Document(name="saturn-doc", content="Space missions to Saturn")],
        )

        results = search_db.keyword_search("Saturn", limit=1)

        assert [result.name for result in results] == ["saturn-doc"]

    def test_keyword_search_sees_document_upserted_after_optimize(self, search_db):
        search_db.insert(
            content_hash="post-index-upsert",
            documents=[Document(name="mars-doc", content="Space missions to Mars")],
        )
        search_db.optimize()
        search_db.upsert(
            content_hash="post-index-upsert",
            documents=[Document(name="jupiter-doc", content="Space missions to Jupiter")],
        )

        results = search_db.keyword_search("Jupiter", limit=1)

        assert [result.name for result in results] == ["jupiter-doc"]


def test_count_includes_unnamed_documents(search_db):
    # This fixture owns a uniquely named temporary table.
    # Remove its three sample documents, not any application data.
    with search_db.db_engine.begin() as connection:
        connection.execute(search_db.table.delete())

    assert search_db.get_count() == 0

    search_db.insert(
        content_hash="count-unnamed",
        documents=[
            Document(content="First unnamed document"),
            Document(content="Second unnamed document"),
        ],
    )
    assert search_db.get_count() == 2

    search_db.insert(
        content_hash="count-named",
        documents=[
            Document(name="named-document", content="Third document"),
        ],
    )
    assert search_db.get_count() == 3


def test_numeric_metadata_update_after_reopen(search_db):
    import json

    from sqlalchemy import select

    content_id = "numeric-metadata-regression"
    search_db.insert(
        content_hash="numeric-metadata-hash",
        documents=[
            Document(
                content="A document for testing metadata updates.",
                content_id=content_id,
                meta_data={"views": 2, "price": 1.25, "nested": {"counts": [3, 4.5]}, "label": "123"},
            )
        ],
        filters={"priority": 7},
    )
    reopened = OracleVector(
        table_name=search_db.table_name, db_engine=search_db.db_engine, embedder=DeterministicEmbedder()
    )

    def read_record():
        with reopened.Session() as session:
            return session.execute(
                select(reopened.table.c.meta_data, reopened.table.c.filters, reopened.table.c.usage).where(
                    reopened.table.c.content_id == content_id
                )
            ).one()

    try:
        before = read_record()
        assert type(before.meta_data["views"]) is int
        assert type(before.meta_data["price"]) is float
        assert type(before.meta_data["nested"]["counts"][0]) is int
        assert type(before.meta_data["nested"]["counts"][1]) is float
        assert type(before.filters["priority"]) is int
        assert type(before.usage["total_tokens"]) is int
        json.dumps(dict(before._mapping))

        patch = {"views": 5, "optional": None}
        reopened.update_metadata(content_id, patch)
        after = read_record()
        assert after.meta_data == {
            "views": 5,
            "price": 1.25,
            "nested": {"counts": [3, 4.5]},
            "label": "123",
            "priority": 7,
            "optional": None,
        }
        assert after.filters == patch
        assert after.usage == before.usage
        assert type(after.meta_data["views"]) is int
        assert type(after.meta_data["price"]) is float
        json.dumps(dict(after._mapping))

        # Preserve the existing shallow-merge contract: replace nested objects,
        # rather than recursively merging them. Explicit None remains a key.
        reopened.update_metadata(content_id, {"nested": {"replacement": 9}})
        final = read_record()
        assert final.meta_data == {**after.meta_data, "nested": {"replacement": 9}}
        assert final.filters == {"nested": {"replacement": 9}}
        json.dumps(dict(final._mapping))
    finally:
        # The fixture owns table/engine cleanup; this adapter borrows its engine.
        reopened.Session.remove()


# Owner isolation


@pytest.fixture
def owner_db(_oracle_server):
    table_name = f"test_iso_{uuid.uuid4().hex[:8]}"
    database = OracleVector(table_name=table_name, db_url=DB_URL, embedder=DeterministicEmbedder())
    database.create()
    yield database
    database.drop()


def _raw_row(owner_db: OracleVector, doc_id: str):
    with owner_db.Session() as sess:
        return sess.execute(
            text(f"SELECT user_id FROM {owner_db.table_name} WHERE id = :id"), {"id": doc_id}
        ).fetchone()


class TestWriteStampsOwner:
    """The owner is stored as a first-class ``user_id`` column."""

    def test_explicit_user_id_stamped(self, owner_db):
        doc = Document(name="alice-salary", content="Alice's salary is 180k.")
        owner_db.insert(content_hash="h1", documents=[doc], user_id="alice")
        record_id = owner_db._scoped_record_id(sha256(doc.content.encode()).hexdigest(), "h1", "alice")
        row = _raw_row(owner_db, record_id)
        assert row is not None and row.user_id == "alice"

    def test_none_user_id_stored_as_null(self, owner_db):
        doc = Document(name="company-holidays", content="The office is closed Jan 1.")
        owner_db.insert(content_hash="h2", documents=[doc], user_id=None)
        record_id = owner_db._scoped_record_id(sha256(doc.content.encode()).hexdigest(), "h2", None)
        row = _raw_row(owner_db, record_id)
        assert row is not None and row.user_id is None


class TestEmptyStringOwnerSentinel:
    """The empty-string-is-null problem: the highest-stakes case this ticket calls
    out. Oracle folds "" to NULL on write, and NULL is the genuine shared bucket --
    an unowned owner arriving from the application must become the sentinel
    (verified directly against the raw stored column) or it silently leaks into
    the shared bucket, with no error raised.
    """

    def test_empty_string_owner_is_not_stored_as_null(self, owner_db):
        doc = Document(name="empty-owner-doc", content="Content with an empty owner string.")
        owner_db.insert(content_hash="h3", documents=[doc], user_id="")
        record_id = owner_db._scoped_record_id(sha256(doc.content.encode()).hexdigest(), "h3", "")
        row = _raw_row(owner_db, record_id)
        assert row is not None
        assert row.user_id is not None, "an empty-string owner must never be stored as NULL (the shared bucket)"
        assert row.user_id == _to_db_user_id("")

    def test_empty_string_owner_does_not_leak_into_shared_search(self, owner_db):
        empty_doc = Document(name="empty-owner-doc", content="Secret content with an empty owner string.")
        shared_doc = Document(name="shared-doc", content="Genuinely shared content.")
        owner_db.insert(content_hash="h4", documents=[empty_doc], user_id="")
        owner_db.insert(content_hash="h5", documents=[shared_doc], user_id=None)

        results = owner_db.search("content", limit=10, user_id="someone-else")
        names = {r.name for r in results}
        assert "shared-doc" in names
        assert "empty-owner-doc" not in names, "an empty-string owner's row leaked into the shared (NULL) bucket"

    def test_empty_string_owner_sees_its_own_content(self, owner_db):
        doc = Document(name="empty-owner-doc", content="Content only the empty-string owner should see again.")
        owner_db.insert(content_hash="h6", documents=[doc], user_id="")
        results = owner_db.search("Content only the empty-string owner", limit=10, user_id="")
        assert any(r.name == "empty-owner-doc" for r in results)


class TestOwnerFoldedId:
    """Two owners uploading the same content get distinct row ids."""

    def test_identical_content_two_owners_get_distinct_ids(self, owner_db):
        content = "The merger closes in Q3."
        alice_id = owner_db._scoped_record_id(__import__("hashlib").md5(content.encode()).hexdigest(), "hX", "alice")
        bob_id = owner_db._scoped_record_id(__import__("hashlib").md5(content.encode()).hexdigest(), "hX", "bob")
        assert alice_id != bob_id

    def test_partitioned_inputs_do_not_collide(self, owner_db):
        """A base id with a trailing fragment must not collide with another whose
        owner happens to supply that fragment -- the two-stage digest folds
        base_id+content_hash first, then owner, rather than joining raw strings.
        """
        id_a = owner_db._scoped_record_id("doc_1", "hY", "alice")
        id_b = owner_db._scoped_record_id("doc", "hY", "1_alice")
        assert id_a != id_b


class TestSearchScope:
    """A scoped search returns own rows plus shared, never another owner's."""

    def test_named_owner_sees_own_and_shared_not_others(self, owner_db):
        owner_db.insert(
            content_hash="ha", documents=[Document(name="alice-doc", content="Alice private note.")], user_id="alice"
        )
        owner_db.insert(
            content_hash="hb", documents=[Document(name="bob-doc", content="Bob private note.")], user_id="bob"
        )
        owner_db.insert(
            content_hash="hs", documents=[Document(name="shared-doc", content="Shared note.")], user_id=None
        )

        alice_results = {r.name for r in owner_db.search("note", limit=10, user_id="alice")}
        assert "alice-doc" in alice_results
        assert "shared-doc" in alice_results
        assert "bob-doc" not in alice_results

    def test_owner_scoping_applies_to_keyword_and_hybrid_too(self, owner_db):
        """Ticket 18: owner scoping is applied before filters on all three
        search paths, not just vector_search.
        """
        owner_db.insert(
            content_hash="ka",
            documents=[Document(name="alice-secret", content="Alice keeps a secret furry pet journal.")],
            user_id="alice",
        )
        owner_db.insert(
            content_hash="kb",
            documents=[Document(name="bob-journal", content="Bob keeps a furry pet journal too.")],
            user_id="bob",
        )
        owner_db.optimize()

        bob_keyword_names = {r.name for r in owner_db.keyword_search("furry pet journal", limit=10, user_id="bob")}
        assert "alice-secret" not in bob_keyword_names
        assert "bob-journal" in bob_keyword_names

        bob_hybrid_names = {r.name for r in owner_db.hybrid_search("furry pet journal", limit=10, user_id="bob")}
        assert "alice-secret" not in bob_hybrid_names
        assert "bob-journal" in bob_hybrid_names


class TestUpsertDedupScope:
    """Upsert dedup keys on content_hash scoped by owner: it clears only the
    writing owner's own bucket, never another owner's or the shared bucket.
    """

    def test_dedup_is_scoped_to_writing_owner(self, owner_db):
        owner_db.insert(content_hash="hd", documents=[Document(name="d1", content="first version")], user_id="carol")
        owner_db.upsert(content_hash="hd", documents=[Document(name="d1", content="second version")], user_id="carol")
        results = owner_db.search("version", limit=10, user_id="carol")
        matching = [r for r in results if r.name == "d1"]
        assert len(matching) == 1
        assert matching[0].content == "second version"

    def test_shared_upsert_dedup_does_not_touch_owned_rows(self, owner_db):
        owner_db.insert(content_hash="he", documents=[Document(name="owned", content="owned content")], user_id="dave")
        owner_db.insert(
            content_hash="he", documents=[Document(name="shared", content="shared content v1")], user_id=None
        )
        owner_db.upsert(
            content_hash="he", documents=[Document(name="shared", content="shared content v2")], user_id=None
        )
        owned_still_there = owner_db.search("owned content", limit=10, user_id="dave")
        assert any(r.name == "owned" for r in owned_still_there)


class TestDeleteScope:
    """``delete_by_content_id``/``_delete_by_content_hash`` with a ``user_id``
    restrict the delete to that owner's rows.
    """

    def test_scoped_delete_by_content_hash_restricts_to_owner(self, owner_db):
        owner_db.insert(
            content_hash="hf", documents=[Document(name="erin-doc", content="erin content")], user_id="erin"
        )
        deleted = owner_db._delete_by_content_hash("hf", user_id="frank")
        assert deleted is False
        still_there = owner_db.search("erin content", limit=10, user_id="erin")
        assert any(r.name == "erin-doc" for r in still_there)

        deleted_correctly = owner_db._delete_by_content_hash("hf", user_id="erin")
        assert deleted_correctly is True
