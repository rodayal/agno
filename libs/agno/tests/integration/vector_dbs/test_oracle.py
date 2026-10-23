"""Oracle regression tests grouped by behavior; no cases are discarded."""

import os
import uuid
from hashlib import md5
from inspect import isawaitable
from typing import Any, Dict, List
from uuid import uuid4

import pytest
from agno.db.oracle._version import detect_capabilities
from agno.knowledge.document import Document
from agno.vectordb.distance import Distance
from agno.vectordb.oracle import OracleVector
from agno.vectordb.pgvector import PgVector
from sqlalchemy import create_engine, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.inspection import inspect

# PostgreSQL ranking parity

PG_URL = "postgresql+psycopg://ai:ai@localhost:5532/ai"
ORACLE_URL = os.getenv("AGNO_TEST_ORACLE_URL", "")


class _CorpusEmbedder:
    """Deterministic, offline, bag-of-hashed-words embedder with genuine
    geometric structure (unlike a pure one-hot embedder, two texts sharing
    words end up closer together), so a differential ranking comparison is
    meaningful rather than trivially degenerate.
    """

    enable_batch = False
    dimensions = 16

    def _vec(self, text: str) -> List[float]:
        words = text.lower().split()
        vec = [0.0] * self.dimensions
        for w in words:
            vec[int(md5(w.encode()).hexdigest(), 16) % self.dimensions] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec] if norm else vec

    def get_embedding(self, text: str) -> List[float]:
        return self._vec(text)

    def get_embedding_and_usage(self, text: str):
        return self._vec(text), {"total_tokens": 1}

    def embed(self, document, *args, **kwargs):
        document.embedding = self._vec(document.content)
        document.usage = {"total_tokens": 1}
        return document


def _reachable(url: str) -> bool:
    if not url:
        return False
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            conn.execute(text("select 1 from dual") if "oracle" in url else text("select 1"))
        return True
    except Exception:
        return False
    finally:
        engine.dispose()


@pytest.fixture(scope="module")
def _servers_up():
    if not _reachable(PG_URL):
        pytest.skip(f"Postgres server not reachable at {PG_URL}")
    if not _reachable(ORACLE_URL):
        pytest.skip("AGNO_TEST_ORACLE_URL not set or Oracle is unavailable")


@pytest.fixture
def pg_db(_servers_up):
    table_name = f"diff_vec_{uuid.uuid4().hex[:8]}"
    database = PgVector(table_name=table_name, db_url=PG_URL, embedder=_CorpusEmbedder(), distance=Distance.cosine)
    database.create()
    yield database
    database.drop()


@pytest.fixture
def oracle_db(_servers_up):
    table_name = f"diff_vec_{uuid.uuid4().hex[:8]}"
    database = OracleVector(
        table_name=table_name, db_url=ORACLE_URL, embedder=_CorpusEmbedder(), distance=Distance.cosine
    )
    database.create()
    yield database
    database.drop()


CORPUS = [
    Document(name="cats", content="Cats are small furry mammals that purr and sleep most of the day."),
    Document(name="dogs", content="Dogs are loyal furry mammals that bark and love to play fetch."),
    Document(name="rockets", content="Rockets use combustion to reach orbit and travel through space."),
    Document(name="planets", content="Planets orbit stars and some have rings made of ice and rock."),
    Document(name="bread", content="Bread is made from flour, water, yeast, and baked in an oven."),
]

QUERIES = [
    "furry pet mammal",
    "space travel orbit",
    "baking food oven",
]


def _run_ranking_scenario(db) -> Dict[str, Any]:
    db.insert(content_hash="corpus-1", documents=[Document(name=d.name, content=d.content) for d in CORPUS])

    results: Dict[str, Any] = {}
    for query in QUERIES:
        matches = db.search(query, limit=5)
        results[query] = {
            "order": [m.name for m in matches],
            "count": len(matches),
        }
    return results


def test_vector_search_ranking_matches_postgres(pg_db, oracle_db):
    """Same corpus, same queries, same embedder: the top-k ordering must match."""
    pg_result = _run_ranking_scenario(pg_db)
    oracle_result = _run_ranking_scenario(oracle_db)

    assert oracle_result == pg_result, (
        f"OracleVector diverged from PgVector.\nPostgres: {pg_result}\nOracle:   {oracle_result}"
    )


def test_similarity_scores_are_close_across_backends(pg_db, oracle_db):
    """The similarity scores themselves (not just the ordering) should agree
    within floating-point tolerance -- both compute cosine distance from the
    same vectors, just through different SQL functions.
    """
    for db in (pg_db, oracle_db):
        db.insert(content_hash="corpus-1", documents=[Document(name=d.name, content=d.content) for d in CORPUS])

    pg_top = pg_db.search("furry pet mammal", limit=1)[0]
    oracle_top = oracle_db.search("furry pet mammal", limit=1)[0]

    assert pg_top.name == oracle_top.name
    pg_score = pg_top.meta_data["similarity_score"]
    oracle_score = oracle_top.meta_data["similarity_score"]
    assert abs(pg_score - oracle_score) < 1e-6, f"similarity scores diverged: pg={pg_score} oracle={oracle_score}"


def _ensure_keyword_index(db) -> None:
    """Create the text/keyword index Oracle strictly requires; skip index
    creation for Postgres entirely.

    Unlike Oracle Text (CONTAINS() cannot run at all without a CTXSYS.CONTEXT
    index), Postgres's ``to_tsvector``/``to_tsquery`` need no index for
    correctness -- only for performance -- and PgVector's own index-creation
    helpers have pre-existing bugs unrelated to this ticket that make calling
    them here counter-productive: ``_create_vector_index`` (HNSW/Ivfflat)
    binds a parameter inside a DDL ``WITH (...)`` clause, invalid syntax under
    psycopg3's extended protocol (confirmed live: "syntax error at or near
    '$1'"); ``_create_gin_index`` interpolates ``content_language`` as a bare
    unquoted SQL identifier instead of a string literal (confirmed live:
    "column 'english' does not exist"). Neither is exercised by this
    ranking-correctness differential, which needs no index on either backend.
    """
    if not isinstance(db, PgVector):
        db.optimize()


def _run_keyword_ranking_scenario(db) -> Dict[str, Any]:
    """Ticket 18's keyword-search differential: same corpus, same query, same
    text-relevance ordering. Query terms are chosen to appear verbatim (after
    each backend's own stemming) in more than one document, so the ordering
    comparison is meaningful rather than trivial.
    """
    db.insert(content_hash="corpus-kw", documents=[Document(name=d.name, content=d.content) for d in CORPUS])
    _ensure_keyword_index(db)
    matches = db.keyword_search("furry mammals", limit=5)
    return {"order": [m.name for m in matches], "count": len(matches)}


def test_keyword_search_ranking_matches_postgres(pg_db, oracle_db):
    pg_result = _run_keyword_ranking_scenario(pg_db)
    oracle_result = _run_keyword_ranking_scenario(oracle_db)

    assert oracle_result == pg_result, (
        f"OracleVector keyword search diverged from PgVector.\nPostgres: {pg_result}\nOracle:   {oracle_result}"
    )


def _run_hybrid_ranking_scenario(db) -> Dict[str, Any]:
    db.insert(content_hash="corpus-hy", documents=[Document(name=d.name, content=d.content) for d in CORPUS])
    _ensure_keyword_index(db)
    db.vector_score_weight = 0.5
    matches = db.hybrid_search("furry mammals pets", limit=5)
    return {"order": [m.name for m in matches], "count": len(matches)}


def test_hybrid_search_ranking_matches_postgres(pg_db, oracle_db):
    """Same corpus/query, same vector_score_weight: hybrid's weighted-sum-in-SQL
    design (not rank fusion) means the ordering should agree across backends,
    not merely happen to overlap.
    """
    pg_result = _run_hybrid_ranking_scenario(pg_db)
    oracle_result = _run_hybrid_ranking_scenario(oracle_db)

    assert oracle_result == pg_result, (
        f"OracleVector hybrid search diverged from PgVector.\nPostgres: {pg_result}\nOracle:   {oracle_result}"
    )


# Oracle record identity


class LocalEmbedder:
    dimensions = 3
    enable_batch = False

    def get_embedding_and_usage(self, content):
        return [1.0, 0.0, 0.0], {"total_tokens": 0}

    async def async_get_embedding_and_usage(self, content):
        return self.get_embedding_and_usage(content)


@pytest.fixture
def store():
    db_url = os.getenv("AGNO_TEST_ORACLE_URL", "")
    if not db_url:
        pytest.skip("AGNO_TEST_ORACLE_URL is not configured")
    if make_url(db_url).drivername != "oracle+oracledb":
        pytest.fail("This test requires a synchronous Oracle connection URL")

    engine = create_engine(db_url, hide_parameters=True)
    database = None
    created = False
    table_name = "test_oid_" + uuid4().hex[:12]
    try:
        if not detect_capabilities(engine).vector:
            pytest.skip("Oracle 23ai+ vector support is required")
        if inspect(engine).has_table(table_name):
            pytest.fail("Refusing to reuse a pre-existing table")
        database = OracleVector(table_name=table_name, db_engine=engine, embedder=LocalEmbedder())
        database.create()
        created = True
        yield database
    finally:
        try:
            if database is not None:
                database.Session.remove()
            if created:
                database.table.drop(engine, checkfirst=False)
                assert not inspect(engine).has_table(table_name), "Temporary table cleanup failed"
        finally:
            engine.dispose()


async def write(store, method, base_id, content_hash, owner, name, content):
    result = getattr(store, method)(
        content_hash=content_hash,
        documents=[Document(id=base_id, name=name, content=content)],
        user_id=owner,
    )
    if isawaitable(result):
        await result


def snapshot(store):
    with store.db_engine.connect() as conn:
        rows = (
            conn.execute(
                select(
                    store.table.c.id,
                    store.table.c.name,
                    store.table.c.content,
                    store.table.c.content_hash,
                    store.table.c.user_id,
                )
            )
            .mappings()
            .all()
        )
    result = {(row["user_id"], row["content_hash"]): dict(row) for row in rows}
    assert len(result) == len(rows), "Duplicate records for one test identity"
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["insert", "upsert", "async_insert", "async_upsert"])
async def test_distinct_hash_fields_survive_and_updates_preserve_other_records(store, method):
    await write(store, method, "a_b", "c", "alice", "first-alice", "Synthetic first document")
    await write(store, method, "a", "b_c", "alice", "second-alice", "Synthetic second document")
    await write(store, method, "a_b", "c", "bob", "bob-control", "Synthetic first document")

    before = snapshot(store)
    assert set(before) == {("alice", "c"), ("alice", "b_c"), ("bob", "c")}
    assert len({row["id"] for row in before.values()}) == 3
    assert all(len(row["id"]) == 64 for row in before.values())
    assert store.get_count() == 3

    update_method = "async_upsert" if method.startswith("async_") else "upsert"
    for _ in range(2):
        await write(store, update_method, "a_b", "c", "alice", "first-alice", "Synthetic revised document")
        after = snapshot(store)
        assert set(after) == set(before)
        assert store.get_count() == 3
        assert after[("alice", "c")]["id"] == before[("alice", "c")]["id"]
        assert after[("alice", "c")]["content"] == "Synthetic revised document"
        assert after[("alice", "b_c")] == before[("alice", "b_c")]
        assert after[("bob", "c")] == before[("bob", "c")]
