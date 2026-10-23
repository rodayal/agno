"""Live Oracle regression tests grouped by behavior; no cases are discarded."""

import os
import uuid

import pytest
from agno.db.oracle import OracleDb
from agno.db.schemas.knowledge import KnowledgeRow
from agno.db.schemas.memory import UserMemory
from agno.session import AgentSession
from sqlalchemy import create_engine, event, text

# Owner encoding at the database boundary

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
def owner_db(_oracle_server):
    suffix = uuid.uuid4().hex[:8]
    database = OracleDb(
        db_url=DB_URL,
        session_table=f"test_owner_sess_{suffix}",
        runs_table=f"test_owner_runs_{suffix}",
        memory_table=f"test_owner_mem_{suffix}",
        metrics_table=f"test_owner_metrics_{suffix}",
        knowledge_table=f"test_owner_knowledge_{suffix}",
        auth_tokens_table=f"test_owner_auth_{suffix}",
    )
    yield database
    database.Session.remove()
    with database.db_engine.begin() as conn:
        for t in (
            database.runs_table_name,
            database.session_table_name,
            database.memory_table_name,
            database.metrics_table_name,
            database.knowledge_table_name,
            database.auth_tokens_table_name,
        ):
            if database.table_exists(t):
                conn.execute(text(f"DROP TABLE {t} CASCADE CONSTRAINTS"))
    database.db_engine.dispose()


def _install_guard(engine):
    """Record every bound parameter whose key names an owner column.

    Returns the list the event listener appends to; the caller inspects it
    after running whatever scenario it wants covered.
    """
    violations: list = []

    def _before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        param_sets = parameters if executemany else [parameters]
        for params in param_sets:
            if params is None:
                continue
            items = params.items() if isinstance(params, dict) else enumerate(params)
            for key, value in items:
                key_name = str(key).lower()
                if "user_id" in key_name and value == "":
                    violations.append((statement, key, value))

    event.listen(engine, "before_cursor_execute", _before_cursor_execute)
    return violations


def test_owner_field_never_sent_as_bare_empty_string(owner_db):
    """Exercises every ticket 03/04/08 path that touches an owner field with
    the unowned-bucket value, and fails if any bound parameter reaches the
    driver as a bare "" instead of the translated sentinel.
    """
    violations = _install_guard(owner_db.db_engine)

    # Sessions: an unowned session.
    session = AgentSession(session_id="owner-guard-session", agent_id="a", user_id="", created_at=1700000000)
    owner_db.upsert_session(session)
    owner_db.get_session("owner-guard-session")
    owner_db.get_sessions(user_id="")
    owner_db.rename_session("owner-guard-session", None, "renamed")
    owner_db.delete_session("owner-guard-session")

    # Memories: an unowned memory.
    memory = UserMemory(memory="unowned memory", user_id="")
    saved = owner_db.upsert_user_memory(memory)
    owner_db.get_user_memory(saved.memory_id, user_id="")
    owner_db.get_user_memories(user_id="")
    owner_db.get_all_memory_topics(user_id="")
    owner_db.delete_user_memory(saved.memory_id, user_id="")

    # Metrics: the unowned bucket is written through calculate_metrics via a
    # session with user_id=None (translated to "" internally by
    # calculate_date_metrics), and read back with user_id="".
    metrics_session = AgentSession(session_id="owner-guard-metrics", agent_id="a", user_id=None, created_at=1700000001)
    owner_db.upsert_session(metrics_session, deserialize=False)
    owner_db.calculate_metrics()
    owner_db.get_metrics(user_id="")
    owner_db.delete_session("owner-guard-metrics")

    # Auth tokens: an unowned token. user_id is NOT NULL on this table, so
    # Postgres itself already stores "" for the unowned case -- exactly the
    # value Oracle's empty-string-is-null folding would otherwise destroy.
    owner_db.upsert_auth_token({"provider": "owner-guard", "user_id": "", "service": "oauth", "token_data": {"t": "1"}})
    owner_db.get_auth_token("owner-guard", "", "oauth")
    owner_db.delete_auth_token("owner-guard", "", "oauth")

    # Knowledge: intentionally NOT exercised with user_id="" -- knowledge's
    # owner column has no unowned-bucket convention (None means shared, per
    # its own schema comment), so "" is not a value that domain ever takes.
    knowledge = KnowledgeRow(name="doc", description="d", user_id=None)
    owner_db.upsert_knowledge_content(knowledge)
    owner_db.get_knowledge_contents(user_id=None)
    owner_db.delete_knowledge_content(knowledge.id)

    assert not violations, (
        f"owner field reached the database as a bare empty string in {len(violations)} bind(s): {violations[:5]}"
    )

    # Cross-check the guard itself is not a no-op: deliberately bind "" for a
    # non-owner-like key and confirm the pattern-matcher would have caught an
    # owner-field violation shaped like it.
    with owner_db.db_engine.connect() as conn:
        conn.execute(text("SELECT :probe_user_id AS x FROM dual"), {"probe_user_id": ""})
    assert any(v[1] == "probe_user_id" for v in violations), "the guard failed to catch its own positive control"


# Learning search errors


@pytest.fixture
def learnings_db(_oracle_server):
    suffix = uuid.uuid4().hex[:8]
    database = OracleDb(db_url=DB_URL, learnings_table=f"test_search_raises_{suffix}")
    yield database
    database.Session.remove()
    if database.table_exists(database.learnings_table_name):
        with database.db_engine.begin() as conn:
            conn.execute(text(f"DROP TABLE {database.learnings_table_name} CASCADE CONSTRAINTS"))
    database.db_engine.dispose()


def test_search_learnings_on_populated_store_finds_nothing_returns_empty(learnings_db):
    """The unpopulated/no-match case is not an error: it returns []."""
    learnings_db.upsert_learning(
        id="l1", learning_type="user_profile", content={"summary": "hello world"}, user_id="alice"
    )
    assert learnings_db.search_learnings(query="no-such-term-zzz") == []


def test_search_learnings_raises_on_a_broken_query_rather_than_returning_empty(learnings_db):
    """A genuine query failure (here: a bind-type mismatch on an existing,
    populated table) must propagate, not come back as an empty list -- the
    caller would otherwise have no way to distinguish "nothing matched" from
    "the query itself is broken".
    """
    learnings_db.upsert_learning(
        id="l1", learning_type="user_profile", content={"summary": "hello world"}, user_id="alice"
    )

    with pytest.raises(Exception):
        learnings_db.search_learnings(query="alice", learning_type={"not": "a valid value"})  # type: ignore[arg-type]
