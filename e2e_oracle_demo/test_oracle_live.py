"""Opt-in storage acceptance against the demo-owned Oracle namespace.

Chat uses the static model. Provider-backed LIVE document tests additionally
require AGNO_DEMO_ALLOW_EMBEDDING_TESTS=1. This is not live-LLM acceptance.
"""

import asyncio
import json
import os
import time
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from .app import create_app
from .runtime import Runtime, make_report

pytestmark = pytest.mark.skipif(
    os.getenv("AGNO_DEMO_ALLOW_ORACLE_TESTS") != "1",
    reason="Explicit Oracle demo test opt-in required",
)


@pytest.fixture(scope="module")
def client():
    current = Runtime()
    with TestClient(create_app(runtime=current)) as connection:
        yield connection


def post(client, path, body):
    response = client.post(path, json=body)
    assert response.status_code == 200, response.text[:800]
    return response.json()


def chat(client, **options):
    return post(
        client,
        "/api/chat",
        {
            "user_id": "alice",
            "mode": "static",
            "message": "A freely entered test message.",
            **options,
        },
    )


def scope(user="alice"):
    return {"user_id": user, "mode": "static", "component": "agent"}


def test_oracle_identity_and_clear_model_status(client):
    result = client.get("/api/status").json()
    assert result["database"]["connected"]
    assert result["database"]["identity"]["user"] == "AGNO_TEST_USER"
    assert result["database"]["identity"]["container"] == "CDB1_PDB1"
    assert result["capabilities"]["paid_requests_automatic"] is False


def test_history_survives_a_fresh_runtime_and_user_cannot_access_other_session(client):
    marker = "history-" + uuid4().hex
    result = chat(client, message=marker)
    assert result["effective_mode"] == "static" and result["status"] == "COMPLETED"
    followup = chat(
        client,
        session_id=result["session_id"],
        message="Show the earlier context you received.",
    )
    assert marker in followup["content"]
    fresh = Runtime()
    try:
        saved = fresh.session(result["session_id"], "alice", "static", "agent")
        assert marker in json.dumps(saved)
    finally:
        fresh.close()
    response = client.get("/api/sessions/" + result["session_id"], params=scope("bob"))
    assert response.status_code in {404, 409}
    assert marker not in response.text


def test_live_without_configuration_uses_labelled_static_namespace(client):
    if client.get("/api/status").json()["model"]["ready"]:
        pytest.skip(
            "Live configuration exists; do not make a paid request in this static suite"
        )
    result = chat(client, mode="live")
    assert result["requested_mode"] == "live" and result["effective_mode"] == "static"
    assert result["fallback"] and "no LLM" in result["notice"]
    assert result["session_id"].startswith("demo_s_")
    assert result["evidence"]["simulated"] is True


def test_model_tool_call_executes_real_python_and_is_saved(client):
    result = chat(client, action="add_numbers", arguments={"a": 17, "b": 25})
    assert "42" in result["content"]
    assert any(
        tool["tool_name"] == "add_numbers" and not tool.get("tool_call_error")
        for tool in result["tools"]
    )
    saved = client.get("/api/evidence/" + result["session_id"], params=scope()).json()
    assert "add_numbers" in json.dumps(saved)


@pytest.mark.parametrize("rows", [500, 1500])
def test_large_tool_output_is_recoverable_after_reopen_and_removed_on_delete(
    client, rows
):
    result = chat(client, action="make_report", arguments={"rows": rows})
    assert result["status"] == "COMPLETED"
    report_tool = next(
        tool for tool in result["tools"] if tool["tool_name"] == "make_report"
    )
    assert 'stored="false"' not in report_tool["result"]
    assert '<result id="res_' in report_tool["result"]
    assert len(report_tool["result"]) < 5000
    fresh = Runtime()
    try:
        fresh.components("static")
        database = fresh.db("static")
        saved = database.get_tool_results_for_session(result["session_id"])
        assert len(saved) == 1
        row = saved[0]
        store = fresh._stores["static"]
        expected = make_report(rows)
        assert store.payload(row["result_id"]).encode("utf-8") == expected.encode(
            "utf-8"
        )
        assert row["size_bytes"] == len(expected.encode("utf-8"))
        assert (
            expected.splitlines()[-1]
            in store.read(row["result_id"], start_line=rows).text
        )
        assert (
            client.delete(
                "/api/sessions/" + result["session_id"], params=scope()
            ).status_code
            == 200
        )
        assert database.get_tool_result(row["result_id"]) is None
        assert store.fs.backend.read(row["namespace"], row["path"]) is None
    finally:
        fresh.close()


def test_memory_and_learning_are_durable_and_owner_scoped(client):
    marker = "note" + uuid4().hex
    chat(client, action="remember_note", arguments={"note": marker})
    assert marker in chat(client, action="recall_notes")["content"]
    assert marker not in chat(client, user_id="bob", action="recall_notes")["content"]
    rule = "rule" + uuid4().hex
    chat(client, action="save_learning", arguments={"rule": rule})
    assert (
        rule
        in chat(client, action="find_learning", arguments={"query": rule})["content"]
    )
    assert (
        rule
        not in chat(
            client, user_id="bob", action="find_learning", arguments={"query": rule}
        )["content"]
    )


def test_stream_has_incremental_content_and_final_persisted_run(client):
    response = client.post(
        "/api/chat", json={**scope(), "message": "Streaming test", "stream": True}
    )
    assert response.status_code == 200
    events = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert not any(event["type"] == "error" for event in events), events
    chunks = [event["text"] for event in events if event["type"] == "content"]
    result = next(event for event in events if event["type"] == "result")
    assert len(chunks) > 1 and "".join(chunks) == result["content"]
    assert (
        client.get("/api/sessions/" + result["session_id"], params=scope()).status_code
        == 200
    )


@pytest.mark.parametrize("approved", [True, False])
def test_approval_pause_reopen_resume_and_repeated_resume_denial(client, approved):
    result = chat(client, action="publish_report")
    assert result["status"] == "PAUSED" and result["requirements"]
    saved = client.get("/api/sessions/" + result["session_id"], params=scope())
    assert saved.status_code == 200
    body = {
        "user_id": "alice",
        "mode": "static",
        "session_id": result["session_id"],
        "run_id": result["run_id"],
        "approved": approved,
    }
    foreign = client.post("/api/continue", json={**body, "user_id": "bob"})
    assert foreign.status_code in {404, 409}
    resumed = post(client, "/api/continue", body)
    assert resumed["status"] == "COMPLETED"
    if approved:
        assert "Approved the fictional report" in resumed["content"]
    else:
        assert "Approved the fictional report" not in resumed["content"]
    assert client.post("/api/continue", json=body).status_code == 409


def test_team_performs_real_agno_member_delegation(client):
    result = chat(
        client, component="team", message="Draft and review a small synthetic plan."
    )
    assert result["status"] == "COMPLETED"
    assert len(result["evidence"]["run"].get("member_responses") or []) >= 2


def test_workflow_runs_both_steps(client):
    result = chat(
        client, component="workflow", message="Draft and review this synthetic example."
    )
    assert result["status"] == "COMPLETED"
    steps = json.dumps(result["evidence"]["run"].get("step_results") or [])
    assert "Draft" in steps and "Review" in steps


@pytest.mark.parametrize("mode", ["static", "live"])
def test_documents_replace_search_and_owner_isolation(client, mode):
    status = client.get("/api/status").json()
    if (
        mode == "live"
        and status["embeddings"]["synthetic"] is False
        and os.getenv("AGNO_DEMO_ALLOW_EMBEDDING_TESTS") != "1"
    ):
        pytest.skip("Explicit real embedding request opt-in required")
    name = "owned-" + uuid4().hex
    token = "token" + uuid4().hex
    document = {
        "user_id": "alice",
        "mode": mode,
        "name": name,
        "content": token + " synthetic oracle note",
    }
    inserted = post(client, "/api/documents", document)
    assert inserted["stored"]
    expected_mode = "live" if mode == "live" and status["model"]["ready"] else "static"
    expected_synthetic = expected_mode == "static" or status["embeddings"]["synthetic"]
    assert inserted["synthetic_embeddings"] is expected_synthetic
    assert inserted["effective_mode"] == expected_mode
    search = {"user_id": "alice", "mode": mode, "query": token}
    found = post(client, "/api/search", search)
    assert found["synthetic_embeddings"] is expected_synthetic
    assert any(row["name"] == name for row in found["results"])
    assert name not in json.dumps(
        post(client, "/api/search", {**search, "user_id": "bob"})
    )
    new_token = "updated" + uuid4().hex
    post(client, "/api/documents", {**document, "content": new_token})
    found = post(client, "/api/search", {**search, "query": new_token})
    assert sum(row["name"] == name for row in found["results"]) == 1


@pytest.mark.parametrize("case", ["json_reopen", "async_storage", "concurrency"])
def test_explicit_storage_checks(client, case):
    result = post(client, "/api/checks", {"mode": "static", "case": case})
    assert result["status"] == "PASS", result
    assert result["live_llm_validation"] == "NOT RUN"


def test_background_owner_scope_and_saved_run(client):
    job = post(
        client, "/api/background", {**scope(), "message": "Background static test"}
    )
    assert job["durable_queue"] is False
    assert (
        client.get(
            "/api/jobs/" + job["job_id"], params={"user_id": "bob", "mode": "static"}
        ).status_code
        == 404
    )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        state = client.get(
            "/api/jobs/" + job["job_id"], params={"user_id": "alice", "mode": "static"}
        ).json()
        if state["status"] in {"completed", "failed"}:
            break
        time.sleep(0.1)
    assert state["status"] == "completed", state
    assert (
        client.get(
            "/api/sessions/" + state["result"]["session_id"], params=scope()
        ).status_code
        == 200
    )


def test_rename_and_owner_scoped_delete(client):
    result = chat(client)
    path = "/api/sessions/" + result["session_id"]
    assert (
        client.patch(path, json={**scope(), "name": "Renamed static test"}).status_code
        == 200
    )
    assert "Renamed static test" in client.get(path, params=scope()).text
    assert client.delete(path, params=scope("bob")).status_code in {404, 409}
    assert client.delete(path, params=scope()).status_code == 200
    assert client.get(path, params=scope()).status_code in {404, 409}


def test_scenarios_remain_manual_and_inventory_is_owned(client):
    cases = client.get("/api/scenarios").json()
    assert len(cases) == 20 and all(case["status"] == "NOT RUN" for case in cases)
    inventory = client.get("/api/inventory").json()
    assert inventory["tables"]
    assert all(
        row["name"].startswith(inventory["prefix"] + "_") for row in inventory["tables"]
    )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_repeated_metrics_refresh_accepts_existing_oracle_date_values(
    client, asynchronous
):
    runtime = client.app.state.runtime
    if not asynchronous:
        database = runtime.db("static")
        assert database.calculate_metrics() is not None
        assert database.calculate_metrics() is not None
        return

    async def exercise():
        import inspect
        from agno.db.oracle import AsyncOracleDb
        from sqlalchemy.engine import make_url
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(
            make_url(runtime._url).set(drivername="oracle+oracledb_async"),
            hide_parameters=True,
        )
        accepted = inspect.signature(AsyncOracleDb).parameters
        overrides = {
            key: value
            for key, value in runtime.settings.tables("static").items()
            if key in accepted
        }
        database = AsyncOracleDb(db_engine=engine, create_schema=False, **overrides)
        try:
            assert await database.calculate_metrics() is not None
            assert await database.calculate_metrics() is not None
        finally:
            await database.close()

    asyncio.run(exercise())


def test_text_index_freshness_uses_positive_matches_not_ranked_candidates(client):
    from hashlib import sha256
    from sqlalchemy import bindparam, func, literal_column, select

    runtime = client.app.state.runtime
    vector = runtime.vector("static")
    marker = "freshness" + uuid4().hex
    old_token, new_token = marker + "oldtoken", marker + "newtoken"
    body = {**scope(), "name": marker + ".txt", "content": old_token}

    def matching_ids(token):
        statement = select(vector.table.c.id).where(
            vector.table.c.name == body["name"],
            vector.table.c.user_id == "alice",
            func.contains(
                vector.table.c.content,
                bindparam("ctx_query", vector._build_ctx_query(token)),
                literal_column("1"),
            )
            > 0,
        )
        with runtime.engine.connect() as connection:
            return list(connection.execute(statement).scalars())

    try:
        post(client, "/api/documents", body)
        vector._create_text_index()
        before = matching_ids(old_token)
        assert len(before) == 1
        post(client, "/api/documents", {**body, "content": new_token})
        assert matching_ids(old_token) == []
        assert matching_ids(new_token) == before
        hybrid = post(
            client,
            "/api/search",
            {**scope(), "query": new_token, "search_type": "hybrid", "limit": 20},
        )
        assert any(
            row["name"] == body["name"] and row["content"] == new_token
            for row in hybrid["results"]
        )
    finally:
        vector.delete_by_content_id(
            sha256(body["name"].encode()).hexdigest(), user_id="alice"
        )
