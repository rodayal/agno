"""Explicit Oracle API scenarios. No LLM requests; no automatic paid batch."""

import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

from sqlalchemy import select

from .app import safe_error
from .runtime import DemoError


def run_check(runtime, case, requested):
    provenance = runtime.mode(requested)
    mode = provenance["effective_mode"]
    database = runtime.db(mode)
    marker = "check_" + uuid4().hex
    payload = {"nested": {"count": 2, "values": [True, None, "example"]}}
    try:
        if case == "json_reopen":
            database.upsert_learning(
                id=marker, learning_type="demo-check", content=payload, user_id="alice"
            )
            try:
                from agno.db.oracle import OracleDb

                fresh = OracleDb(
                    db_engine=runtime.engine,
                    create_schema=False,
                    **runtime.settings.tables(mode),
                )
                try:
                    row = fresh.get_learning_by_id(marker)
                    assert row is not None and row["content"] == payload
                finally:
                    fresh.Session.remove()
            finally:
                database.delete_learning(marker)
            detail = "Nested JSON survived reflection through a newly constructed Oracle adapter."
        elif case == "async_storage":

            async def exercise():
                from agno.db.oracle import AsyncOracleDb
                from sqlalchemy.engine import make_url
                from sqlalchemy.ext.asyncio import create_async_engine

                engine = create_async_engine(
                    make_url(runtime._url).set(drivername="oracle+oracledb_async"),
                    hide_parameters=True,
                )
                accepted = inspect.signature(AsyncOracleDb).parameters
                overrides = {
                    k: v
                    for k, v in runtime.settings.tables(mode).items()
                    if k in accepted
                }
                adapter = AsyncOracleDb(
                    db_engine=engine, create_schema=False, **overrides
                )
                try:
                    await adapter.upsert_learning(
                        id=marker,
                        learning_type="demo-check",
                        content=payload,
                        user_id="alice",
                    )
                    row = await adapter.get_learning_by_id(marker)
                    assert row is not None and row["content"] == payload
                finally:
                    await adapter.delete_learning(marker)
                    await adapter.close()

            asyncio.run(exercise())
            detail = "AsyncOracleDb wrote, read and removed a nested JSON record."
        elif case == "concurrency":
            if mode != "static":
                raise DemoError(
                    "Choose static mode for this isolated, no-provider concurrency check."
                )
            from agno.knowledge.document import Document

            vector = runtime.vector(mode)
            vector.insert(
                marker,
                [
                    Document(
                        id=marker,
                        content_id=marker,
                        content="Synthetic concurrency check",
                        meta_data={},
                    )
                ],
                user_id="alice",
            )
            try:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = [
                        pool.submit(vector.update_metadata, marker, patch)
                        for patch in ({"a": 1}, {"b": 2})
                    ]
                    for future in futures:
                        future.result()
                vector.update_metadata(marker, {"explicit_null": None})
                with runtime.engine.connect() as connection:
                    saved = connection.execute(
                        select(vector.table.c.meta_data).where(
                            vector.table.c.content_id == marker
                        )
                    ).scalar_one()
                assert saved == {"a": 1, "b": 2, "explicit_null": None}
            finally:
                vector.delete_by_content_id(marker, user_id="alice")
            detail = "Two independently started metadata updates survived and explicit null was retained; this is a smoke check, not exhaustive scheduling proof."
        elif case == "metrics":
            first = database.calculate_metrics()
            second = database.calculate_metrics()
            if first is None or second is None:
                raise DemoError(
                    "Metrics calculation returned no success result; inspect the known repeated-refresh issue."
                )
            detail = "Two metrics refresh calls completed."
        elif case == "bulk":
            from agno.db.base import SessionType
            from agno.session.agent import AgentSession

            ids = [marker + "_" + str(i) for i in range(1001)]
            bob = marker + "_bob"
            component_id = marker
            try:
                for session_id in ids:
                    assert (
                        database.upsert_session(
                            AgentSession(
                                session_id=session_id,
                                user_id="alice",
                                agent_id=component_id,
                            )
                        )
                        is not None
                    )
                assert (
                    database.upsert_session(
                        AgentSession(
                            session_id=bob, user_id="bob", agent_id=component_id
                        )
                    )
                    is not None
                )
                database._get_table("runs", create_table_if_not_found=True)
                rows = database.get_sessions(
                    session_type=SessionType.AGENT,
                    component_id=component_id,
                    user_id="alice",
                    include_runs=True,
                    limit=None,
                    deserialize=False,
                )
                if isinstance(rows, tuple):
                    rows = rows[0]
                assert {row["session_id"] for row in rows} == set(ids)
                database.delete_sessions(ids + [bob], user_id="alice")
                assert database.get_session(bob, user_id="bob") is not None
                assert not any(
                    database.get_session(key, user_id="alice") for key in ids
                )
            finally:
                database.delete_sessions(ids, user_id="alice")
                database.delete_session(bob, user_id="bob")
            detail = "1001 sessions hydrated and owner-scoped deletion retained Bob's control. This server result does not certify Oracle 19c."
        else:
            raise DemoError("Unknown check.")
        return {
            **provenance,
            "case": case,
            "status": "PASS",
            "detail": detail,
            "live_llm_validation": "NOT RUN",
        }
    except Exception as error:
        return {
            **provenance,
            "case": case,
            "status": "FAIL",
            "error": safe_error(error),
            "live_llm_validation": "NOT RUN",
        }
