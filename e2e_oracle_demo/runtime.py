"""Actual Agno components and isolated Oracle persistence for both demo modes."""

import json
import math
import re
import threading
from datetime import datetime, timezone
from hashlib import sha256
from uuid import uuid4

from sqlalchemy import create_engine, event, inspect, select, text

from agno.agent import Agent
from agno.db.base import SessionType
from agno.db.oracle import OracleDb
from agno.db.schemas.memory import UserMemory
from agno.fs import FileSystem
from agno.fs.db import DbFileSystem
from agno.offload import ResultStore
from agno.run import RunContext
from agno.team import Team
from agno.tools import tool
from agno.workflow import Step, Workflow

from . import config
from .config import (
    ConfigurationError,
    Settings,
    load_provider,
    read_database_url,
    resolve_mode,
)
from .models import (
    STATIC_REQUEST,
    StaticEmbedder,
    StaticModel,
    build_live_embedder,
    build_live_model,
)


class DemoError(RuntimeError):
    pass


@tool(requires_confirmation=True)
def publish_report() -> str:
    """Approve a fictional report. This never publishes to an external service."""
    return "Approved the fictional report. No external publication occurred."


def add_numbers(a: float, b: float) -> str:
    """Add two finite numbers using real Python arithmetic."""
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("Use finite numbers.")
    return str(a + b)


def make_report(rows: int = 500) -> str:
    """Produce a synthetic inventory report for the offloading boundary check."""
    if not 1 <= rows <= 1500:
        raise ValueError("Choose between 1 and 1500 rows.")
    return "\n".join(
        f"item-{i:04}: synthetic inventory data, stock={i % 17}" for i in range(rows)
    )


class Runtime:
    def __init__(self):
        self.settings = Settings.load()
        self.provider = load_provider()
        self._url = read_database_url(self.settings)
        self.engine = create_engine(
            self._url,
            pool_pre_ping=True,
            hide_parameters=True,
            isolation_level="READ COMMITTED",
            connect_args={"tcp_connect_timeout": 8},
        )
        self._lock = threading.RLock()
        self._session_locks = {}
        self._components = {}
        self._dbs = {}
        self._vectors = {}
        self._stores = {}
        self._closed = False
        try:
            with self.engine.connect() as connection:
                row = connection.execute(
                    text(
                        "SELECT USER, SYS_CONTEXT('USERENV','CON_NAME'), SYS_CONTEXT('USERENV','CURRENT_SCHEMA') FROM dual"
                    )
                ).one()
                version = connection.connection.driver_connection.version
            if tuple(row) != (
                self.settings.expected_user,
                self.settings.expected_container,
                self.settings.expected_user,
            ):
                raise ConfigurationError(
                    "Connected Oracle identity does not match this deployment; no demo tables were created."
                )
            self.identity = {
                "user": row[0],
                "container": row[1],
                "schema": row[2],
                "version": version,
            }
            self._claim_namespace()
            self._existing_other = (
                set(inspect(self.engine).get_table_names()) - self.owned_names()
            )
            event.listen(self.engine, "before_cursor_execute", self._protect_existing)
        except Exception:
            self.engine.dispose()
            raise

    def owned_names(self):
        return set(
            json.loads((config.STATE_DIR / "ownership.json").read_text())["tables"]
        )

    def _claim_namespace(self):
        names = set(self.settings.tables("static").values()) | set(
            self.settings.tables("live").values()
        )
        names |= {
            f"{self.settings.prefix}_{mode}_{suffix}"
            for mode in ("s", "l")
            for suffix in ("results", "payloads")
        }
        path = config.STATE_DIR / "ownership.json"
        identity = {k: self.identity[k] for k in ("user", "container", "schema")}
        if path.exists():
            stored = json.loads(path.read_text())
            if (
                stored.get("prefix") != self.settings.prefix
                or stored.get("identity") != identity
                or not names.issubset(stored["tables"])
            ):
                raise ConfigurationError(
                    "Saved namespace ownership does not match this deployment."
                )
            if any(
                not re.fullmatch(
                    re.escape(self.settings.prefix)
                    + r"_[sl]_(\d{2}|results|payloads|v[0-9a-f]{8})",
                    name,
                )
                for name in stored["tables"]
            ):
                raise ConfigurationError(
                    "Unexpected object in the namespace ownership file."
                )
        else:
            if names & set(inspect(self.engine).get_table_names()):
                raise ConfigurationError(
                    "Proposed demo tables already exist without an ownership record; refusing reuse."
                )
            config.write_private(
                path,
                json.dumps(
                    {
                        "prefix": self.settings.prefix,
                        "identity": identity,
                        "tables": sorted(names),
                    },
                    indent=2,
                ),
            )

    def _protect_existing(
        self, connection, cursor, statement, parameters, context, executemany
    ):
        if re.match(
            r"^\s*(DROP|TRUNCATE|ALTER|INSERT|UPDATE|DELETE|MERGE)\b", statement, re.I
        ):
            for name in self._existing_other:
                if re.search(
                    r"(?<![\w$#])" + re.escape(name) + r"(?![\w$#])", statement, re.I
                ):
                    raise DemoError(
                        "Refusing a change to a pre-existing non-demo Oracle table."
                    )

    def db(self, mode):
        with self._lock:
            if mode not in self._dbs:
                database = OracleDb(
                    db_engine=self.engine,
                    create_schema=False,
                    id=f"{self.settings.prefix}-{mode}",
                    **self.settings.tables(mode),
                )
                database.tool_results_table_name = (
                    f"{self.settings.prefix}_{mode[0]}_results"
                )
                self._dbs[mode] = database
            return self._dbs[mode]

    def mode(self, requested):
        return resolve_mode(requested, self.provider)

    @staticmethod
    def component_id(mode, component):
        return f"{mode}-{component}"

    def status(self):
        with self.engine.connect() as connection:
            connection.execute(text("SELECT 1 FROM dual"))
        return {
            "database": {
                "connected": True,
                "identity": self.identity,
                "prefix": self.settings.prefix,
            },
            "model": self.provider.status(),
            "embeddings": self.provider.embedding_status(),
            "static": {"label": "Static simulation: real Agno + Oracle, no LLM"},
            "capabilities": {
                "chat": True,
                "tools": True,
                "team": True,
                "workflow": True,
                "streaming": True,
                "background": "in-process, not durable queue recovery",
                "paid_requests_automatic": False,
            },
        }

    def components(self, mode):
        with self._lock:
            if mode in self._components:
                return self._components[mode]
            database = self.db(mode)
            backend = DbFileSystem(
                db=database, table_name=f"{self.settings.prefix}_{mode[0]}_payloads"
            )
            filesystem = FileSystem(
                backend=backend, namespace="tool-results", max_file_bytes=8_000_000
            )
            store = ResultStore(db=database, fs=filesystem, threshold_chars=16000)
            database.tool_result_filesystems = [filesystem]
            self._stores[mode] = store

            def remember_note(note: str, run_context: RunContext) -> str:
                """Save the current user's note in Oracle memory."""
                result = database.upsert_user_memory(
                    UserMemory(
                        memory=note,
                        memory_id=uuid4().hex,
                        user_id=run_context.user_id,
                        topics=["demo"],
                    )
                )
                if result is None:
                    raise DemoError("Oracle did not save the note.")
                return "Saved note in Oracle: " + note

            def recall_notes(run_context: RunContext) -> str:
                """Read only the current user's Oracle notes."""
                rows = database.get_user_memories(user_id=run_context.user_id)
                if isinstance(rows, tuple):
                    rows = rows[0]
                return json.dumps([row.memory for row in rows])

            def save_learning(rule: str, run_context: RunContext) -> str:
                """Save a rule owned by the current user in Oracle."""
                key = uuid4().hex
                database.upsert_learning(
                    id=key,
                    learning_type="demo",
                    content={"rule": rule},
                    user_id=run_context.user_id,
                )
                if database.get_learning_by_id(key) is None:
                    raise DemoError("Oracle did not save the rule.")
                return "Saved learning in Oracle: " + rule

            def find_learning(query: str, run_context: RunContext) -> str:
                """Search only the current user's Oracle learnings."""
                return json.dumps(
                    database.search_learnings(query=query, user_id=run_context.user_id),
                    default=str,
                )

            def search_documents(query: str, run_context: RunContext) -> str:
                """Search the current user's documents using this mode's vector space."""
                rows = self.vector(mode).search(query, user_id=run_context.user_id)
                return json.dumps([row.to_dict() for row in rows], default=str)

            def model(role):
                return (
                    StaticModel(role_name=role)
                    if mode == "static"
                    else build_live_model(self.provider)
                )

            tools = [
                add_numbers,
                make_report,
                remember_note,
                recall_notes,
                save_learning,
                find_learning,
            ]
            if mode == "static" or self.provider.embeddings_ready:
                tools.append(search_documents)
            tools.append(publish_report)

            agent = Agent(
                id=self.component_id(mode, "agent"),
                name=f"{mode.title()} Oracle assistant",
                model=model("Assistant"),
                db=database,
                add_history_to_context=True,
                num_history_runs=10,
                telemetry=False,
                offload_tool_results=store,
                tool_call_limit=5,
                markdown=False,
                tools=tools,
                instructions=[
                    "Answer the user's own request. Use tools only when appropriate.",
                    "Do not claim an action succeeded without its tool result.",
                    "publish_report requires approval and never publishes externally.",
                ],
            )
            worker = Agent(
                id="worker",
                name="Worker",
                model=model("Draft worker"),
                db=database,
                instructions="Draft a useful response to the supplied task.",
                telemetry=False,
            )
            reviewer = Agent(
                id="reviewer",
                name="Reviewer",
                model=model("Reviewer"),
                db=database,
                instructions="Review the supplied draft, correct errors and provide a final answer.",
                telemetry=False,
            )
            team = Team(
                id=self.component_id(mode, "team"),
                name=f"{mode.title()} worker and reviewer",
                model=model("Coordinator"),
                members=[worker, reviewer],
                db=database,
                telemetry=False,
                instructions="Delegate drafting to worker and review to reviewer, then summarize the result.",
            )
            workflow = Workflow(
                id=self.component_id(mode, "workflow"),
                name=f"{mode.title()} draft and review",
                steps=[
                    Step(name="Draft", agent=worker),
                    Step(name="Review", agent=reviewer),
                ],
                db=database,
                telemetry=False,
            )
            self._components[mode] = {
                "agent": agent,
                "team": team,
                "workflow": workflow,
            }
            return self._components[mode]

    def vector(self, mode):
        with self._lock:
            if mode in self._vectors:
                return self._vectors[mode]
            if int(self.identity["version"].split(".")[0]) < 23:
                raise DemoError("Oracle vector search requires Database 23ai or newer.")
            from agno.vectordb.oracle import OracleVector

            embedder = (
                StaticEmbedder()
                if mode == "static"
                else build_live_embedder(self.provider)
            )
            identity = getattr(
                embedder, "space_identity", f"{embedder.id}:{embedder.dimensions}:local"
            )
            suffix = sha256(identity.encode()).hexdigest()[:8]
            name = f"{self.settings.prefix}_{mode[0]}_v{suffix}"
            path = config.STATE_DIR / "ownership.json"
            owned = json.loads(path.read_text())
            if name not in owned["tables"]:
                if inspect(self.engine).has_table(name):
                    raise DemoError(
                        "The requested vector table exists without demo ownership; refusing reuse."
                    )
                owned["tables"].append(name)
                temporary = path.with_name("ownership-" + uuid4().hex + ".json")
                config.write_private(temporary, json.dumps(owned, indent=2))
                temporary.replace(path)
            vector = OracleVector(
                table_name=name,
                db=self.db(mode),
                embedder=embedder,
                create_schema=False,
            )
            vector.create()
            self._vectors[mode] = vector
            return vector

    @staticmethod
    def session_prefix(mode, user_id, component):
        if (
            user_id not in {"alice", "bob"}
            or mode not in {"static", "live"}
            or component not in {"agent", "team", "workflow"}
        ):
            raise DemoError("Invalid user, mode or component.")
        return f"demo_{mode[0]}_{user_id}_{component}_"

    def owned_session(self, session_id, user_id, mode, component):
        prefix = self.session_prefix(mode, user_id, component)
        if not session_id.startswith(prefix) or not re.fullmatch(
            re.escape(prefix) + r"[a-f0-9]{32}", session_id
        ):
            raise DemoError("Session not found for this user, mode and component.")
        row = self.db(mode).get_session(
            session_id,
            session_type=SessionType(component),
            user_id=user_id,
            deserialize=False,
        )
        if row is not None and row.get("user_id") != user_id:
            raise DemoError("Session owner mismatch.")
        return row

    def sessions(self, user_id, requested, component):
        mode = self.mode(requested)["effective_mode"]
        rows = self.db(mode).get_sessions(
            session_type=SessionType(component),
            component_id=self.component_id(mode, component),
            user_id=user_id,
            include_runs=False,
            deserialize=False,
            limit=100,
        )
        if isinstance(rows, tuple):
            rows = rows[0]
        return [
            {
                "session_id": row["session_id"],
                "name": (row.get("session_data") or {}).get("session_name")
                or row["session_id"],
                "updated_at": row.get("updated_at"),
            }
            for row in rows
        ]

    def session(self, session_id, user_id, requested, component):
        mode = self.mode(requested)["effective_mode"]
        row = self.owned_session(session_id, user_id, mode, component)
        if row is None:
            raise DemoError("Session not found.")
        messages = []
        for run in row.get("runs") or []:
            if run.get("parent_run_id"):
                continue
            if run.get("input"):
                value = run["input"]
                messages.append(
                    {
                        "role": "user",
                        "content": str(value.get("input_content", value))
                        if isinstance(value, dict)
                        else str(value),
                    }
                )
            if run.get("content"):
                messages.append({"role": "assistant", "content": str(run["content"])})
        return {**row, "messages": messages, "effective_mode": mode}

    def chat(self, body, on_chunk=None):
        provenance = self.mode(body["mode"])
        mode = provenance["effective_mode"]
        component = body.get("component", "agent")
        user_id = body["user_id"]
        session_id = body.get("session_id")
        if session_id:
            if self.owned_session(session_id, user_id, mode, component) is None:
                raise DemoError("Session not found. Start a new session.")
        else:
            session_id = self.session_prefix(mode, user_id, component) + uuid4().hex
        with self._lock:
            lock = self._session_locks.setdefault(session_id, threading.Lock())
        with lock:
            token = STATIC_REQUEST.set(
                {
                    "action": body.get("action") or "auto",
                    "arguments": body.get("arguments") or {},
                }
            )
            try:
                runner = self.components(mode)[component]
                metadata = {**provenance, "simulated": mode == "static", "demo": True}
                if on_chunk and component != "workflow":
                    result = None
                    for item in runner.run(
                        body["message"],
                        user_id=user_id,
                        session_id=session_id,
                        metadata=metadata,
                        stream=True,
                        stream_events=True,
                        yield_run_output=True,
                    ):
                        if getattr(item, "event", "") in {"RunError", "TeamRunError"}:
                            raise DemoError(
                                "The model/provider failed during streaming. "
                                "Inspect the saved error run in Oracle. "
                                "No static response was substituted."
                            )
                        if not hasattr(item, "event") and hasattr(item, "run_id"):
                            result = item
                        elif getattr(item, "event", "") == (
                            "TeamRunContent" if component == "team" else "RunContent"
                        ) and getattr(item, "content", None):
                            on_chunk(str(item.content))
                    if result is None:
                        raise DemoError(
                            "Agno did not yield the final saved run for this stream."
                        )
                else:
                    result = runner.run(
                        body["message"],
                        user_id=user_id,
                        session_id=session_id,
                        metadata=metadata,
                        stream=False,
                    )
                response = self.result(result, provenance, session_id)
                if on_chunk and component == "workflow":
                    for offset in range(0, len(response["content"]), 60):
                        on_chunk(response["content"][offset : offset + 60])
                response["evidence"]["streaming"] = (
                    "Agno streamed model events"
                    if on_chunk and component != "workflow"
                    else "nonstreaming; workflow SSE frames completed content"
                )
                return response
            finally:
                STATIC_REQUEST.reset(token)

    @staticmethod
    def result(result, provenance, session_id):
        data = result.to_dict()
        status = getattr(result.status, "value", str(result.status))
        if "error" in status.lower():
            raise DemoError(
                "Agno reported a failed run. No static replacement was substituted; inspect the saved run locally."
            )
        return {
            **provenance,
            "session_id": session_id,
            "run_id": result.run_id,
            "content": str(result.content or ""),
            "status": status,
            "tools": data.get("tools") or [],
            "requirements": data.get("requirements") or [],
            "evidence": {
                "run": data,
                "database": "Oracle",
                "simulated": provenance["effective_mode"] == "static",
                "streaming": "nonstreaming",
            },
        }

    def continue_run(self, body):
        provenance = self.mode(body["mode"])
        mode = provenance["effective_mode"]
        if (
            self.owned_session(body["session_id"], body["user_id"], mode, "agent")
            is None
        ):
            raise DemoError("Session not found.")
        run = self.db(mode).get_run(body["run_id"])
        if (
            run is None
            or run.session_id != body["session_id"]
            or run.agent_id != self.component_id(mode, "agent")
            or not run.is_paused
        ):
            raise DemoError(
                "This agent run is not awaiting approval in the selected session."
            )
        for requirement in run.requirements or []:
            if body["approved"]:
                requirement.confirm()
            else:
                requirement.reject("Rejected in the local demo")
        token = STATIC_REQUEST.set({"action": "auto"})
        try:
            result = self.components(mode)["agent"].continue_run(
                run_id=run.run_id,
                requirements=run.requirements,
                session_id=run.session_id,
                user_id=body["user_id"],
            )
            return self.result(result, provenance, run.session_id)
        finally:
            STATIC_REQUEST.reset(token)

    def ingest(self, body):
        from agno.knowledge.document import Document

        provenance = self.mode(body["mode"])
        mode = provenance["effective_mode"]
        base_id = sha256(body["name"].encode()).hexdigest()
        content_hash = sha256(
            json.dumps([body["user_id"], body["name"]]).encode()
        ).hexdigest()
        vector = self.vector(mode)
        vector.upsert(
            content_hash,
            [
                Document(
                    id=base_id,
                    name=body["name"],
                    content=body["content"],
                    content_id=base_id,
                    meta_data={
                        "mode": mode,
                        "synthetic_embeddings": vector.embedder.synthetic,
                        "embedding_model": vector.embedder.id,
                        "embedding_dimensions": vector.embedder.dimensions,
                    },
                )
            ],
            user_id=body["user_id"],
        )
        with self.engine.connect() as connection:
            exists = (
                connection.execute(
                    select(vector.table.c.id).where(
                        vector.table.c.user_id == body["user_id"],
                        vector.table.c.content_hash == content_hash,
                    )
                ).first()
                is not None
            )
        if not exists:
            raise DemoError("Ingestion did not persist the document.")
        return {
            **provenance,
            "stored": True,
            "table": vector.table.name,
            "synthetic_embeddings": vector.embedder.synthetic,
            "embedding_model": vector.embedder.id,
            "embedding_dimensions": vector.embedder.dimensions,
        }

    def search(self, body):
        provenance = self.mode(body["mode"])
        mode = provenance["effective_mode"]
        vector = self.vector(mode)
        method = {
            "vector": vector.search,
            "keyword": vector.keyword_search,
            "hybrid": vector.hybrid_search,
        }[body["search_type"]]
        rows = method(body["query"], limit=body["limit"], user_id=body["user_id"])
        return {
            **provenance,
            "synthetic_embeddings": vector.embedder.synthetic,
            "embedding_model": vector.embedder.id,
            "embedding_dimensions": vector.embedder.dimensions,
            "results": [row.to_dict() for row in rows],
        }

    def inventory(self):
        rows = []
        with self.engine.connect() as connection:
            for name in sorted(
                self.owned_names() & set(inspect(self.engine).get_table_names())
            ):
                identifier = self.engine.dialect.identifier_preparer.quote(name)
                count = connection.execute(
                    text("SELECT COUNT(*) FROM " + identifier)
                ).scalar_one()
                rows.append({"name": name, "rows": count})
        return {
            "tables": rows,
            "identity": self.identity,
            "prefix": self.settings.prefix,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

    def close(self):
        if self._closed:
            return
        self._closed = True
        for database in self._dbs.values():
            database.Session.remove()
        self.engine.dispose()
