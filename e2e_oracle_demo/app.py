"""Loopback-only browser API: explicit static/live provenance on every execution."""

import asyncio
import json
import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from queue import SimpleQueue
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .config import ROOT, ConfigurationError
from .models import STATIC_ACTIONS
from .runtime import DemoError, Runtime

Mode = Literal["static", "live"]
User = Literal["alice", "bob"]
Component = Literal["agent", "team", "workflow"]


class ChatRequest(BaseModel):
    user_id: User = "alice"
    mode: Mode = "static"
    component: Component = "agent"
    session_id: str | None = Field(default=None, max_length=160)
    message: str = Field(min_length=1, max_length=12000)
    stream: bool = False
    action: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_action(self):
        if self.action and self.action not in STATIC_ACTIONS:
            raise ValueError("Unknown static action.")
        if len(json.dumps(self.arguments)) > 20000:
            raise ValueError("Tool arguments are too large.")
        return self


class RenameRequest(BaseModel):
    user_id: User
    mode: Mode
    component: Component = "agent"
    name: str = Field(min_length=1, max_length=100)


class ContinueRequest(BaseModel):
    user_id: User
    mode: Mode
    session_id: str = Field(max_length=160)
    run_id: str = Field(max_length=160)
    approved: bool


class DocumentRequest(BaseModel):
    user_id: User
    mode: Mode
    name: str = Field(min_length=1, max_length=120)
    content: str = Field(min_length=1, max_length=50000)


class SearchRequest(BaseModel):
    user_id: User
    mode: Mode
    query: str = Field(min_length=1, max_length=4000)
    search_type: Literal["vector", "keyword", "hybrid"] = "vector"
    limit: int = Field(default=5, ge=1, le=20)


class IndexRequest(BaseModel):
    mode: Mode = "static"


class CheckRequest(BaseModel):
    mode: Mode = "static"
    case: Literal["json_reopen", "async_storage", "concurrency", "metrics", "bulk"]


def safe_error(error):
    if isinstance(error, (ConfigurationError, DemoError)):
        message = str(error)
    else:
        oracle_code = re.search(r"ORA-\d{5}", str(error))
        message = "Operation failed; credentials and provider details are hidden."
        if oracle_code:
            message += " Oracle code: " + oracle_code.group()
    return {"type": type(error).__name__, "message": message[:600]}


def create_app(runtime=None, factory=None):
    state = {"runtime": runtime, "error": None}
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="oracle-demo")
    jobs = {}

    @asynccontextmanager
    async def lifespan(app):
        if state["runtime"] is None:
            try:
                state["runtime"] = await asyncio.to_thread(factory or Runtime)
            except Exception as error:
                state["error"] = safe_error(error)
        app.state.runtime = state["runtime"]
        yield
        executor.shutdown(wait=True, cancel_futures=True)
        if state["runtime"] is not None:
            await asyncio.to_thread(state["runtime"].close)

    app = FastAPI(title="Agno + Oracle: static and live demo", lifespan=lifespan)
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"]
    )

    @app.middleware("http")
    async def browser_boundary(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin and origin != f"{request.url.scheme}://{request.headers.get('host')}":
            return JSONResponse(
                {"detail": "Use the page served by this application."}, status_code=403
            )
        if request.method in {"POST", "PATCH", "DELETE"}:
            if request.method != "DELETE" and not request.headers.get(
                "content-type", ""
            ).startswith("application/json"):
                return JSONResponse(
                    {"detail": "JSON content type required."}, status_code=415
                )
            if (
                request.headers.get("content-length", "").isdigit()
                and int(request.headers["content-length"]) > 100000
            ):
                return JSONResponse({"detail": "Request too large."}, status_code=413)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        )
        return response

    @app.exception_handler(DemoError)
    async def demo_error(request, error):
        return JSONResponse({"detail": safe_error(error)}, status_code=409)

    @app.exception_handler(Exception)
    async def unexpected_error(request, error):
        return JSONResponse({"detail": safe_error(error)}, status_code=500)

    def connected():
        if state["runtime"] is None:
            raise HTTPException(503, state["error"] or "Oracle is not configured.")
        return state["runtime"]

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(ROOT / "static/index.html")

    @app.get("/health")
    @app.get("/api/status")
    def status():
        if state["runtime"] is None:
            return {
                "database": {"connected": False},
                "model": {"ready": False},
                "error": state["error"],
            }
        return connected().status()

    @app.get("/api/sessions")
    def sessions(
        user_id: User = "alice", mode: Mode = "static", component: Component = "agent"
    ):
        return connected().sessions(user_id, mode, component)

    @app.get("/api/sessions/{session_id}")
    @app.get("/api/evidence/{session_id}")
    def session(
        session_id: str,
        user_id: User = "alice",
        mode: Mode = "static",
        component: Component = "agent",
    ):
        return jsonable_encoder(
            connected().session(session_id, user_id, mode, component)
        )

    @app.patch("/api/sessions/{session_id}")
    def rename(session_id: str, body: RenameRequest):
        current = connected()
        mode = current.mode(body.mode)["effective_mode"]
        if (
            current.owned_session(session_id, body.user_id, mode, body.component)
            is None
        ):
            raise HTTPException(404, "Session not found.")
        from agno.db.base import SessionType

        result = current.db(mode).rename_session(
            session_id,
            SessionType(body.component),
            body.name,
            user_id=body.user_id,
            deserialize=False,
        )
        if result is None:
            raise DemoError("Oracle did not rename the session.")
        return {"renamed": True}

    @app.delete("/api/sessions/{session_id}")
    def delete_session(
        session_id: str,
        user_id: User = "alice",
        mode: Mode = "static",
        component: Component = "agent",
    ):
        current = connected()
        effective = current.mode(mode)["effective_mode"]
        if current.owned_session(session_id, user_id, effective, component) is None:
            raise HTTPException(404, "Session not found.")
        if not current.db(effective).delete_session(session_id, user_id=user_id):
            raise DemoError("Oracle did not delete the session.")
        return {"deleted": True}

    @app.post("/api/chat")
    async def chat(body: ChatRequest):
        current = connected()
        if not body.stream:
            return jsonable_encoder(
                await asyncio.to_thread(current.chat, body.model_dump())
            )
        queue = SimpleQueue()

        def produce():
            try:
                result = current.chat(
                    body.model_dump(),
                    on_chunk=lambda chunk: queue.put(
                        {"type": "content", "text": chunk}
                    ),
                )
                queue.put({"type": "result", **jsonable_encoder(result)})
            except Exception as error:
                queue.put({"type": "error", "detail": safe_error(error)})
            finally:
                queue.put(None)

        executor.submit(produce)

        def events():
            while True:
                item = queue.get()
                if item is None:
                    break
                yield "data: " + json.dumps(item) + "\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/api/continue")
    async def continue_run(body: ContinueRequest):
        return jsonable_encoder(
            await asyncio.to_thread(connected().continue_run, body.model_dump())
        )

    @app.post("/api/documents")
    async def documents(body: DocumentRequest):
        return await asyncio.to_thread(connected().ingest, body.model_dump())

    @app.post("/api/search")
    async def search(body: SearchRequest):
        return jsonable_encoder(
            await asyncio.to_thread(connected().search, body.model_dump())
        )

    @app.post("/api/indexes")
    async def indexes(body: IndexRequest):
        current = connected()
        provenance = current.mode(body.mode)
        await asyncio.to_thread(current.vector(provenance["effective_mode"]).optimize)
        return {
            **provenance,
            "requested": True,
            "note": "Inspect search results and Oracle evidence; optimize can log index errors.",
        }

    @app.get("/api/inventory")
    def inventory():
        return connected().inventory()

    @app.get("/api/scenarios")
    def scenarios():
        from .scenarios import SCENARIOS

        return SCENARIOS

    @app.post("/api/checks")
    async def checks(body: CheckRequest):
        from .checks import run_check

        return await asyncio.to_thread(run_check, connected(), body.case, body.mode)

    @app.post("/api/background")
    def background(body: ChatRequest):
        current = connected()
        if len(jobs) >= 100:
            raise HTTPException(
                429, "Restart the demo to clear its bounded in-process job list."
            )
        job_id = uuid4().hex
        job = {
            "job_id": job_id,
            "user_id": body.user_id,
            "mode": body.mode,
            "status": "queued",
            "durable_queue": False,
            "notice": "In-process background execution; not a durable queue recovery test.",
        }
        jobs[job_id] = job

        def perform():
            job["status"] = "running"
            try:
                job["result"] = jsonable_encoder(
                    current.chat({**body.model_dump(), "stream": False})
                )
                job["status"] = "completed"
            except Exception as error:
                job.update(status="failed", error=safe_error(error))

        executor.submit(perform)
        return dict(job)

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str, user_id: User, mode: Mode):
        result = jobs.get(job_id)
        if result is None or result["user_id"] != user_id or result["mode"] != mode:
            raise HTTPException(404, "Job not found for this user and mode.")
        return result

    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
    return app
