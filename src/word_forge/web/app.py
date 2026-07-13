"""FastAPI application behind the local Word Forge workspace."""

from __future__ import annotations

import asyncio
import re
import secrets
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from word_forge import __version__
from word_forge.database.database_manager import (
    DatabaseError,
    RevisionConflictError,
    TermNotFoundError,
)
from word_forge.lexicon import Lexicon
from word_forge.maintenance import backup_database, inspect_database, restore_database
from word_forge.web.events import EventHub
from word_forge.web.operations import OperationGate
from word_forge.web.schemas import (
    BackupCreate,
    EntryCreate,
    RelationshipCreate,
    RestoreRequest,
    SenseWrite,
    TranslationCreate,
)

_DEFAULT_ALLOWED_HOSTS = ("127.0.0.1", "localhost", "[::1]", "testserver")


def _same_origin(request: Request) -> bool:
    origin = request.headers.get("origin")
    if not origin:
        return True
    return urlsplit(origin).netloc.casefold() == request.headers.get("host", "").casefold()


def create_app(
    database_path: str | Path | None = None,
    *,
    allowed_hosts: Sequence[str] = _DEFAULT_ALLOWED_HOSTS,
    api_token: str | None = None,
) -> FastAPI:
    """Create an isolated local GUI application for one lexicon database."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.lexicon = Lexicon(database_path)
        application.state.events = EventHub()
        application.state.operation_gate = OperationGate()
        try:
            yield
        finally:
            application.state.lexicon.close()

    app = FastAPI(
        title="Word Forge",
        summary="Local multilingual living lexicon workspace",
        version=__version__,
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(allowed_hosts))

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        if request.url.path.startswith("/api/") and api_token:
            supplied_token = request.headers.get("x-word-forge-token", "")
            if not secrets.compare_digest(supplied_token, api_token):
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "invalid_session",
                        "message": "A valid Word Forge session token is required",
                    },
                )
        if request.method in {"POST", "PUT", "PATCH", "DELETE"} and not _same_origin(request):
            return JSONResponse(
                status_code=403,
                content={
                    "error": "cross_origin_request",
                    "message": "Mutation origin does not match this server",
                },
            )
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data: blob:; connect-src 'self'; "
            "font-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
        )
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(TermNotFoundError)
    async def missing_entry(_: Request, exc: TermNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={"error": "entry_not_found", "message": str(exc)},
        )

    @app.exception_handler(DatabaseError)
    async def database_failure(_: Request, exc: DatabaseError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"error": "database_failure", "message": str(exc)},
        )

    @app.exception_handler(RevisionConflictError)
    async def revision_conflict(_: Request, exc: RevisionConflictError) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={
                "error": "revision_conflict",
                "message": str(exc),
                "expected_revision": exc.expected,
                "current_revision": exc.actual,
            },
        )

    def lexicon(request: Request) -> Lexicon:
        return request.app.state.lexicon

    def events(request: Request) -> EventHub:
        return request.app.state.events

    def operation_gate(request: Request) -> OperationGate:
        return request.app.state.operation_gate

    async def run_operation(
        request: Request,
        function: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Run blocking database work while participating in restore coordination."""

        async with operation_gate(request).operation():
            return await asyncio.to_thread(function, *args, **kwargs)

    def managed_backup_directory(current: Lexicon) -> Path:
        return current.path.parent / "backups"

    @app.get("/api/health")
    async def health(request: Request) -> dict[str, object]:
        current = lexicon(request)
        return {"ok": True, "version": __version__, "database": str(current.path)}

    @app.get("/api/stats")
    async def statistics(request: Request) -> dict[str, int]:
        result = await run_operation(request, lexicon(request).statistics)
        return result.as_dict()

    @app.get("/api/entries")
    async def entries(
        request: Request,
        query: str = Query(min_length=1, max_length=512),
        language: str | None = Query(default=None, max_length=35),
        limit: int = Query(default=20, ge=1, le=1_000),
    ) -> list[dict[str, object]]:
        result = await run_operation(
            request,
            lexicon(request).find,
            query,
            language=language,
            limit=limit,
        )
        return [entry.as_dict() for entry in result]

    @app.get("/api/entry")
    async def entry(
        request: Request,
        term: str = Query(min_length=1, max_length=512),
        language: str = Query(default="en", max_length=35),
    ) -> dict[str, object]:
        result = await run_operation(request, lexicon(request).get, term, language=language)
        return result.as_dict()

    @app.post("/api/entries", status_code=201)
    async def create_entry(request: Request, payload: EntryCreate) -> dict[str, object]:
        result = await run_operation(
            request,
            lexicon(request).add,
            payload.term,
            definition=payload.definition,
            part_of_speech=payload.part_of_speech,
            language=payload.language,
            examples=payload.examples,
            source=payload.source,
            base_term=payload.base_term,
            expected_revision=payload.expected_revision,
        )
        await events(request).publish(
            {"type": "entry.changed", "term": result.term, "language": result.language}
        )
        return result.as_dict()

    @app.post("/api/relationships", status_code=201)
    async def create_relationship(
        request: Request, payload: RelationshipCreate
    ) -> dict[str, object]:
        created = await run_operation(
            request,
            lexicon(request).relate,
            payload.term,
            payload.target,
            kind=payload.kind,
        )
        await events(request).publish(
            {"type": "graph.changed", "term": payload.term, "target": payload.target}
        )
        return {"created": created}

    @app.post("/api/translations", status_code=201)
    async def create_translation(request: Request, payload: TranslationCreate) -> dict[str, bool]:
        await run_operation(
            request,
            lexicon(request).translate,
            payload.term,
            source_language=payload.source_language,
            target_language=payload.target_language,
            target_term=payload.target_term,
            source=payload.source,
            bidirectional=payload.bidirectional,
        )
        await events(request).publish(
            {"type": "graph.changed", "term": payload.term, "target": payload.target_term}
        )
        return {"created": True}

    @app.post("/api/senses", status_code=201)
    async def write_sense(request: Request, payload: SenseWrite) -> dict[str, object]:
        result = await run_operation(
            request,
            lexicon(request).add_sense,
            payload.term,
            payload.definition,
            language=payload.language,
            part_of_speech=payload.part_of_speech,
            domain=payload.domain,
            register=payload.register_name,
            source=payload.source,
            confidence=payload.confidence,
            review_state=payload.review_state,
            sense_id=payload.sense_id,
            expected_revision=payload.expected_revision,
        )
        await events(request).publish(
            {"type": "sense.changed", "term": payload.term, "language": payload.language}
        )
        return result.as_dict()

    @app.delete("/api/senses/{sense_id}")
    async def delete_sense(
        request: Request,
        sense_id: str,
        expected_revision: int | None = Query(default=None, ge=1),
    ) -> dict[str, bool]:
        deleted = await run_operation(
            request,
            lexicon(request).remove_sense,
            sense_id,
            expected_revision=expected_revision,
        )
        if deleted:
            await events(request).publish({"type": "sense.changed", "sense_id": sense_id})
        return {"deleted": deleted}

    @app.get("/api/network")
    async def network(
        request: Request,
        query: str = Query(default="", max_length=512),
        language: str | None = Query(default=None, max_length=35),
        limit: int = Query(default=400, ge=1, le=2_000),
    ) -> dict[str, object]:
        result = await run_operation(
            request,
            lexicon(request).network,
            query=query,
            language=language,
            limit=limit,
        )
        return result.as_dict()

    @app.get("/api/maintenance/integrity")
    async def database_integrity(request: Request) -> dict[str, object]:
        current = lexicon(request)
        report = await run_operation(request, inspect_database, current.path)
        return report.as_dict()

    @app.get("/api/maintenance/backups")
    async def list_backups(request: Request) -> list[dict[str, object]]:
        directory = managed_backup_directory(lexicon(request))
        if not directory.is_dir():
            return []
        snapshots: list[dict[str, object]] = []
        for path in sorted(directory.glob("*.sqlite"), reverse=True):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                metadata = path.stat()
            except OSError:
                continue
            snapshots.append(
                {
                    "name": path.name,
                    "size": metadata.st_size,
                    "modified_at": metadata.st_mtime,
                }
            )
        return snapshots

    @app.post("/api/maintenance/backups", status_code=201)
    async def create_backup(request: Request, payload: BackupCreate) -> dict[str, object]:
        current = lexicon(request)
        directory = managed_backup_directory(current)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        label = re.sub(r"[^A-Za-z0-9._-]+", "-", payload.label.strip()).strip("-._")
        suffix = f"-{label}" if label else ""
        destination = directory / f"word-forge-{timestamp}{suffix}.sqlite"
        created = await run_operation(request, backup_database, current.path, destination)
        metadata = created.stat()
        return {
            "name": created.name,
            "size": metadata.st_size,
            "modified_at": metadata.st_mtime,
        }

    @app.post("/api/maintenance/restore")
    async def restore_backup(request: Request, payload: RestoreRequest) -> dict[str, object]:
        if payload.confirmation != "RESTORE":
            raise HTTPException(status_code=400, detail="Type RESTORE to confirm replacement")
        if Path(payload.backup_name).name != payload.backup_name:
            raise HTTPException(status_code=400, detail="Invalid managed backup name")
        current = lexicon(request)
        backup_directory = managed_backup_directory(current).resolve()
        candidate = backup_directory / payload.backup_name
        if candidate.is_symlink():
            raise HTTPException(status_code=400, detail="Managed backup cannot be a symbolic link")
        try:
            backup_path = candidate.resolve(strict=True)
        except OSError:
            raise HTTPException(status_code=404, detail="Managed backup was not found") from None
        if (
            backup_path.parent != backup_directory
            or not backup_path.is_file()
            or backup_path.suffix != ".sqlite"
        ):
            raise HTTPException(status_code=404, detail="Managed backup was not found")
        database_file = current.path
        async with operation_gate(request).exclusive():
            current.close()
            try:
                await asyncio.to_thread(
                    restore_database,
                    backup_path,
                    database_file,
                    force=True,
                )
            finally:
                request.app.state.lexicon = Lexicon(database_file)
        await events(request).publish({"type": "database.restored", "backup": backup_path.name})
        return {"restored": True, "backup_name": backup_path.name}

    @app.websocket("/api/events")
    async def event_stream(websocket: WebSocket) -> None:
        origin = websocket.headers.get("origin")
        host = websocket.headers.get("host", "")
        if origin and urlsplit(origin).netloc.casefold() != host.casefold():
            await websocket.close(code=1008, reason="Origin does not match server")
            return
        requested_protocols = [
            item.strip()
            for item in websocket.headers.get("sec-websocket-protocol", "").split(",")
            if item.strip()
        ]
        if api_token and (
            len(requested_protocols) < 2
            or requested_protocols[0] != "word-forge"
            or not secrets.compare_digest(requested_protocols[1], api_token)
        ):
            await websocket.close(code=1008, reason="A valid session token is required")
            return
        accepted_protocol = "word-forge" if requested_protocols else None
        await websocket.accept(subprotocol=accepted_protocol)
        hub: EventHub = websocket.app.state.events
        since_raw = websocket.query_params.get("since")
        try:
            since = max(0, int(since_raw)) if since_raw is not None else None
        except ValueError:
            await websocket.close(code=1008, reason="since must be a non-negative integer")
            return
        queue = await hub.subscribe(since=since)
        event_task: asyncio.Task[Any] | None = None
        receive_task: asyncio.Task[Any] | None = None
        try:
            await websocket.send_json(
                {"type": "ready", "version": __version__, "sequence": await hub.sequence()}
            )
            event_task = asyncio.create_task(queue.get())
            receive_task = asyncio.create_task(websocket.receive())
            while True:
                completed, _ = await asyncio.wait(
                    {event_task, receive_task}, return_when=asyncio.FIRST_COMPLETED
                )
                if receive_task in completed:
                    message = receive_task.result()
                    if message.get("type") == "websocket.disconnect":
                        break
                    receive_task = asyncio.create_task(websocket.receive())
                if event_task in completed:
                    await websocket.send_json(event_task.result())
                    event_task = asyncio.create_task(queue.get())
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        finally:
            for task in (event_task, receive_task):
                if task is not None:
                    task.cancel()
            await asyncio.gather(
                *(task for task in (event_task, receive_task) if task is not None),
                return_exceptions=True,
            )
            await hub.unsubscribe(queue)

    static_directory = Path(__file__).resolve().parent / "static"
    if static_directory.is_dir():
        app.mount("/", StaticFiles(directory=static_directory, html=True), name="workspace")

    return app


__all__ = ["create_app"]
