"""Runtime state installed on ``app.state`` during startup."""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from . import breaker as breakermod
from .docs_build import NavEntry, RenderedDoc
from .settings import ModelEntry, Settings


@dataclass(slots=True)
class WrapperRuntime:
    settings: Settings
    engine: AsyncEngine
    sessions: async_sessionmaker
    http: httpx.AsyncClient
    # Multi-page /tutorial docs site (ACS-89). Pages keyed by slug
    # ("overview", "examples/logprobs", …); nav is the editorial left-rail
    # order. Both built once at startup from wrapper/docs/*.md.
    tutorial_pages: dict[str, RenderedDoc]
    tutorial_nav: list[NavEntry]
    # Single-page plain-markdown copy of the whole tutorial, served at /llms.txt
    # for LLM consumption (generated from the same sources as the pages).
    tutorial_combined: str
    boot_time: dt.datetime
    models: dict[str, ModelEntry]
    default_model_id: str
    last_completion_at: dict[str, dt.datetime]
    key_semaphores: dict[uuid.UUID, Any]
    breakers: breakermod.BackendBreakers
    generations: dict[uuid.UUID, Any]
    # In-memory fan-out for active loom generates (ACS-148), keyed by gen_id.
    # Separate from ``generations`` because a loom generate fans out N branches
    # (LoomGenerationState) rather than one stream (GenerationState).
    loom_generations: dict[uuid.UUID, Any] = field(default_factory=dict)
    scheduler: Any | None = None


def install_runtime(app: FastAPI, runtime: WrapperRuntime) -> None:
    """Attach typed runtime plus legacy ``app.state`` aliases.

    The aliases preserve the existing route/test surface while giving new code
    a single object to depend on.
    """
    app.state.runtime = runtime
    app.state.settings = runtime.settings
    app.state.engine = runtime.engine
    app.state.sessions = runtime.sessions
    app.state.http = runtime.http
    app.state.tutorial_pages = runtime.tutorial_pages
    app.state.tutorial_nav = runtime.tutorial_nav
    app.state.tutorial_combined = runtime.tutorial_combined
    app.state.boot_time = runtime.boot_time
    app.state.models = runtime.models
    app.state.default_model_id = runtime.default_model_id
    app.state.last_completion_at = runtime.last_completion_at
    app.state.key_semaphores = runtime.key_semaphores
    app.state.breakers = runtime.breakers
    app.state.generations = runtime.generations
    app.state.loom_generations = runtime.loom_generations
    app.state.scheduler = runtime.scheduler


def set_scheduler(app: FastAPI, scheduler: Any | None) -> None:
    app.state.scheduler = scheduler
    runtime = getattr(app.state, "runtime", None)
    if runtime is not None:
        runtime.scheduler = scheduler
