"""Shared FastAPI dependencies."""

from __future__ import annotations

import httpx
from fastapi import Request

from .settings import Settings


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_http(request: Request) -> httpx.AsyncClient:
    return request.app.state.http

