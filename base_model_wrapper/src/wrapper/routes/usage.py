"""Per-user usage route: the usage tab + monthly CSV usage report."""

from __future__ import annotations

from pathlib import Path

import csv
import io

from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from .. import web_auth as webauth
from ..db import get_session
from ..models import User
from ..services.usage_reports import (
    build_usage_context,
    build_usage_report,
    default_report_month,
    usage_report_months,
)

_TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

router = APIRouter()


# CSV label for request-log rows with no recorded model (issue #10, story 13) —
# keeps the file's totals reconcilable with the /usage page.
_UNKNOWN_MODEL_LABEL = "(unknown)"


# --- usage tab ---------------------------------------------------------------


@router.get("/usage")
async def usage_tab(
    request: Request,
    key: str | None = None,
    user: User | None = webauth.CurrentUserDep,
    session: AsyncSession = Depends(get_session),
):
    """Per-user usage view: this-month total, 30-day chart, breakdowns, recent requests.

    The optional ``?key=<id>`` query param scopes the whole view to a single
    one of the caller's keys; absent or unrecognised, the view aggregates
    across every key (the "All keys" default).
    """
    if user is None:
        return RedirectResponse(url="/login", status_code=303)

    context = await build_usage_context(session, user, selected_key=key)
    context["report_months"] = await usage_report_months(session, user)
    context["default_report_month"] = default_report_month().strftime("%Y-%m")
    return templates.TemplateResponse(
        request,
        "usage.html",
        context,
    )


# --- monthly usage report (CSV download) --------------------------------------


@router.get("/usage/report.csv")
async def usage_report_csv(
    request: Request,
    month: str | None = None,
    user: User | None = webauth.CurrentUserDep,
    session: AsyncSession = Depends(get_session),
):
    """Download one month's per-model token usage as CSV (issue #10).

    The service picks the month (defaulting to the previous month when
    ``?month=`` is absent / unparseable / not offered); the download always
    covers *every* key the user owns — the page's ``?key=`` filter never
    scopes it — so report totals tie out to budget accounting. Data source
    and row semantics per ADR 0001 (see ``build_usage_report``).
    """
    if user is None:
        return RedirectResponse(url="/login", status_code=303)

    report = await build_usage_report(session, user, month=month)

    buf = io.StringIO(newline="")
    writer = csv.writer(buf, lineterminator="\r\n")  # CRLF per RFC 4180 / Excel
    writer.writerow(
        (
            "key_name",
            "key_prefix",
            "model",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "requests",
        )
    )
    for row in report["rows"]:
        writer.writerow(
            (
                row["key_name"] or "Unnamed key",
                row["key_prefix"],
                # Only a truly unrecorded model is "(unknown)" — an empty
                # string is a recorded (if odd) model value.
                row["model"] if row["model"] is not None else _UNKNOWN_MODEL_LABEL,
                row["prompt_tokens"],
                row["completion_tokens"],
                row["total_tokens"],
                row["requests"],
            )
        )
    # UTF-8 BOM so Excel / Sheets open the file as UTF-8 without an import
    # dialog (issue #10, story 8).
    body = b"\xef\xbb\xbf" + buf.getvalue().encode("utf-8")

    filename = f"usage-report-{report['month'].strftime('%Y-%m')}.csv"
    return Response(
        content=body,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
