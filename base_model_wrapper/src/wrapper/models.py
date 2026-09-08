"""SQLAlchemy ORM models matching the schema in wrapper-implementation-plan.md."""

from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import (
    ARRAY,
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import INET, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


#: Every value ``users.status`` may hold. Deliberately NOT a DB CHECK: the
#: column is plain ``Text`` (migration 0005) and every gate in the app is
#: deny-by-default (``status != 'approved'``), so a new state costs no
#: migration — the ``EmailLog.kind`` open-vocabulary precedent rather than the
#: ``harvest_jobs`` CHECK one. ``tests/test_user_status.py`` pins the literals
#: the code actually uses against this set, which is what a CHECK would have
#: bought us, without the downgrade burden of folding live rows.
#:
#:   pending   — public /signup application awaiting review
#:   approved  — the only status with web + API access
#:   rejected  — application declined (terminal; keys revoked)
#:   suspended — access parked, reversible, keys left intact (ACS-353)
USER_STATUSES: frozenset[str] = frozenset({"pending", "approved", "rejected", "suspended"})


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    email: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    name: Mapped[str | None] = mapped_column(Text)
    org: Mapped[str | None] = mapped_column(Text)
    # Free-text "who are you / how will you use this" captured on public /signup
    # (ACS-24) so admins have something to evaluate at approval time. NULL for
    # invite-accept / admin-created / pre-signup-flow rows.
    signup_use_case: Mapped[str | None] = mapped_column(Text)
    # When the applicant ticked the mandatory usage-rules agreement on /signup
    # (ACS-170). NULL = never went through that gate (invited / admin-created /
    # legacy rows). Retained as a light audit trail of consent.
    agreed_terms_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    # Which channel the applicant came from — the ``?src=`` tag on the /signup
    # link we post per community (ACS-210), e.g. 'constellation', 'cyborgism'.
    # NULL = organic / untagged link / pre-attribution rows.
    signup_source: Mapped[str | None] = mapped_column(Text)
    # Split application questions (ACS-302). signup_use_case doubles as the
    # "Planned usage" answer; these three carry the rest. All nullable —
    # pre-redesign and invite-onboarded rows stay NULL.
    signup_outcome: Mapped[str | None] = mapped_column(Text)
    signup_prior_work: Mapped[str | None] = mapped_column(Text)
    signup_referral: Mapped[str | None] = mapped_column(Text)
    # Applicant's own profile page (org page / Scholar / LinkedIn / X / LW) —
    # pins the account to an online identity (ACS-303, feeds ACS-155).
    signup_profile_link: Mapped[str | None] = mapped_column(Text)
    # Discord account linking via OAuth2 guilds.join (ACS-269). Unique on the
    # Discord id: one Discord account can't claim two platform accounts.
    discord_user_id: Mapped[str | None] = mapped_column(Text, unique=True)
    discord_username: Mapped[str | None] = mapped_column(Text)
    discord_connected_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    # Did the bot's guilds.join actually put them in the server? NULL = never
    # attempted (not linked). False = linked but the join failed, so they still
    # need the invite (ACS-307). Goes stale if someone later leaves the server —
    # only a gateway bot or periodic reconciliation would catch that (ACS-301).
    discord_guild_joined: Mapped[bool | None] = mapped_column(Boolean)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    notes: Mapped[str | None] = mapped_column(Text)
    # NULL = password not set yet (legacy rows from before web auth landed; admin can run
    # `acs-keys set-password <email>` to assign one).
    password_hash: Mapped[bytes | None] = mapped_column(LargeBinary)
    # 'user' | 'admin' — checked by /admin* routes. Defaulted to 'user'; admins
    # promoted manually.
    role: Mapped[str] = mapped_column(Text, nullable=False, server_default="user", default="user")
    # Per-user aggregate monthly budget across ALL the user's keys (item 9).
    # NULL = unlimited / no aggregate cap (existing users keep their per-key
    # behaviour untouched). When set, the sum of every key's tokens-this-month
    # must stay under this number; checked as an extra dimension at clamp time.
    monthly_token_budget_total: Mapped[int | None] = mapped_column(BigInteger)
    # Signup lifecycle. 'approved' is the server_default so existing rows
    # backfill cleanly; new ``POST /signup`` rows insert 'pending' explicitly.
    # Login is blocked unless status == 'approved'.
    status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default="approved", default="approved"
    )
    approved_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    approved_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL", use_alter=True),
    )
    rejected_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    # Suspend audit stamps (ACS-353). Set when an admin suspends the account,
    # cleared on unsuspend (and on approve, which is also a valid way back).
    # Suspension deliberately does NOT touch ``api_keys``: the auth-layer status
    # check (``auth.py``) already refuses every request on status alone, so a
    # key sweep would buy nothing and would make the action irreversible — the
    # opposite of the point. See ``admin_suspend_user``.
    suspended_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    suspended_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL", use_alter=True),
    )
    # One-shot post-approval key delivery. ``pending_key_plaintext`` holds the
    # plaintext encrypted with the session_secret-derived pending-key cipher;
    # the first ``/dashboard`` render after approval decrypts, shows, and clears both
    # columns in the same transaction. See ``mailer.send_approval_email`` for
    # the email-side wording (we never put the plaintext in the email body).
    pending_key_plaintext: Mapped[str | None] = mapped_column(Text)
    pending_key_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("api_keys.id", ondelete="SET NULL", use_alter=True),
    )

    keys: Mapped[list[ApiKey]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        foreign_keys="ApiKey.user_id",
    )


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    key_hash: Mapped[bytes] = mapped_column(LargeBinary, nullable=False, unique=True)
    key_prefix: Mapped[str] = mapped_column(Text, nullable=False)
    name: Mapped[str | None] = mapped_column(Text)
    monthly_token_budget: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    last_used_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    # Permanent soft-delete (the "trash" / revoke control). Once set, the key is
    # gone for good — usage history persists for accounting.
    revoked_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    # Reversible "pause": owner can disable a key temporarily and resume it later.
    # A key is usable iff ``revoked_at IS NULL AND disabled_at IS NULL`` — both
    # the bearer-auth path and the cookie-session helpers reject a paused key.
    disabled_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    scopes: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, default=lambda: ["completions"]
    )
    # Per-key daily ceiling (item 9). NULL = unlimited / no daily cap. Default
    # is intentionally NULL (not monthly/10) so existing keys behave identically
    # to before the column was added. Admins opt in via `acs-keys create
    # --daily-budget=...`.
    daily_token_budget: Mapped[int | None] = mapped_column(BigInteger)
    # Per-direction monthly budgets (item 9, Anthropic-style split). NULL =
    # no per-direction cap; the existing ``monthly_token_budget`` total still
    # applies. When both are set the per-direction caps clamp on top of the
    # total.
    monthly_input_token_budget: Mapped[int | None] = mapped_column(BigInteger)
    monthly_output_token_budget: Mapped[int | None] = mapped_column(BigInteger)
    # Per-key monthly cap on ACTIVATION requests (harvesting/steering), which
    # each wake an expensive activation GPU engine (ACS-199). ``0`` = unlimited,
    # following the legacy ``monthly_token_budget`` convention. Counted as a
    # request tally, not tokens, because activation cost tracks
    # requests × layers × prompt-tokens, not generated tokens.
    monthly_activation_budget: Mapped[int] = mapped_column(
        BigInteger, nullable=False, server_default="0"
    )
    # Per-key monthly cap on self-serve bulk HARVEST jobs (POST /v1/harvest,
    # ACS-245). ``0`` = unlimited, same legacy convention as the token /
    # activation budgets. A job tally (not tokens): each job is one Modal GPU
    # spawn, so the unit of cost is the job start.
    monthly_harvest_budget: Mapped[int] = mapped_column(
        BigInteger, nullable=False, server_default="0"
    )

    user: Mapped[User] = relationship(back_populates="keys", foreign_keys=[user_id])


class ApiRequest(Base):
    __tablename__ = "api_requests"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    # Time-range + per-key/ts queries are already served by indexes created in
    # migration 0001 (``ix_api_requests_ts`` and ``ix_api_requests_key_id_ts``).
    ts: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    key_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("api_keys.id", ondelete="CASCADE"), nullable=False
    )
    ip: Mapped[str | None] = mapped_column(INET)
    endpoint: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str | None] = mapped_column(Text)
    n_prompt: Mapped[int | None] = mapped_column(Integer)
    n_completion: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[int] = mapped_column(Integer, nullable=False)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    error_kind: Mapped[str | None] = mapped_column(Text)

    # --- beta event-logging fields (migration 0019) -------------------------
    # All additive, nullable / defaulted — pure request metadata, never body
    # content (no prompt / completion / logprobs text), so the privacy
    # guarantee is untouched. These power the beta usage-pattern dashboards
    # (cold-start pain, interactive-vs-batch, sampling-param adoption).
    #
    # Wrapper→upstream round-trip time (Modal/vLLM time, excludes wrapper
    # overhead). Already emitted to stdout; now persisted for p50/p95 dashboards.
    upstream_latency_ms: Mapped[int | None] = mapped_column(Integer)
    # Time-to-first-token for streaming requests (wall clock from request
    # arrival to first SSE chunk). NULL for unary requests.
    ttft_ms: Mapped[int | None] = mapped_column(Integer)
    # Whether this was an SSE streaming request (interactive) vs unary (batch).
    stream: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    # Whether the request hit a cold (scaled-to-zero) backend. On the unary API
    # path this is a 503 the client retries; on the stream path the request
    # waits through the boot. Cold-start frequency signal.
    cold_boot: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    # Client-declared workload via the optional X-Acs-Workload header
    # ('batch' | 'interactive'). NULL when not sent. Behavioral inference from
    # request cadence is still possible in SQL on top of this ground-truth label.
    workload_type: Mapped[str | None] = mapped_column(Text)
    # Requested max_tokens (before any budget clamp) — distinct from the actual
    # n_completion produced. Distribution signal for sizing decisions.
    req_max_tokens: Mapped[int | None] = mapped_column(Integer)
    # Sampling params the caller actually used (distributions, not the prompt).
    temperature: Mapped[float | None] = mapped_column(Float)
    top_p: Mapped[float | None] = mapped_column(Float)
    # Adoption flags for the research-facing features (value not stored, just
    # whether the caller opted in).
    logprobs_set: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    prompt_logprobs_set: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )
    seed_set: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    echo_set: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")

    # --- activation harvesting / steering telemetry (migration 0031, ACS-199) --
    # Additive, nullable/defaulted request metadata (no tensors, no body content
    # — privacy guarantee intact). Powers per-user activation cost attribution
    # and the activation-adoption dashboards; also the source for the per-key
    # monthly activation-request quota count.
    # Whether this request used the activation engine (capture and/or steering).
    activation: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    # Capture is all-or-nothing (the engine returns every layer), so this is -1
    # (ALL_LAYERS_SENTINEL) when the request captured, else NULL. Kept for
    # dashboard continuity; ``activation`` is the meaningful did-it-capture flag.
    activation_layers: Mapped[int | None] = mapped_column(Integer)
    # Number of steering vectors applied (len(apply_steering_vectors)); NULL when
    # the request didn't steer.
    activation_steering_vectors: Mapped[int | None] = mapped_column(Integer)


class UsageMonthly(Base):
    __tablename__ = "usage_monthly"

    key_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("api_keys.id", ondelete="CASCADE"),
        primary_key=True,
    )
    period_start: Mapped[dt.date] = mapped_column(Date, primary_key=True)
    tokens_prompt: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    tokens_completion: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    request_count: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)


class UsageDaily(Base):
    """Per-day usage rollup, mirroring ``UsageMonthly`` (item 9).

    Written in the same UPSERT pattern, in the same transaction, on every
    successful completion. Used to enforce per-key daily ceilings without
    walking the much larger ``api_requests`` table on the hot path.
    """

    __tablename__ = "usage_daily"

    key_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("api_keys.id", ondelete="CASCADE"),
        primary_key=True,
    )
    period_start: Mapped[dt.date] = mapped_column(Date, primary_key=True)
    tokens_prompt: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    tokens_completion: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    request_count: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)


class UserSession(Base):
    """Server-side web session, created on successful `/login`.

    Signed cookie holds the session id; lookup by id checks revoked_at + age.
    Distinct from `api_keys` (programmatic API access) and from `chat_sessions`
    (UI multi-turn prompt state, added later).
    """

    __tablename__ = "user_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    last_seen_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    revoked_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    user_agent: Mapped[str | None] = mapped_column(Text)
    ip: Mapped[str | None] = mapped_column(INET)


class PasswordReset(Base):
    """A single-use, short-lived password-reset token (forgot-password flow).

    Security: we never store the presented token in plaintext — only its
    SHA-256 hash (``token_hash``), mirroring ``api_keys.key_hash``. The token
    itself is high-entropy (``secrets.token_urlsafe(32)``) so SHA-256 (not
    bcrypt) is the right choice; the threat model is DB-theft, not brute force.

    Lookup is by hashing the presented token and matching ``token_hash``.
    ``used_at`` enforces single-use (consumed via a guarded UPDATE so two
    concurrent submits can't both win); ``expires_at`` enforces a short window
    (see ``settings.password_reset_expiry_minutes``). Rows are kept after use
    as an audit trail; CASCADE on the user FK drops them with the account.
    """

    __tablename__ = "password_resets"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    # SHA-256 of the urlsafe token, hex-encoded. Unique + indexed for O(1)
    # lookup on the presented token's hash.
    token_hash: Mapped[str] = mapped_column(Text, nullable=False, unique=True, index=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    expires_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    used_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))


class ChatSession(Base):
    """A saved workbench prompt the user can return to and continue.

    This is the first place in the wrapper that persists prompt/completion
    text — every other table stores only token counts. See
    ``tutorial.html`` for the user-visible privacy note.
    """

    __tablename__ = "chat_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(Text, nullable=False, server_default="Untitled")
    prompt_text: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    last_max_tokens: Mapped[int] = mapped_column(Integer, nullable=False, server_default="200")
    # 1.0 = faithful base-model sampling (was 0.7, a chat-tuned default) — ACS-145.
    # Changed in migration 0022; existing rows keep their stored value.
    last_temperature: Mapped[float] = mapped_column(Float, nullable=False, server_default="1.0")
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    archived_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    # Set when the user pins the chat in the sidebar (ACS-257). Pinned chats
    # sort above the recency-ordered rest; NULL = not pinned.
    pinned_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    # Short model id selected from the workbench dropdown (e.g. "llama-1b",
    # "llama-405b", "kimi-k2-base"). NULL on legacy rows; the UI falls back
    # to settings.default_model_id when missing.
    model: Mapped[str | None] = mapped_column(Text)
    # API key chosen for this chat in the workbench. NULL = no explicit choice;
    # the route falls back to the user's newest active key (primary_authed_caller).
    # ON DELETE SET NULL so revoking/deleting a key doesn't orphan the chat —
    # it simply reverts to the default key.
    api_key_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("api_keys.id", ondelete="SET NULL")
    )

    snapshots: Mapped[list[ChatSnapshot]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    generations: Mapped[list[ChatGeneration]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class Loom(Base):
    """A standalone tree-exploration document — ACS-148 (first-class object).

    A loom used to hang 1:1 off a ``ChatSession``; it is now its own saved
    object (like a workbench chat): created, named/renamed, listed, opened and
    deleted on its own, at its own ``/loom/<id>`` URL. Its nodes live in
    ``loom_nodes`` and cascade-delete with it.

    ``model`` / ``api_key_id`` remember the last model + key used in this loom's
    controls (parity with ``ChatSession``); both NULL until the user generates.
    Backfilled looms reuse their originating session's id as their own ``id`` so
    the old ``/workbench/{session_id}/loom`` URL redirects deterministically to
    ``/loom/{session_id}`` (idempotent — no duplicate looms on repeat hits).
    """

    __tablename__ = "looms"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(Text, nullable=False, server_default="Untitled loom")
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    archived_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    # Short model id last selected in this loom's controls (parity with
    # ChatSession.model). NULL until the user generates; UI falls back to the
    # global default_model_id.
    model: Mapped[str | None] = mapped_column(Text)
    # API key last used for this loom. ON DELETE SET NULL so revoking a key
    # doesn't orphan the loom — it reverts to the default key.
    api_key_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("api_keys.id", ondelete="SET NULL")
    )

    nodes: Mapped[list[LoomNode]] = relationship(
        back_populates="loom", cascade="all, delete-orphan"
    )


class ChatSnapshot(Base):
    """One row per ``Continue`` generation, capturing the state for revert.

    ``prompt_before`` is what we sent upstream; ``completion_text`` is what
    came back (possibly partial when ``cancelled=True``). Reverting a session
    to a snapshot resets ``chat_sessions.prompt_text`` to ``prompt_before``.
    """

    __tablename__ = "chat_snapshots"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    ts: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    prompt_before: Mapped[str] = mapped_column(Text, nullable=False)
    completion_text: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    n_completion: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    max_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    temperature: Mapped[float] = mapped_column(Float, nullable=False)
    cancelled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    # Short model id of the model used for this Continue. NULL on legacy rows
    # (pre-0009 migration); the UI hides the model label when missing.
    model: Mapped[str | None] = mapped_column(Text)
    # Per-token logprobs for ``completion_text`` (heatmap + top-k), same
    # normalised ``[{token, logprob, top:[{token, logprob}]}]`` shape the loom
    # stores (``LoomNode.logprobs``) and the shared _logprobs_heatmap.html
    # primitive consumes. NULL when logprobs weren't requested for this run
    # (legacy rows, or the logprobs toggle was off) — the UI then shows plain
    # text. Populated server-side from the streamed chunks (ACS-189).
    logprobs: Mapped[list | None] = mapped_column(JSONB)

    session: Mapped[ChatSession] = relationship(back_populates="snapshots")


class CompareSnapshot(Base):
    """One row per Compare-mode *Run all*, capturing every lane as a unit.

    Unlike ``ChatSnapshot`` (one row per single-pane Continue, single model),
    a compare run fires N lanes against one shared prompt. We deliberately
    store the whole run as ONE snapshot with the per-lane detail in a JSONB
    ``lanes`` column (ACS-180) rather than N per-lane rows: it sidesteps the
    "which lane is the model" problem, so adding/removing lanes between runs
    stays unrestricted — we just snapshot whatever lanes existed at Run-all
    time.

    ``lanes`` is a list of dicts, one per lane, shaped like::

        {"model": "trinity-truebase", "max_tokens": 200, "temperature": 0.7,
         "top_p": 1.0, "top_k": -1, "min_p": 0.0, "presence_penalty": 0.0,
         "frequency_penalty": 0.0, "repetition_penalty": 1.0, "seed": null,
         "stop": null, "completion_text": "...", "cancelled": false}

    No ORM relationship back to ``ChatSession`` — callers query by
    ``session_id`` (scoped to the owning user), the same access pattern
    ``chat_revert`` uses, so there's no backref to keep in sync.
    """

    __tablename__ = "compare_snapshots"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    ts: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    # The shared prompt sent to every lane at Run-all time.
    prompt: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    # Per-lane config + final completion, see class docstring.
    lanes: Mapped[list] = mapped_column(JSONB, nullable=False)
    # Denormalised len(lanes) so the history list can show "N lanes" without
    # parsing the JSONB.
    n_lanes: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    # Server-side batch identity of the Compare "Run all" this snapshot captures
    # (ACS-186). Set on both the server-side barrier write and the fallback
    # client POST; a UNIQUE constraint (multiple NULLs allowed in Postgres) makes
    # this the single dedup point — the last lane of a batch to finish, and any
    # racing client POST, both target the same run id and exactly one insert
    # wins. NULL on legacy #158 rows written before this column existed.
    compare_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), unique=True
    )


class ChatGeneration(Base):
    """Durable record of one workbench Continue, decoupled from the HTTP conn.

    Created by ``POST /workbench/{id}/generations``; updated by the background
    streaming task (periodic ``completion_text`` flush, then a final write on
    terminal status). Reopening a tab mid-generation reads the buffered text
    from RAM (via ``app.state.generations``) or, if that's been evicted, from
    this row's ``completion_text``.

    Status values: ``running``, ``completed``, ``failed``, ``cancelled``.
    """

    __tablename__ = "chat_generations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    prompt_before: Mapped[str] = mapped_column(Text, nullable=False)
    completion_text: Mapped[str] = mapped_column(
        Text, nullable=False, server_default="", default=""
    )
    model: Mapped[str] = mapped_column(Text, nullable=False)
    max_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    temperature: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)
    n_prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    n_completion_tokens: Mapped[int | None] = mapped_column(Integer)
    started_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    ended_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    # Compare "Run all" batch identity (ACS-186). NULL for single-pane Continue
    # and loom generations; set to a shared UUID for the N lanes of one Compare
    # Run-all so the last lane to reach a terminal state can detect "whole batch
    # done" (barrier) and assemble the CompareSnapshot server-side — surviving a
    # tab close mid-run. Indexed for the per-batch sibling count in that barrier.
    compare_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), index=True
    )
    # Per-lane sampling config for a compare lane (ACS-186), stored as the
    # clamped vLLM request body plus the lane's index. The base ChatGeneration
    # columns only carry model/max_tokens/temperature; top_p/top_k/min_p/
    # penalties/seed/stop live only here, so the server-side barrier can rebuild
    # the snapshot lane faithfully without trusting the client. NULL for
    # single-pane / loom.
    compare_config: Mapped[dict | None] = mapped_column(JSONB)

    session: Mapped[ChatSession] = relationship(back_populates="generations")


class LoomNode(Base):
    """One node in a *loom* (tree/branching exploration) — ACS-148.

    Each node belongs to a ``Loom`` (the first-class owner-scoped container);
    ownership is enforced via that loom's ``user_id``.
    Each node holds one model continuation; ``parent_id`` is the node it
    branched from (NULL for a root node, whose ``text`` is the seed prompt the
    user typed). Generating ``n`` completions from a node inserts ``n`` child
    rows that share the same ``parent_id``.

    ``logprobs`` is the per-token observability payload for this node's text —
    a JSON list of ``{token, logprob, top: [{token, logprob}, …]}`` entries the
    UI renders as a green→red heatmap with a top-k popover. NULL when logprobs
    weren't requested. ``seed`` is the sampler seed pinned for this branch so it
    can be reproduced; NULL means "unpinned / sampler default".

    Tier-1 scope: owner-only, lightly persisted, no public sharing. Text +
    logprobs are stored verbatim (same privacy posture as ``ChatSnapshot``).
    """

    __tablename__ = "loom_nodes"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    loom_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("looms.id", ondelete="CASCADE"),
        nullable=False,
    )
    # Self-referential parent. NULL = root node (its ``text`` is the seed
    # prompt). ON DELETE CASCADE so deleting a subtree's parent removes the
    # descendants too (Postgres cascades through the self-FK).
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("loom_nodes.id", ondelete="CASCADE"),
    )
    # The text this node contributes: for a root, the user's seed prompt; for a
    # generated node, the model's continuation only (not the cumulative prefix
    # — the UI walks parent links to reconstruct the full context).
    text: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    # Short model id used to generate this node (NULL on a hand-typed root).
    model: Mapped[str | None] = mapped_column(Text)
    # Sampler seed pinned for this branch, so it can be reproduced. NULL = none.
    seed: Mapped[int | None] = mapped_column(BigInteger)
    # Stable sibling ordinal (ACS-340). Siblings from one generation batch share
    # a ``created_at`` to the microsecond, so ordering by time alone is unstable
    # and export→import silently reorders children. ``position`` is assigned at
    # creation (0-based within a parent) and siblings sort by
    # ``(position, created_at)``. Not unique per ``(parent_id, position)``: two
    # concurrent generations under one parent can collide on ``max+1`` — that
    # rare case degrades to ``created_at`` order rather than losing a branch.
    position: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    # Per-token logprob payload for ``text`` (heatmap + top-k). NULL when
    # logprobs were not requested for this generation.
    logprobs: Mapped[list | None] = mapped_column(JSONB)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    loom: Mapped[Loom] = relationship(back_populates="nodes")


class Feedback(Base):
    """In-app feedback submitted by logged-in users (bug / feature / general).

    Low-friction channel used during conferences: a floating button opens a
    modal, the user types a note and optionally attaches screenshots, and the
    row lands here for admins to triage under ``/admin/feedback``.

    ``user_id`` is nullable on purpose: a submission can be marked anonymous
    (``is_anonymous``), in which case we never store who sent it; and even a
    non-anonymous row should survive the submitter's account being deleted
    (``ondelete=SET NULL``) so triage history isn't lost. ``user_agent`` and
    ``page_path`` are captured automatically to help reproduce bug reports.
    """

    __tablename__ = "feedback"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    # 'bug' | 'feature' | 'general' — CHECK in the migration; route validates too.
    category: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    is_anonymous: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false", default=False
    )
    user_agent: Mapped[str | None] = mapped_column(Text)
    page_path: Mapped[str | None] = mapped_column(Text)
    # 'open' | 'resolved' — CHECK in the migration. Newest-first by created_at
    # is the admin list's main sort, hence the index.
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="open", default="open")
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )
    resolved_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    resolved_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    admin_notes: Mapped[str | None] = mapped_column(Text)

    screenshots: Mapped[list[FeedbackScreenshot]] = relationship(
        back_populates="feedback", cascade="all, delete-orphan"
    )


class FeedbackScreenshot(Base):
    """A screenshot attached to a ``Feedback`` row, stored inline as bytea.

    This app has no object storage and runs single-replica, so small images
    live in Postgres directly (capped at ~2 MB each, max 5 per submission —
    enforced in the route). Served back only via the admin-only
    ``/admin/feedback/screenshots/{id}`` route.
    """

    __tablename__ = "feedback_screenshots"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    feedback_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("feedback.id", ondelete="CASCADE"),
        nullable=False,
    )
    content_type: Mapped[str] = mapped_column(Text, nullable=False)
    image_bytes: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    feedback: Mapped[Feedback] = relationship(back_populates="screenshots")


class EmailLog(Base):
    """One row per transactional email we *attempt* to send (approve/reject/…).

    Logged for every attempt — including suppressed ones (``send_status``
    'skipped' when email is disabled or no API key) and failures — so the
    admin Emails page is a complete audit of what the app tried to send, not
    only what succeeded. We store the full rendered ``body_html`` + ``body_text``
    on purpose: full control / "more info than needed" beats wishing we'd kept
    it. ``resend_message_id`` is the join key for delivery webhooks landing in
    ``email_events``.

    Privacy footgun for the future: do NOT log live secret links/keys in the
    body. The current kinds (approval/rejection) carry only the user's name +
    dashboard link — the approval *key* is delivered via /dashboard, never the
    email body. The approval email may also carry a single-use *set-password*
    link (ACS-99), but ``send_approval_email`` re-renders a redacted body
    (placeholder swapped for the URL) before persisting it here, exactly like
    ``send_password_reset_email``. Keep that redaction for any new
    secret-bearing kind before logging it.
    """

    __tablename__ = "email_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    # Recipient user when known; SET NULL so a deleted account doesn't drop the
    # audit row. Nullable also covers any future system mail with no user.
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    # 'approval' | 'rejection' | … — free text on purpose so adding a new email
    # kind needs no migration; the admin filter derives its options from rows.
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    to_email: Mapped[str] = mapped_column(Text, nullable=False)
    from_email: Mapped[str] = mapped_column(Text, nullable=False)
    subject: Mapped[str] = mapped_column(Text, nullable=False)
    body_html: Mapped[str] = mapped_column(Text, nullable=False)
    body_text: Mapped[str] = mapped_column(Text, nullable=False)
    # Our send outcome (distinct from delivery — that comes from webhooks below).
    # 'sent' | 'skipped' | 'failed' — CHECK in the migration.
    send_status: Mapped[str] = mapped_column(Text, nullable=False)
    # Populated when send_status='skipped': 'email_disabled' | 'no_api_key'.
    skip_reason: Mapped[str | None] = mapped_column(Text)
    # HTTP body / exception repr when send_status='failed'.
    error: Mapped[str | None] = mapped_column(Text)
    http_status: Mapped[int | None] = mapped_column(Integer)
    # Resend's message id from the send response; the webhook join key.
    resend_message_id: Mapped[str | None] = mapped_column(Text, index=True)
    # Denormalised latest delivery event for the list view (avoids a join /
    # subquery per row); the full timeline lives in email_events.
    last_event: Mapped[str | None] = mapped_column(Text)
    last_event_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )

    events: Mapped[list[EmailEvent]] = relationship(
        back_populates="email_log", cascade="all, delete-orphan"
    )


class EmailEvent(Base):
    """One row per Resend delivery webhook (delivered/bounced/complained/…).

    The entire verified webhook body is kept verbatim in ``raw_payload`` (JSONB)
    so we never lose detail Resend sends — full control over the audit trail.
    ``email_log_id`` is nullable: a webhook can arrive for a message id we don't
    have a log row for (e.g. sent before this table existed), and we still keep
    it, matched later by ``resend_message_id``.
    """

    __tablename__ = "email_events"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    email_log_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("email_logs.id", ondelete="CASCADE")
    )
    resend_message_id: Mapped[str | None] = mapped_column(Text, index=True)
    # Resend event type, e.g. 'email.delivered', 'email.bounced'.
    event_type: Mapped[str] = mapped_column(Text, nullable=False)
    raw_payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    # ``occurred_at`` is Resend's timestamp from the payload (best-effort parse);
    # ``received_at`` is our clock when the webhook hit us.
    occurred_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    received_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )

    email_log: Mapped[EmailLog | None] = relationship(back_populates="events")


class SignupInvite(Base):
    """Multi-use, expiring invite token granting auto-approved account creation.

    Admins create these via ``/admin/users`` to bypass the normal
    pending→approve flow. A token is either link-only (``email`` is NULL) or
    targeted (``email`` set — used to address the invite email and to display
    in the admin table, but does NOT constrain what email the invitee signs
    up with; the form's email field is always editable).

    The plaintext token is stored directly so admins can copy the magic link
    any time from the ``/admin/users`` table until the link is exhausted,
    expired, or revoked.

    **Usage cap.** ``max_uses`` caps how many accounts a link can create:
    ``None`` = unlimited, a positive integer = that many. Each accepted signup
    inserts one ``SignupInviteRedemption`` row; a link is still claimable while
    ``revoked_at`` is NULL, ``now() <= expires_at``, and (``max_uses`` is NULL
    OR redemption count < ``max_uses``). Enforcement is race-safe at acceptance
    via ``SELECT ... FOR UPDATE`` on the invite row (see ``invite_accept``).

    **Legacy single-use columns.** ``accepted_at`` / ``accepted_by_user_id``
    are retained for backwards compatibility and stamped on the *first*
    redemption only (so old rows/queries still read sensibly). The redemption
    log is the source of truth for the count and the list of created accounts.

    Derived display status (Python, not a column — see ``_invite_status``):
      - 'revoked'   — revoked_at is not None
      - 'used'      — capped link whose redemptions have reached max_uses
      - 'expired'   — expires_at < now (and not revoked/used)
      - 'accepted'  — at least one redemption, still has capacity (or unlimited)
      - 'pending'   — no redemptions yet, still claimable
    """

    __tablename__ = "signup_invites"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    # Plaintext token — stored directly so admins can re-copy the magic link.
    # Unique index for fast lookup on the acceptance path.
    token: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    # NULL → link-only; non-NULL → intended recipient email (informational only,
    # does not restrict which email the invitee uses at signup).
    email: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Usage cap: NULL = unlimited; a positive integer caps accounts created.
    max_uses: Mapped[int | None] = mapped_column(Integer)
    created_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    expires_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    # Stamped on the FIRST redemption only (legacy single-use compatibility).
    accepted_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    accepted_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    revoked_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    # Auto-tag (ACS-371): every account created from this invite gets this tag
    # applied at claim time. NULL = no tagging. This is how a cohort tags itself
    # going forward — paste one link into a bulk-email CSV and everyone who
    # claims it is identifiable without anybody remembering to label them after.
    # Normalized by ``services.user_tags.normalize_tag`` at the create site.
    tag: Mapped[str | None] = mapped_column(Text)

    redemptions: Mapped[list[SignupInviteRedemption]] = relationship(
        back_populates="invite", cascade="all, delete-orphan"
    )


class UserTag(Base):
    """One admin-assigned label on a user account (ACS-371).

    Generic on purpose. Before this, every cross-cutting attribute got its own
    column — ``signup_source`` (write-once from ``?src=`` on public signup,
    never queried, not admin-editable) is the cautionary example — and "which
    accounts belong to the hiring cohort / the HAAISS batch / the internal
    team" had no home at all.

    **Why a join table** rather than ``users.tags``: an ``ARRAY(Text)`` column
    would introduce this repo's first array predicate *and* its first GIN index
    at once (``api_keys.scopes`` is an array but is never queried in SQL, only
    membership-tested in Python), and the ``CompareSnapshot`` docstring is the
    house rule against JSONB for anything you filter on. A plain join table
    answers every query we need — filter the roster, exclude a cohort from a
    Metabase tile — with SQL this codebase already writes. Shape mirrors
    :class:`SignupInviteRedemption`.

    **Open vocabulary**, like ``EmailLog.kind``: ``tag`` is free text, so a new
    label costs no migration and the admin filter derives its options from the
    rows that exist. Values are normalized (lowercase, ``[a-z0-9-]``, <=32
    chars) by ``services.user_tags.normalize_tag`` at every write site.
    """

    __tablename__ = "user_tags"
    __table_args__ = (
        UniqueConstraint("user_id", "tag", name="uq_user_tags_user_id_tag"),
        Index("ix_user_tags_tag", "tag"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    tag: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    # Who applied it. SET NULL so removing an admin doesn't drop the tag itself.
    created_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )


class SignupInviteRedemption(Base):
    """One row per account created from a :class:`SignupInvite` link.

    The source of truth for how many accounts a (possibly multi-use) invite has
    created and which accounts those were. ``user_id`` is ``SET NULL`` on user
    delete so the redemption count survives account deletion.
    """

    __tablename__ = "signup_invite_redemptions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    invite_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("signup_invites.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    redeemed_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    invite: Mapped[SignupInvite] = relationship(back_populates="redemptions")


class ProbeSchedule(Base):
    """Single-row settings table for the capacity-probe cron expression.

    Lives in the wrapper so admins can edit the schedule from /admin without
    redeploying Modal. The APScheduler job on startup reads this row and
    installs a CronTrigger; the /admin/probe/schedule POST replaces the job
    in place.
    """

    __tablename__ = "probe_schedule"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    cron_expression: Mapped[str] = mapped_column(
        Text, nullable=False, server_default="0 7,10,14,16,19,23 * * *"
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )


class ProbeResult(Base):
    """Append-only log of capacity-probe runs.

    Written by the wrapper's APScheduler job (or by /admin/probe/trigger).
    The admin UI shows the most-recent N rows; we never delete here — Postgres
    can hold years of these (a few KB per day) before it becomes a concern.
    """

    __tablename__ = "probe_results"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
        default=uuid.uuid4,
    )
    fired_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )
    ok: Mapped[bool] = mapped_column(Boolean, nullable=False)
    elapsed_s: Mapped[float | None] = mapped_column(Float)
    gpu_type: Mapped[str | None] = mapped_column(Text)
    cloud: Mapped[str | None] = mapped_column(Text)
    region: Mapped[str | None] = mapped_column(Text)
    gpu_count: Mapped[int | None] = mapped_column(Integer)
    gpu_memory_total_mb: Mapped[float | None] = mapped_column(Float)
    gpu_memory_total_std_mb: Mapped[float | None] = mapped_column(Float)
    gpu_memory_used_mb: Mapped[float | None] = mapped_column(Float)
    gpu_memory_used_std_mb: Mapped[float | None] = mapped_column(Float)
    gpu_utilization_pct: Mapped[float | None] = mapped_column(Float)
    gpu_utilization_std_pct: Mapped[float | None] = mapped_column(Float)
    gpu_temperature_c: Mapped[float | None] = mapped_column(Float)
    gpu_temperature_std_c: Mapped[float | None] = mapped_column(Float)
    gpu_power_w: Mapped[float | None] = mapped_column(Float)
    gpu_power_std_w: Mapped[float | None] = mapped_column(Float)
    driver_version: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)


class ModelWarmWindow(Base):
    """Per-model warm-window config driving APScheduler.

    Used to be a hard-coded ``modal.Cron`` inside ``modal_app.py`` (one pair
    per model that needed conference-week scheduling). Now editable from
    /admin without a Modal redeploy: a row per model_id with a warm cron
    (flips ``min_containers`` to 1) and a cool cron (flips back to 0).
    ``enabled`` lets admins pause the window without losing the cron text.
    """

    __tablename__ = "model_warm_window"

    model_id: Mapped[str] = mapped_column(Text, primary_key=True)
    warm_cron: Mapped[str] = mapped_column(Text, nullable=False)
    cool_cron: Mapped[str] = mapped_column(Text, nullable=False)
    timezone: Mapped[str] = mapped_column(Text, nullable=False, server_default="UTC", default="UTC")
    enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="true", default=True
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )


class GpuCostSample(Base):
    """Periodic per-model GPU spend estimate, written by the cost-monitor job.

    ``est_usd = running_containers × hourly_usd_per_container × period_seconds/3600``,
    where the per-container rate = the gpu_type's per-GPU $/hour × the model's
    GPU count. Approximate (samples the running-container count at tick time, so
    it misses sub-interval churn) — enough to chart $/model/day and alert on a
    cost spike. See ``cost_monitor.py``.
    """

    __tablename__ = "gpu_cost_sample"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    ts: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )
    model_id: Mapped[str] = mapped_column(Text, nullable=False)
    gpu_type: Mapped[str] = mapped_column(Text, nullable=False)
    # GPU count per container (n_gpu × n_nodes).
    gpu_count: Mapped[int] = mapped_column(Integer, nullable=False)
    running_containers: Mapped[int] = mapped_column(Integer, nullable=False)
    hourly_usd_per_container: Mapped[float] = mapped_column(Float, nullable=False)
    period_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    est_usd: Mapped[float] = mapped_column(Float, nullable=False)


class BulkEmailBatch(Base):
    """One admin-uploaded bulk-email campaign (ACS-228).

    A batch is created from a CSV upload as a *draft*, reviewed/edited in the
    admin UI, then sent immediately or scheduled. Sending never deletes rows —
    the batch plus its items double as the campaign audit (individual sends
    are additionally logged to ``email_logs`` like all other mail).
    """

    __tablename__ = "bulk_email_batches"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    # Admin who uploaded it; SET NULL so deleting the account keeps the audit.
    created_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    # Label shown in the batch list — defaults to the uploaded filename.
    name: Mapped[str] = mapped_column(Text, nullable=False)
    # 'draft' | 'scheduled' | 'sending' | 'sent' | 'canceled' — CHECK in migration.
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="draft")
    # UTC time a scheduled batch becomes due (status='scheduled' only).
    scheduled_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    sent_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )

    items: Mapped[list[BulkEmailItem]] = relationship(
        back_populates="batch", cascade="all, delete-orphan"
    )


class BulkEmailItem(Base):
    """One recipient row of a bulk-email batch — a fully-addressed email.

    Each row carries its own headers and body (no per-recipient templating:
    the CSV is expected to arrive pre-personalized). Editable in the admin UI
    while the batch is a draft. ``email_log_id`` links to the ``email_logs``
    audit row once a send is attempted.
    """

    __tablename__ = "bulk_email_items"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    batch_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("bulk_email_batches.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # 0-based CSV row order. created_at can't order rows: the whole batch
    # inserts in one transaction and Postgres now() is fixed per transaction.
    position: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    from_email: Mapped[str] = mapped_column(Text, nullable=False)
    to_email: Mapped[str] = mapped_column(Text, nullable=False)
    # cc/bcc hold zero or more comma-separated addresses; reply_to one address.
    cc: Mapped[str | None] = mapped_column(Text)
    bcc: Mapped[str | None] = mapped_column(Text)
    reply_to: Mapped[str | None] = mapped_column(Text)
    subject: Mapped[str] = mapped_column(Text, nullable=False)
    body_html: Mapped[str] = mapped_column(Text, nullable=False)
    body_text: Mapped[str] = mapped_column(Text, nullable=False)
    # 'pending' | 'suppressed' | 'sent' | 'failed' | 'skipped' — CHECK in
    # migration. 'suppressed' = recipient opted out; 'skipped' = mailer soft-
    # skip (email disabled / no API key).
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="pending")
    error: Mapped[str | None] = mapped_column(Text)
    email_log_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("email_logs.id", ondelete="SET NULL")
    )
    sent_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    batch: Mapped[BulkEmailBatch] = relationship(back_populates="items")


class EmailOptOut(Base):
    """One opted-out address — never send bulk email here again (ACS-228).

    Populated by the public one-click /unsubscribe/{token} link that every
    bulk email carries in its footer. Checked (case-insensitively; addresses
    stored lowercased) at CSV upload and again at send time. Applies to bulk
    mail only — transactional email (approvals, password resets) still sends.
    """

    __tablename__ = "email_optouts"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    email: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    # 'link' (clicked unsubscribe) | 'manual' (admin-entered).
    source: Mapped[str] = mapped_column(Text, nullable=False, server_default="link")
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class UpdateSubscriber(Base):
    """One address from the public "subscribe to updates" field (ACS-228).

    Stored lowercased with the opt-in timestamp as the consent record. The
    admin bulk-email page can copy the current list as CSV rows; suppression
    still runs through ``email_optouts`` at send time.
    """

    __tablename__ = "update_subscribers"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    email: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    # Where the opt-in came from; 'landing' for the index-page field.
    source: Mapped[str] = mapped_column(Text, nullable=False, server_default="landing")
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class HarvestJob(Base):
    """One self-serve bulk activation-harvest job (ACS-245, migration 0035).

    Lifecycle: ``POST /v1/harvest`` inserts the row as ``pending`` INSIDE the
    quota-gate transaction (under a ``FOR UPDATE`` lock on the caller's
    api_keys row, so concurrent submits serialize and the concurrency cap is
    race-free), commits, and only then spawns the Modal function — flipping the
    row to ``running`` (or ``failed`` if the spawn errors). ``GET
    /v1/harvest/<id>`` lazily reconciles ``running`` rows against the Modal
    FunctionCall and persists the terminal outcome (``done``/``failed`` +
    ``completed_at``). Rows double as the per-key monthly harvest-quota tally
    (``api_keys.monthly_harvest_budget``; spawn-failures that never reached
    Modal are excluded) and the per-key running-job concurrency count
    (``pending`` counts; rows older than the max harvest runtime are lazily
    aged out to ``failed``).

    ``params`` / ``result`` are request/response metadata only (prompt COUNT,
    layer indices, shard/batch sizes; manifest + shard URLs, timings) — never
    prompt text or tensors, so the api_requests privacy stance carries over.
    ``error`` holds a sanitized client-safe string (exception class name at
    most) — raw exception text stays in server logs.
    """

    __tablename__ = "harvest_jobs"

    # UUID4 as text — the job id is an opaque API token (returned to and
    # presented by clients), not a join target needing the native UUID type.
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    key_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("api_keys.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    model_id: Mapped[str] = mapped_column(Text, nullable=False)
    # ``hv-<12 hex>`` — the harvest run id, which names the output directory on
    # the acs-activation-harvest Volume / bucket prefix.
    run_id: Mapped[str] = mapped_column(Text, nullable=False)
    # Modal FunctionCall id for the spawned harvest. NULL while ``pending`` and
    # on rows whose spawn failed (those are excluded from the monthly quota
    # tally — no GPU was ever engaged).
    modal_call_id: Mapped[str | None] = mapped_column(Text)
    # 'pending' (inserted, spawn not yet confirmed) | 'running' | 'done'
    # | 'failed' | 'cancelled' (owner cancelled via DELETE — ACS-344). The
    # CHECK constraint ``ck_harvest_jobs_status`` (migrations 0035 + 0041) pins
    # this set; a terminal 'cancelled' row frees the key's concurrency slot.
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="pending")
    params: Mapped[dict] = mapped_column(JSONB, nullable=False)
    result: Mapped[dict | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    # When the job reached a terminal status. For ``done`` this anchors the
    # presigned-URL freshness window surfaced as ``urls_expire_at`` (the bucket
    # URLs carry a 7-day TTL; the wrapper has no bucket credentials, so expired
    # URLs are re-minted by an operator re-running the upload pass, not here).
    completed_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
