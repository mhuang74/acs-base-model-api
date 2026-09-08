import json
import sys
from dataclasses import dataclass
from importlib import util as importlib_util
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _wrapper_defaults_for_model(model_id: str) -> dict[str, Any] | None:
    """Return shared-registry wrapper defaults for known model ids."""
    try:
        from acs_model_registry import wrapper_defaults
    except ModuleNotFoundError:
        registry_path = Path(__file__).resolve().parents[3] / "acs_model_registry" / "__init__.py"
        if not registry_path.is_file():
            return None
        spec = importlib_util.spec_from_file_location("acs_model_registry", registry_path)
        if spec is None or spec.loader is None:
            return None
        registry_module = importlib_util.module_from_spec(spec)
        sys.modules["acs_model_registry"] = registry_module
        spec.loader.exec_module(registry_module)
        wrapper_defaults = registry_module.wrapper_defaults
    return wrapper_defaults(model_id)


# Default idle scaledown windows (seconds), used when a MODELS_REGISTRY_JSON
# entry doesn't carry its own. These mirror ``acs_model_registry``'s
# ``DEFAULT_SCALEDOWN_WINDOW_S`` / ``DEFAULT_ACTIVATION_SCALEDOWN_WINDOW_S``; the
# registry package isn't reliably importable at module scope here (see the
# importlib fallback in ``_wrapper_defaults_for_model``), so we keep local copies
# and lock them to the registry with a test rather than import the constants.
DEFAULT_SCALEDOWN_WINDOW_S = 30 * 60
DEFAULT_ACTIVATION_SCALEDOWN_WINDOW_S = 5 * 60


@dataclass(frozen=True)
class ModelEntry:
    """One row of the wrapper-side model registry.

    Built from ``settings.parsed_models_registry()`` at startup, stashed on
    ``app.state.models``. Each entry maps a short model id (used in request
    bodies + the workbench dropdown) to the upstream Modal URL, tokenizer
    repo, and human-readable shape label.
    """

    model_id: str
    upstream_url: str
    served_model_name: str
    tokenizer_repo: str
    gpu_shape_label: str
    status: str = "live"  # 'live' | 'staging' | 'disabled'
    # Modal app name for the admin Stop / Keep-warm / Release controls.
    # When None, the model has no lifecycle buttons in the admin UI.
    modal_app_name: str | None = None
    # Upstream vLLM ``--max-model-len`` value (prompt + completion tokens).
    # Used for the pre-flight sequence-length 400 in /v1/completions and
    # surfaced in /v1/models as the per-model ``max_model_len`` capability.
    # ``None`` means "unknown; skip the pre-flight check" (caller will see
    # whatever vLLM returns instead of a clean 400 from the wrapper).
    max_model_len: int | None = None
    # Per-model upstream timeout, in seconds. Overrides the global
    # ``Settings.upstream_timeout_s`` when set. Small models can get a tight
    # bound (a 1B model should never sit on the wire for 20 min); 405B/Kimi
    # cold-boots need generous budgets. ``None`` = inherit global.
    upstream_timeout_s: float | None = None
    # Activation-engine upstream (the ``acs-<id>-activation`` vLLM-Lens side-car).
    # Set only for models that expose activation harvesting + steering. When
    # present, requests carrying activation params are routed here instead of
    # ``upstream_url``; when ``None`` the model has no activation support and such
    # requests are rejected (ACS-199). Also drives the ``activations`` capability
    # flag in /v1/models.
    activation_upstream_url: str | None = None
    # Per-model cap on the PROMPT length (tokens) for an activation CAPTURE
    # request. Capture returns every layer's residual stream, payload ≈
    # n_layers × prompt_tokens × d_model × 2 B, and only prompt length is
    # per-request tunable. Since ACS-250 streams the capture response (no
    # wrapper-RAM materialization) the bound is the RESPONSE SIZE the client
    # downloads — per-model: ~4 MB/token on 405B (126 × 16384, stays tight ~64)
    # vs ~0.27 MB/token on 8B (32 × 4096, set to 256). ``None`` → the wrapper's
    # ``DEFAULT_ACTIVATION_MAX_PROMPT_TOKENS``. Only meaningful when
    # ``activation_upstream_url`` is set. (ACS-199 / ACS-250)
    activation_max_prompt_tokens: int | None = None
    # Decoder-layer count, used to scale the capture cap when a request asks
    # for a SUBSET of layers: payload ≈ len(layers) × tokens × d_model × 2 B, so
    # a 1-of-32 capture affords ~32× the prompt length at the same response size
    # (ACS-317). ``None`` → no scaling (the flat cap stands), so leaving it unset
    # is safe; set it per model in MODELS_REGISTRY_JSON to enable subset scaling.
    n_layers: int | None = None
    # Modal app name of this model's BULK harvester (ACS-245). ``None`` → derived
    # from the MODEL ID as ``acs-{model_id}-harvest`` (matches
    # serving/harvest_offline.py's ``acs-<shared-registry MODEL_ID>-harvest``
    # whenever the wrapper's registry key is the deploy-time MODEL_ID — true
    # for all current models, including Trinity; ACS-273 changed this from a
    # ``{modal_app_name}-harvest`` derivation that broke on Trinity's
    # pre-rename serving app name). Set explicitly only when a wrapper alias
    # diverges from the deploy-time MODEL_ID.
    harvest_app_name: str | None = None
    # Modal app name of this model's ACTIVATION engine (cost sampler, ACS-221).
    # ``None`` → derived as ``acs-{model_id}-activation`` (matches
    # modal_app_activation.py's ``acs-<MODEL_ID>-activation`` whenever the
    # wrapper's registry key is the shared-registry MODEL_ID the app was
    # deployed under — true for all current models, including Trinity, whose
    # SERVING app kept the pre-rename name but whose activation app did not).
    # Set explicitly only when a wrapper alias diverges from the deploy-time
    # MODEL_ID, or the app was deployed with an ``ACT_APP_NAME`` override.
    # Only meaningful when ``activation_upstream_url`` is set.
    activation_app_name: str | None = None
    # Idle window (seconds) after which Modal scales this model's SERVING engine
    # to zero — mirrors the shared registry's ``scaledown_window_s``. Drives the
    # RPC-free cold-hint (``_model_is_cold``) for the serving breaker key, so a
    # request arriving after the real teardown takes the keepalive path instead
    # of a global 30-min guess (ACS-226). Defaults to the registry's 30-min
    # default when a JSON entry doesn't carry it.
    scaledown_window_s: int = DEFAULT_SCALEDOWN_WINDOW_S
    # Idle window (seconds) for this model's ACTIVATION engine — the separate
    # ``acs-<id>-activation`` side-car scales down on its own (usually shorter)
    # schedule than serving. Drives the cold-hint for the ``<model>::activation``
    # breaker key (ACS-226). Registry default is 5 min; deployed activation
    # models use 10 min.
    activation_scaledown_window_s: int = DEFAULT_ACTIVATION_SCALEDOWN_WINDOW_S


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    database_url: str = Field(
        ...,
        description="Postgres DSN. Railway provides this as DATABASE_URL on the Postgres add-on.",
    )

    # Upstream Modal/vLLM endpoint (LEGACY single-model fallback)
    modal_base_url: str = Field(
        ...,
        description="Base URL of the deployed Modal vLLM endpoint. Used as the fallback registry entry when MODELS_REGISTRY_JSON is unset.",
    )
    vllm_api_key: str = Field(
        ...,
        description="Shared bearer token vLLM expects (server-side only — never sent to wrapper clients). Same key works across all Modal apps that share the vllm-api Secret.",
    )

    # Admin
    admin_token: str = Field(
        ...,
        description="Root credential for /admin/* endpoints. Long random.",
    )

    # Model & tokenizer (LEGACY — see models_registry_json for multi-model)
    served_model_name: str = "meta-llama/Llama-3.1-405B"
    hf_token: str | None = Field(
        default=None,
        description="HF token for downloading the tokenizer (gated repo). Not the same as vllm_api_key.",
    )

    # Multi-model routing registry. JSON string mapping model_id to entry.
    # When unset / empty, the wrapper synthesises a single entry from
    # ``modal_base_url`` + ``served_model_name`` so existing single-model
    # deploys keep working. Example:
    #   MODELS_REGISTRY_JSON='{"llama-8b":{"upstream_url":"https://..."},
    #   "custom":{"upstream_url":"https://...","served_model_name":"gpt2","tokenizer_repo":"gpt2"}}'
    models_registry_json: str = Field(
        default="",
        description="JSON object: model_id → {upstream_url, served_model_name, tokenizer_repo, gpu_shape_label, status}. Empty = fall back to single-model from modal_base_url.",
    )
    default_model_id: str = Field(
        default="",
        description="Model id used when a request omits 'model'. If empty, the first registry entry is used.",
    )

    # Behaviour
    log_level: str = "INFO"
    upstream_timeout_s: float = (
        1200.0  # > observed cold-boot p_high (797 s, researchlog 2026-05-13)
    )
    last_used_throttle_s: int = 300  # update api_keys.last_used_at at most every N seconds
    chat_completions_handout_url: str = ""
    log_ip: bool = True

    # --- Sentry error tracking (ACS-40) ------------------------------------
    # No-op when sentry_dsn is empty (local/dev/tests). Privacy is enforced in
    # observability.init_sentry regardless of these values: send_default_pii is
    # forced off and request bodies/PII are scrubbed before send (ACS-43) — a
    # prompt must never reach Sentry.
    sentry_dsn: str = ""
    sentry_environment: str = "production"
    sentry_release: str = ""  # e.g. git SHA; empty = let Sentry infer/none
    sentry_traces_sample_rate: float = 1.0  # fine at beta volume; dial down later
    sentry_profiles_sample_rate: float = 0.0  # profiling off by default
    sentry_enable_logs: bool = True  # forward stdlib logs (our prompt logs bypass stdlib)

    # Rate limits (slowapi expression strings). Note: these are currently not
    # wired up — actual limits are hardcoded on the decorators in main.py.
    rate_limit_per_ip: str = "240/minute"
    rate_limit_login_per_ip: str = "5/minute"

    # CORS for the public /v1 API (ACS-146). Default "*" because /v1 is a
    # token-in-Authorization-header API like OpenAI's — wildcard origin is
    # appropriate. allow_credentials stays False app-wide so browsers will
    # not send the cookie-authenticated acs_session cross-origin (no new
    # CSRF surface; cf. ACS-102). Comma-separated list; "*" = any origin.
    cors_allow_origins: str = "*"

    # Web sessions (/login, /logout, /dashboard, /chat)
    session_secret: str | None = Field(
        default=None,
        description="Secret key for signing session cookies (itsdangerous). Long random string. Rotating invalidates every active session AND any pending one-shot keys stored encrypted on users.pending_key_plaintext.",
    )
    cookie_secure: bool = Field(
        default=True,
        description="Set the Secure flag on cookies. Must be False for localhost dev (no HTTPS).",
    )
    session_max_age_days: int = 30

    # Signup + approval workflow
    signup_enabled: bool = Field(
        default=False,
        description="If False, /signup returns 404 and the nav link is suppressed. Default is False during private beta: the public frontpage shows a 'Request access' mailto CTA instead of an 'Apply for access' button.",
    )
    signup_notify_email: str = Field(
        default="infra@acsresearch.org",
        description="Single address the 'a new signup is pending review' notification is sent to when a public /signup application arrives (ACS-24), so the team doesn't have to poll /admin/users. Point it at a shared inbox / distribution list to fan out. Only sent when email_enabled.",
    )
    beta_discord_url: str | None = Field(
        default=None,
        description="Beta Discord invite surfaced to approved users — included in the approval email (ACS-24) and available to the CSV-personalized invite flow. Set per-env via BETA_DISCORD_URL; when unset the Discord link is simply omitted everywhere. (The old checked-in default had expired — ACS-269.)",
    )

    # Discord account linking (ACS-269) — OAuth2 `identify guilds.join` flow on
    # the existing community bot app. All four must be set for the feature to
    # exist; otherwise the Connect button is hidden and /discord/* returns 404,
    # so this is safe to deploy before the Developer-Portal setup.
    discord_client_id: str | None = Field(
        default=None,
        description="Discord application (bot) OAuth2 client id. From the Developer Portal OAuth2 tab.",
    )
    discord_client_secret: str | None = Field(
        default=None,
        description="Discord application OAuth2 client secret. Never logged.",
    )
    discord_bot_token: str | None = Field(
        default=None,
        description="Bot token used for the guilds.join member-add call. Shared with the Claude Discord MCP config — resetting it in the portal breaks that too. Never logged.",
    )
    discord_guild_id: str | None = Field(
        default=None,
        description="Snowflake id of the community server the bot adds connected users to.",
    )

    @property
    def discord_oauth_enabled(self) -> bool:
        return all(
            (
                self.discord_client_id,
                self.discord_client_secret,
                self.discord_bot_token,
                self.discord_guild_id,
            )
        )
    invite_expiry_days: int = Field(
        default=7,
        description="How many days an unused invite link remains valid (both link-only and invite-by-email share this). After this the token is expired and cannot be claimed. Override per-env via INVITE_EXPIRY_DAYS — bump it if invitees need longer to act during a beta.",
    )
    default_monthly_token_budget_total: int = Field(
        default=5_000_000,
        description="Per-user aggregate cap assigned on admin-approve (only when the user's existing aggregate is NULL — never overwrites a manual budget).",
    )
    default_per_key_budget: int = Field(
        default=5_000_000,
        description="Per-key monthly budget for the auto-created key on admin-approve, AND for self-service /me/keys when the user doesn't supply one.",
    )

    # Email (Resend)
    email_enabled: bool = Field(
        default=False,
        description="Master switch. If False, mailer.send returns without making an HTTP call (dev-friendly).",
    )
    email_provider: str = Field(
        default="resend",
        description="Currently only 'resend' is honoured. SMTP fallback noted in mailer.py.",
    )
    resend_api_key: str | None = Field(
        default=None,
        description="Resend API key. If None even with email_enabled=True, the mailer logs a warning and returns.",
    )
    resend_webhook_secret: str | None = Field(
        default=None,
        description="Resend (Svix) webhook signing secret, 'whsec_...'. Used by POST /webhooks/resend to verify delivery events. If None, the endpoint accepts (200) but records nothing — unverified payloads are never stored.",
    )
    email_from: str = Field(
        default="ACS Infra <infra@acsresearch.org>",
        description="From: header on outgoing transactional email. Domain must be SPF/DKIM-verified in Resend.",
    )
    public_base_url: str = Field(
        default="http://localhost:8000",
        description=(
            "Public origin of the deployed app (scheme + host). The single knob "
            "for the domain: builds links in transactional email (dashboard / "
            "invite / password-reset URLs) and the API base shown in the "
            "/tutorial docs. Change this one env var when the domain moves "
            "(e.g. to https://infra.acsresearch.org)."
        ),
    )
    password_reset_expiry_minutes: int = Field(
        default=60,
        description="How long a self-service password-reset token stays valid, in minutes. Short on purpose — the link is single-use and the window only needs to cover reading the email.",
    )
    approval_set_password_expiry_minutes: int = Field(
        default=7 * 24 * 60,
        description="How long the 'set your password' link in an admin-approval email stays valid, in minutes. Longer than a self-service reset (default 7 days) because an approved user may not open the email immediately; it's still single-use and only issued to password-less accounts.",
    )

    # GPU cost monitoring (see cost_monitor module + the hourly scheduler job).
    gpu_hourly_usd_by_type_json: str = Field(
        default='{"H200": 4.54, "L40S": 1.95}',
        description="Per-GPU on-demand $/hour by Modal gpu_type, as JSON. A container's hourly cost is this rate × the model's GPU count (n_gpu × n_nodes), e.g. 8×H200 ≈ $36/hr. Retune if Modal pricing moves; unknown types cost 0 (logged).",
    )
    cost_alert_daily_usd: float = Field(
        default=1200.0,
        description="Rolling-24h estimated GPU spend (USD) above which the cost monitor raises a Sentry alert (grouped, routes to Discord). 0 disables the alert. Tune to ~one always-on large model + headroom.",
    )
    cost_sample_interval_minutes: int = Field(
        default=60,
        description="How often the cost monitor samples running container counts to estimate spend. Each sample is one Modal control-plane lookup per live model.",
    )
    cost_sample_warmup_seconds: int = Field(
        default=120,
        description="Delay before the first cost sample after boot. Gives the Modal client time to be ready so the (uncached) runner-count read returns true counts instead of a cold-start 0. The sampler then repeats every cost_sample_interval_minutes.",
    )

    # --- Self-serve bulk activation harvest (ACS-245) -----------------------
    # Caps on POST /v1/harvest, the endpoint that spawns the offline bulk
    # harvester (serving/harvest_offline.py) on Modal for untrusted input.
    harvest_max_prompts: int = Field(
        default=4096,
        description="Max prompts per harvest job. One job = one Modal GPU spawn; larger corpora should be split into multiple jobs (each is one quota unit).",
    )
    harvest_max_est_tokens: int = Field(
        default=2_000_000,
        description="Max ESTIMATED total tokens per harvest job (chars/4 across all prompts — cheap pre-flight bound, no tokenizer pass over thousands of prompts). Caps GPU time + activation storage per job.",
    )
    harvest_max_running_per_key: int = Field(
        default=1,
        description="Max concurrently-running BIG-MODEL (multi-GPU) harvest jobs per API key. Each job holds a whole multi-GPU container for its run, so >1 lets a single key fan out expensive hardware. Small-model jobs have their own lane (harvest_max_running_per_key_small) and are counted separately.",
    )
    harvest_max_running_per_key_small: int = Field(
        default=3,
        description="Max concurrently-running SMALL-MODEL (single-GPU) harvest jobs per API key (ACS-321). Job sizes differ by ~3 orders of magnitude, so a single flat cap meant an 8-prompt interactive job queued behind a 4096-prompt corpus run — a blocked session rather than a fairness win. Cheap 1×L40S containers, so a few at once is affordable.",
    )
    harvest_max_running_big_model: int = Field(
        default=2,
        description="Global (cross-key) cap on concurrently-running BIG-MODEL (multi-GPU, 8×H200) harvest jobs. The per-key cap can't stop N different keys each spawning an 8×H200 405B/Trinity harvest (80 H200 at once — un-allocatable + ~$360/hr); this bounds the multi-GPU fleet across all keys. Cheap 1×L40S (8B) harvests stay uncapped.",
    )
    harvest_max_longpoll_per_key: int = Field(
        default=4,
        description="Max simultaneously-WAITING GET /v1/harvest/<id>?wait= requests per API key (ACS-321). Default 4 = harvest_max_running_per_key_small (3) + harvest_max_running_per_key (1): a key may legally run three small jobs AND one big one at once, so it must be able to wait on all four, or the last caller is declined by construction — the exact failure this ceiling is meant to avoid. Over the cap the poll is answered early with the current state, an X-Acs-Longpoll: declined header and Retry-After.",
    )
    harvest_max_longpoll_total: int = Field(
        default=8,
        description="Global cap on simultaneously-WAITING harvest polls across all keys (ACS-321). poll_harvest runs on the event loop's DEFAULT thread pool, shared with full-vocab logprob serialization, zstd compression and spawn_harvest — at 500 concurrent waiters that pool's latency measured ~3s, which lands on /v1/completions traffic. This is the safety valve; lower it if completions latency correlates with harvest polling.",
    )
    harvest_max_body_bytes: int = Field(
        default=50 * 1024 * 1024,
        description="Max Content-Length accepted by POST /v1/harvest (default 50 MB). Rejected with 413 BEFORE the body is read, so an oversized prompt corpus can't balloon wrapper RAM; the chars/4 estimated-token cap still applies after parsing.",
    )

    # Modal-side admin controls + scheduled probe (see modal_ops + probe modules).
    modal_token_id: str | None = Field(
        default=None,
        description="Workspace-scoped Modal token id. When unset, admin model controls return 503; the rest of /admin still renders.",
    )
    modal_token_secret: str | None = Field(
        default=None,
        description="Companion to modal_token_id.",
    )
    modal_probe_url: str | None = Field(
        default=None,
        description="HTTP endpoint URL exposed by serving/capacity_scheduled.py. The wrapper-side APScheduler job POSTs here.",
    )
    modal_probe_bearer: str | None = Field(
        default=None,
        description="Bearer token shared with the probe-bearer Modal secret. Must match.",
    )

    def parsed_gpu_hourly_rates(self) -> dict[str, float]:
        """Per-GPU $/hour by gpu_type, parsed from ``gpu_hourly_usd_by_type_json``.

        Returns an empty dict on malformed JSON (logged by the caller / falls
        back to 0 cost per unknown type) — a bad rate config must never crash
        the cost-sampler job.
        """
        raw = (self.gpu_hourly_usd_by_type_json or "").strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if not isinstance(parsed, dict):
            return {}
        out: dict[str, float] = {}
        for k, v in parsed.items():
            try:
                out[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
        return out

    def parsed_models_registry(self) -> dict[str, ModelEntry]:
        """Parse ``models_registry_json`` once.

        Fall back to a single synthesised entry from ``modal_base_url`` +
        ``served_model_name`` when the env var is empty, so single-model
        deployments keep working without setting MODELS_REGISTRY_JSON.

        Raises:
            RuntimeError on malformed JSON or missing required fields — fail
            fast at startup rather than 500 per request.
        """
        raw = (self.models_registry_json or "").strip()
        if not raw:
            # Legacy single-model fallback.
            entry = ModelEntry(
                model_id=self.served_model_name,
                upstream_url=self.modal_base_url,
                served_model_name=self.served_model_name,
                tokenizer_repo=self.served_model_name,
                gpu_shape_label="legacy",
                status="live",
            )
            return {entry.model_id: entry}

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"MODELS_REGISTRY_JSON is not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("MODELS_REGISTRY_JSON must be a JSON object")
        out: dict[str, ModelEntry] = {}
        required = ("upstream_url", "served_model_name", "tokenizer_repo")
        for model_id, raw_entry in parsed.items():
            if not isinstance(raw_entry, dict):
                raise RuntimeError(f"MODELS_REGISTRY_JSON entry {model_id!r} is not a JSON object")
            defaults = _wrapper_defaults_for_model(str(model_id)) or {}
            entry_data = {**defaults, **raw_entry}
            for field in required:
                if field not in entry_data:
                    raise RuntimeError(
                        f"MODELS_REGISTRY_JSON entry {model_id!r} missing required field {field!r}"
                    )
            max_len_raw = entry_data.get("max_model_len")
            max_model_len: int | None
            if max_len_raw is None:
                max_model_len = None
            else:
                try:
                    max_model_len = int(max_len_raw)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"MODELS_REGISTRY_JSON entry {model_id!r} has non-integer "
                        f"max_model_len={max_len_raw!r}"
                    ) from exc
                if max_model_len < 1:
                    raise RuntimeError(
                        f"MODELS_REGISTRY_JSON entry {model_id!r} has max_model_len "
                        f"< 1 ({max_model_len})"
                    )
            timeout_raw = entry_data.get("upstream_timeout_s")
            upstream_timeout_s: float | None
            if timeout_raw is None:
                upstream_timeout_s = None
            else:
                try:
                    upstream_timeout_s = float(timeout_raw)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"MODELS_REGISTRY_JSON entry {model_id!r} has non-numeric "
                        f"upstream_timeout_s={timeout_raw!r}"
                    ) from exc
                if upstream_timeout_s <= 0:
                    raise RuntimeError(
                        f"MODELS_REGISTRY_JSON entry {model_id!r} has "
                        f"upstream_timeout_s <= 0 ({upstream_timeout_s})"
                    )
            activation_upstream_url = entry_data.get("activation_upstream_url")
            if activation_upstream_url is not None and not isinstance(activation_upstream_url, str):
                raise RuntimeError(
                    f"MODELS_REGISTRY_JSON entry {model_id!r} has non-string "
                    f"activation_upstream_url={activation_upstream_url!r}"
                )
            activation_app_name = entry_data.get("activation_app_name")
            if activation_app_name is not None and not isinstance(activation_app_name, str):
                raise RuntimeError(
                    f"MODELS_REGISTRY_JSON entry {model_id!r} has non-string "
                    f"activation_app_name={activation_app_name!r}"
                )
            harvest_app_name = entry_data.get("harvest_app_name")
            if harvest_app_name is not None and not isinstance(harvest_app_name, str):
                raise RuntimeError(
                    f"MODELS_REGISTRY_JSON entry {model_id!r} has non-string "
                    f"harvest_app_name={harvest_app_name!r}"
                )
            n_layers_raw = entry_data.get("n_layers")
            n_layers: int | None
            if n_layers_raw is None:
                n_layers = None
            elif isinstance(n_layers_raw, bool):
                # `true` would coerce to 1 and silently disable scaling.
                raise RuntimeError(
                    f"MODELS_REGISTRY_JSON entry {model_id!r} has boolean "
                    f"n_layers={n_layers_raw!r}; expected an integer"
                )
            else:
                try:
                    n_layers = int(n_layers_raw)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"MODELS_REGISTRY_JSON entry {model_id!r} has non-integer "
                        f"n_layers={n_layers_raw!r}"
                    ) from exc
                if n_layers < 1:
                    raise RuntimeError(
                        f"MODELS_REGISTRY_JSON entry {model_id!r} has n_layers="
                        f"{n_layers} (must be >= 1)"
                    )
            act_max_prompt_raw = entry_data.get("activation_max_prompt_tokens")
            activation_max_prompt_tokens: int | None
            if act_max_prompt_raw is None:
                activation_max_prompt_tokens = None
            else:
                try:
                    activation_max_prompt_tokens = int(act_max_prompt_raw)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"MODELS_REGISTRY_JSON entry {model_id!r} has non-integer "
                        f"activation_max_prompt_tokens={act_max_prompt_raw!r}"
                    ) from exc
                if activation_max_prompt_tokens < 1:
                    raise RuntimeError(
                        f"MODELS_REGISTRY_JSON entry {model_id!r} has "
                        f"activation_max_prompt_tokens < 1 ({activation_max_prompt_tokens})"
                    )

            def _window_field(field_name: str, default: int) -> int:
                """Parse a positive-int scaledown window, defaulting when absent.

                Unlike ``max_model_len``/``upstream_timeout_s`` (nullable), the
                windows always resolve to a positive int so ``_model_is_cold``
                never has to special-case ``None`` on the hot path. An *absent*
                key defaults (a custom model that declares no window); an
                *explicit* ``null`` raises — a window is never legitimately
                "unset", and silently substituting the default would override the
                registry-inherited value in the unsafe direction (judged warm
                past the real teardown).
                """
                if field_name in entry_data and entry_data[field_name] is None:
                    raise RuntimeError(
                        f"MODELS_REGISTRY_JSON entry {model_id!r} has "
                        f"{field_name}=null (window may not be null; omit it to "
                        f"inherit the default)"
                    )
                raw_val = entry_data.get(field_name)
                if raw_val is None:
                    return default
                try:
                    val = int(raw_val)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"MODELS_REGISTRY_JSON entry {model_id!r} has non-integer "
                        f"{field_name}={raw_val!r}"
                    ) from exc
                if val < 1:
                    raise RuntimeError(
                        f"MODELS_REGISTRY_JSON entry {model_id!r} has "
                        f"{field_name} < 1 ({val})"
                    )
                return val

            scaledown_window_s = _window_field(
                "scaledown_window_s", DEFAULT_SCALEDOWN_WINDOW_S
            )
            activation_scaledown_window_s = _window_field(
                "activation_scaledown_window_s", DEFAULT_ACTIVATION_SCALEDOWN_WINDOW_S
            )
            out[model_id] = ModelEntry(
                model_id=model_id,
                upstream_url=entry_data["upstream_url"],
                served_model_name=entry_data["served_model_name"],
                tokenizer_repo=entry_data["tokenizer_repo"],
                gpu_shape_label=entry_data.get("gpu_shape_label", ""),
                status=entry_data.get("status", "live"),
                modal_app_name=entry_data.get("modal_app_name"),
                max_model_len=max_model_len,
                upstream_timeout_s=upstream_timeout_s,
                activation_upstream_url=activation_upstream_url or None,
                activation_max_prompt_tokens=activation_max_prompt_tokens,
                n_layers=n_layers,
                harvest_app_name=harvest_app_name or None,
                activation_app_name=activation_app_name or None,
                scaledown_window_s=scaledown_window_s,
                activation_scaledown_window_s=activation_scaledown_window_s,
            )
        if not out:
            raise RuntimeError("MODELS_REGISTRY_JSON has no entries")
        return out

    def resolve_default_model_id(self, registry: dict[str, ModelEntry]) -> str:
        """The model id used when a request omits ``model``.

        Honours ``default_model_id`` if set + present in the registry;
        otherwise picks the first live entry (insertion order).
        """
        if self.default_model_id and self.default_model_id in registry:
            return self.default_model_id
        live = [m for m in registry.values() if m.status == "live"]
        if not live:
            raise RuntimeError("Models registry has no 'live' entries")
        return live[0].model_id
