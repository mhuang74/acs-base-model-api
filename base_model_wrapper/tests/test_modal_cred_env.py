"""Modal credential env bridge (ACS-126).

``mirror_modal_credentials_to_env`` keeps the two readers of the Modal creds in
sync: the admin-page gating reads the ``Settings`` object (which can be sourced
from ``.env``), while ``modal_ops._auth`` + the Modal SDK read ``os.environ``.
Without the bridge, ``.env``-only creds make the page claim Modal is available
while every RPC fails auth → all models render ``unknown``.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from wrapper.lifespan import mirror_modal_credentials_to_env

_KEYS = ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")


@pytest.fixture(autouse=True)
def _restore_modal_env():
    """Snapshot/restore the Modal env keys so the helper's writes (made via
    ``os.environ`` directly, not monkeypatch) don't leak into the wider suite."""
    saved = {k: os.environ.get(k) for k in _KEYS}
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _settings(token_id, token_secret):
    # The helper only reads these two attributes; a namespace avoids building a
    # full Settings (which needs database_url etc.).
    return SimpleNamespace(modal_token_id=token_id, modal_token_secret=token_secret)


def test_fills_env_from_settings_when_env_unset():
    for k in _KEYS:
        os.environ.pop(k, None)

    mirror_modal_credentials_to_env(_settings("id-from-dotenv", "secret-from-dotenv"))

    assert os.environ["MODAL_TOKEN_ID"] == "id-from-dotenv"
    assert os.environ["MODAL_TOKEN_SECRET"] == "secret-from-dotenv"


def test_real_env_vars_win_over_settings():
    # Prod path: real process env vars are authoritative; setdefault must not
    # clobber them even if Settings resolved different values.
    os.environ["MODAL_TOKEN_ID"] = "real-env-id"
    os.environ["MODAL_TOKEN_SECRET"] = "real-env-secret"

    mirror_modal_credentials_to_env(_settings("other-id", "other-secret"))

    assert os.environ["MODAL_TOKEN_ID"] == "real-env-id"
    assert os.environ["MODAL_TOKEN_SECRET"] == "real-env-secret"


def test_noop_when_settings_have_no_tokens():
    for k in _KEYS:
        os.environ.pop(k, None)

    mirror_modal_credentials_to_env(_settings(None, None))

    assert "MODAL_TOKEN_ID" not in os.environ
    assert "MODAL_TOKEN_SECRET" not in os.environ


def test_requires_both_tokens():
    # A half-configured Settings should not partially populate the env — _auth
    # needs both, so mirroring one would be misleading.
    for k in _KEYS:
        os.environ.pop(k, None)

    mirror_modal_credentials_to_env(_settings("only-id", None))

    assert "MODAL_TOKEN_ID" not in os.environ
    assert "MODAL_TOKEN_SECRET" not in os.environ
