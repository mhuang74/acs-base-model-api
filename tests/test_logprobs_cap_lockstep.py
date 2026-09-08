"""Pin the logprobs-cap invariant across the two packages (ACS-84 / ACS-191).

The wrapper validates a POSITIVE ``logprobs``/``prompt_logprobs`` count against
``MAX_LOGPROBS`` (``base_model_wrapper/.../schemas.py``) and advertises it via
``/v1/models``. The vLLM serve command caps the *upstream* with ``--max-logprobs``,
fed by ``MAX_LOGPROBS_CAP`` (``modal_app.py``). They live in separate packages —
the wrapper can't import the root serving module at runtime — so the relationship
is enforced here.

The rule changed with full-vocab logprobs (ACS-191). The wrapper is now the
guardrail and vLLM is deployed **uncapped** (``MAX_LOGPROBS_CAP`` default ``-1``)
so it accepts full-vocab ``prompt_logprobs=-1`` and anything else the wrapper
forwards. The invariant is therefore "server cap is uncapped (-1), OR server cap
>= wrapper cap" — never the other way round, which would make vLLM 400 a request
the wrapper already accepted.

This test reads the literals from each source file (no imports, so it triggers
no Modal/vLLM side effects). The wrapper-side value also has its own pin in
``base_model_wrapper/tests/test_completions_validation.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_MODAL_APP = _ROOT / "modal_app.py"
_SCHEMAS = _ROOT / "base_model_wrapper" / "src" / "wrapper" / "schemas.py"


def _read_plain_int(path: Path, name: str) -> int:
    """Read ``NAME = <int>`` (a bare integer literal) from ``path``."""
    m = re.search(
        rf"^{re.escape(name)}\s*=\s*(-?\d+)\b", path.read_text(), re.MULTILINE
    )
    assert m is not None, f"could not find `{name} = <int>` in {path}"
    return int(m.group(1))


def _read_env_default_int(path: Path, name: str) -> int:
    """Read the default from ``NAME = int(os.environ.get("NAME", "<int>"))``.

    ``MAX_LOGPROBS_CAP`` is env-overridable; the *default* string is what the
    deployed server uses because Modal does not forward the shell env into the
    container unless it's baked into the image. So the default is the value
    that matters for the lockstep invariant.
    """
    text = path.read_text()
    m = re.search(
        rf'^{re.escape(name)}\s*=\s*int\(\s*os\.environ\.get\(\s*"{re.escape(name)}"\s*,\s*"(-?\d+)"\s*\)\s*\)',
        text,
        re.MULTILINE,
    )
    assert m is not None, (
        f"could not find `{name} = int(os.environ.get(...))` default in {path}"
    )
    return int(m.group(1))


def test_logprobs_cap_invariant_across_packages():
    modal_cap = _read_env_default_int(_MODAL_APP, "MAX_LOGPROBS_CAP")
    wrapper_cap = _read_plain_int(_SCHEMAS, "MAX_LOGPROBS")
    # -1 = uncapped server (the current, deliberate ACS-191 default); any other
    # value must be a real positive ceiling at least as large as the wrapper's,
    # or callers get a 400 from vLLM the wrapper never anticipated.
    assert modal_cap == -1 or modal_cap >= wrapper_cap, (
        f"logprobs cap invariant violated: modal_app.MAX_LOGPROBS_CAP default="
        f"{modal_cap} must be -1 (uncapped) or >= schemas.MAX_LOGPROBS="
        f"{wrapper_cap} (ACS-84 / ACS-191)."
    )
    # The wrapper's positive top-k cap is still pinned so a change is deliberate
    # and updates /v1/models + the completions-validation pin in lockstep.
    assert wrapper_cap == 100
