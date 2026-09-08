"""Local HF tokenizer for pre-flight token counting + budget clamp.

Loaded once at startup. The tokenizer is small (~few MB); only the model
weights are 810 GB. We download it on first run via HF Hub auth using HF_TOKEN
from settings — gated for Llama.
"""

from __future__ import annotations

import os
from functools import lru_cache

# Default transformers to error-only verbosity *before* importing it, so its
# advisory "PyTorch was not found..." notice (written to stderr on import) is
# suppressed. entrypoint.sh sets these in production too; doing it here as well
# covers local/dev launches (scripts/dev-local.sh) that don't go through the
# entrypoint. setdefault keeps any explicit override the operator set.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from transformers import AutoTokenizer  # noqa: E402  (after env setup above)


class TokenCounter:
    def __init__(self, model_name: str, hf_token: str | None) -> None:
        self._tok = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    def count(self, text: str) -> int:
        """Tokens the ENGINE will see for this text, BOS included.

        vLLM tokenizes text prompts with ``add_special_tokens=True``, so
        counting without them undercounted every text prompt by one (Llama
        adds BOS only). That made the context-window check and the activation
        capture cap both off by one: a prompt sitting exactly on the limit
        passed our pre-flight and then failed upstream, where the error is far
        less legible (ACS-317, reported by the usersonas team).

        Pre-tokenized ``list[int]`` prompts are counted by length elsewhere —
        the engine passes those through without adding anything.
        """
        return len(self._tok.encode(text, add_special_tokens=True))

    def count_completion(self, text: str) -> int:
        """Tokens in GENERATED text — no specials.

        The engine prepends BOS to prompts, never to its own output, so
        counting a completion inclusively would bill one phantom token
        (workbench fallback path, review finding on ACS-317).
        """
        return len(self._tok.encode(text, add_special_tokens=False))


@lru_cache(maxsize=None)
def get_token_counter(model_name: str, hf_token: str | None) -> TokenCounter:
    """One TokenCounter per (model_name, hf_token) pair, cached for life of process.

    ``maxsize=None`` because each registry entry gets its own tokenizer (and
    the number of base models we serve is small — N ~ 3–5, not unbounded).
    """
    return TokenCounter(model_name, hf_token)
