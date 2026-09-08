"""Key generation, hashing, parsing.

A key looks like:  acs-bm-<8-char-prefix>-<token_urlsafe(32)>
                   └─product tag─┘ └─identifies in logs─┘ └─256-bit secret─┘

Storage: only SHA-256(full_key) is persisted, plus the unhashed prefix (used in
log lines so we can attribute a request to a key without ever holding the
secret).

SHA-256 (not argon2/bcrypt) is correct here: the secret is already 256 bits of
random entropy, so the threat model is DB-theft → rainbow-tabling, which is
infeasible against this much entropy. argon2 would burn CPU on every request
for no security gain.
"""

from __future__ import annotations

import hashlib
import re
import secrets
from dataclasses import dataclass

PREFIX_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"
PREFIX_LEN = 8
KEY_TAG = "acs-bm"
KEY_RE = re.compile(r"^acs-bm-([a-z0-9]{8})-([A-Za-z0-9_\-]{40,})$")


@dataclass(frozen=True)
class GeneratedKey:
    plaintext: str
    prefix: str
    hash_: bytes


def generate() -> GeneratedKey:
    prefix = "".join(secrets.choice(PREFIX_ALPHABET) for _ in range(PREFIX_LEN))
    secret = secrets.token_urlsafe(32)
    plaintext = f"{KEY_TAG}-{prefix}-{secret}"
    return GeneratedKey(plaintext=plaintext, prefix=prefix, hash_=hash_key(plaintext))


def hash_key(plaintext: str) -> bytes:
    return hashlib.sha256(plaintext.encode("utf-8")).digest()


def parse_prefix(plaintext: str) -> str | None:
    """Return the 8-char prefix from a well-formed key, else None.

    Used purely for log decoration on the unauth path (e.g. logging a 401 with
    the prefix the caller presented). The full key is still hashed for the DB
    lookup — never trust the prefix alone.
    """
    m = KEY_RE.match(plaintext or "")
    return m.group(1) if m else None
