from wrapper.keys import KEY_RE, generate, hash_key, parse_prefix


def test_generate_returns_valid_key_shape():
    gk = generate()
    assert gk.plaintext.startswith("acs-bm-")
    assert KEY_RE.match(gk.plaintext) is not None
    assert len(gk.prefix) == 8
    assert len(gk.hash_) == 32  # SHA-256
    assert hash_key(gk.plaintext) == gk.hash_


def test_each_generate_is_unique():
    keys = {generate().plaintext for _ in range(50)}
    assert len(keys) == 50


def test_parse_prefix_valid():
    gk = generate()
    assert parse_prefix(gk.plaintext) == gk.prefix


def test_parse_prefix_invalid():
    assert parse_prefix("") is None
    assert parse_prefix("not-a-key") is None
    assert parse_prefix("sk-openai-leaked") is None


def test_hash_is_stable_for_same_input():
    h1 = hash_key("acs-bm-abcd1234-thisissecret")
    h2 = hash_key("acs-bm-abcd1234-thisissecret")
    assert h1 == h2


def test_pending_key_serializer_encrypts_plaintext_key():
    from wrapper.routes.admin.common import _pending_key_serializer

    plaintext = generate().plaintext
    token = _pending_key_serializer("session-secret").dumps(plaintext)

    assert plaintext not in token
    assert token.startswith("fernet:")
    assert _pending_key_serializer("session-secret").loads(token) == plaintext


def test_pending_key_serializer_accepts_legacy_signed_rows():
    from itsdangerous import URLSafeSerializer

    from wrapper import web_auth as webauth
    from wrapper.routes.admin.common import _pending_key_serializer

    plaintext = generate().plaintext
    legacy = URLSafeSerializer(
        "session-secret", salt=webauth.PENDING_KEY_SALT
    ).dumps(plaintext)

    assert _pending_key_serializer("session-secret").loads(legacy) == plaintext
