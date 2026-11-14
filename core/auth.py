"""Password hashing helpers for local ProcureGuard user accounts.

Uses the Python standard library only. This module does not issue sessions or
replace transport security, rate limiting, MFA, or account-recovery controls.
"""

import base64
import hashlib
import hmac
import secrets

_ALGORITHM = "pbkdf2_sha256"
_ITERATIONS = 310_000
_SALT_BYTES = 16


def hash_password(password: str) -> str:
    """Return a versioned salted PBKDF2-HMAC-SHA256 password hash."""
    if not isinstance(password, str) or not password:
        raise ValueError("Password must be a non-empty string.")
    salt = secrets.token_bytes(_SALT_BYTES)
    derived = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, _ITERATIONS
    )
    salt_text = base64.urlsafe_b64encode(salt).decode("ascii").rstrip("=")
    hash_text = base64.urlsafe_b64encode(derived).decode("ascii").rstrip("=")
    return f"{_ALGORITHM}${_ITERATIONS}$" + salt_text + "$" + hash_text


def verify_password(password: str, encoded_hash: str) -> bool:
    """Verify a password hash without exposing parsing or comparison details."""
    if not isinstance(password, str) or not isinstance(encoded_hash, str):
        return False
    try:
        algorithm, iterations_text, salt_text, expected_text = encoded_hash.split("$")
        if algorithm != _ALGORITHM:
            return False
        iterations = int(iterations_text)
        if not 100_000 <= iterations <= 2_000_000:
            return False
        salt = base64.urlsafe_b64decode(salt_text + "=" * (-len(salt_text) % 4))
        expected = base64.urlsafe_b64decode(
            expected_text + "=" * (-len(expected_text) % 4)
        )
        if len(salt) != _SALT_BYTES or len(expected) != hashlib.sha256().digest_size:
            return False
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, iterations
        )
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError, UnicodeError):
        return False


def create_access_token(user: dict, secret: str, expires_in_seconds: int = 1800) -> str:
    """Create a compact HMAC-SHA256 bearer token with a short expiry."""
    import json
    import time
    if not secret or len(secret) < 32:
        raise ValueError("USER_SESSION_SECRET must contain at least 32 characters.")
    if not 60 <= expires_in_seconds <= 86400:
        raise ValueError("Token lifetime must be between 60 and 86400 seconds.")
    def encode(value):
        raw = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")
    header = encode({"alg": "HS256", "typ": "JWT"})
    payload = encode({
        "sub": str(user["id"]), "email": user["email"], "role": user["role"],
        "iat": int(time.time()), "exp": int(time.time()) + expires_in_seconds,
    })
    signing_input = header + "." + payload
    signature = hmac.new(secret.encode(), signing_input.encode(), hashlib.sha256).digest()
    return signing_input + "." + base64.urlsafe_b64encode(signature).decode().rstrip("=")


def decode_access_token(token: str, secret: str) -> dict | None:
    """Validate signature and expiry; return claims only for a valid token."""
    import json
    import time
    if not secret or len(secret) < 32 or not isinstance(token, str):
        return None
    try:
        header_text, payload_text, signature_text = token.split(".")
        signing_input = header_text + "." + payload_text
        supplied = base64.urlsafe_b64decode(signature_text + "=" * (-len(signature_text) % 4))
        expected = hmac.new(secret.encode(), signing_input.encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(supplied, expected):
            return None
        header = json.loads(base64.urlsafe_b64decode(header_text + "=" * (-len(header_text) % 4)))
        claims = json.loads(base64.urlsafe_b64decode(payload_text + "=" * (-len(payload_text) % 4)))
        if header != {"alg": "HS256", "typ": "JWT"}:
            return None
        now = int(time.time())
        if int(claims["exp"]) <= now or int(claims["iat"]) > now + 60:
            return None
        if not str(claims["sub"]).isdigit() or not claims.get("email") or not claims.get("role"):
            return None
        return claims
    except (ValueError, TypeError, KeyError, UnicodeError, json.JSONDecodeError):
        return None
