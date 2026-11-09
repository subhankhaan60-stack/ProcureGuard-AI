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
