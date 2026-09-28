"""Central password policy.

Every place that sets a password -- ``POST /api/v1/auth/register``,
``POST /api/v1/customers``, self-service change, admin reset, the
``vpp users`` CLI and the first-boot bootstrap -- calls
:func:`validate_password`, so the rules live in one place:

* at least ``VPP_PASSWORD_MIN_LENGTH`` characters (default 12);
* at most 72 bytes UTF-8 encoded (bcrypt's input limit -- longer inputs
  would otherwise be silently truncated or rejected by the hash);
* at least 5 distinct characters (rejects ``aaaaaaaaaaaa``, ``121212...``);
* not a well-known password, even with digits/symbols appended
  (``password123!``, ``letmein2024``);
* not containing the username.

Deliberately no composition rules (upper/lower/digit/symbol): length and a
deny-list stop the guessable passwords without pushing users to
``Password1!`` (NIST SP 800-63B, section 5.1.1.2).
"""

from __future__ import annotations

import re

from vpp.settings import get_settings

#: bcrypt only hashes the first 72 bytes of its input.
MAX_PASSWORD_BYTES = 72
MIN_DISTINCT_CHARS = 5

# Compared lower-cased, with symbols removed and trailing digits dropped.
_COMMON_PASSWORDS = frozenset(
    {
        "password",
        "passw0rd",
        "passwort",
        "motdepasse",
        "contrasena",
        "qwerty",
        "qwertyuiop",
        "asdfgh",
        "asdfghjkl",
        "zxcvbnm",
        "azerty",
        "letmein",
        "welcome",
        "admin",
        "administrator",
        "root",
        "toor",
        "changeme",
        "changemenow",
        "default",
        "secret",
        "iloveyou",
        "monkey",
        "dragon",
        "master",
        "sunshine",
        "princess",
        "football",
        "baseball",
        "superman",
        "trustno",
        "abc",
        "abcdef",
        "abcdefgh",
        "abcdefghijkl",
        "vpp",
        "virtualpowerplant",
        "powerplant",
        "operator",
        "viewer",
        "guest",
        "test",
        "testing",
        "login",
        "user",
        "hello",
        "freedom",
        "whatever",
        "starwars",
    }
)

_LEET = str.maketrans({"@": "a", "4": "a", "3": "e", "1": "i", "0": "o", "$": "s", "5": "s"})


def _is_common(password: str) -> bool:
    lowered = password.lower()
    if lowered in _COMMON_PASSWORDS:
        return True
    core = re.sub(r"[^a-z0-9@$]", "", lowered)
    stripped = re.sub(r"[^a-z0-9]", "", core).rstrip("0123456789")
    return (
        stripped in _COMMON_PASSWORDS
        or core.translate(_LEET).rstrip("0123456789") in _COMMON_PASSWORDS
        or re.sub(r"[^a-z]", "", core.translate(_LEET)) in _COMMON_PASSWORDS
    )


class PasswordPolicyError(ValueError):
    """Raised with a human-readable reason when a password is rejected."""


def validate_password(password: str, *, username: str | None = None) -> None:
    """Raise :class:`PasswordPolicyError` unless ``password`` meets the policy."""
    min_length = max(1, get_settings().password_min_length)
    if len(password) < min_length:
        raise PasswordPolicyError(f"Password must be at least {min_length} characters long")
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise PasswordPolicyError(f"Password must be at most {MAX_PASSWORD_BYTES} bytes long")
    if len(set(password)) < MIN_DISTINCT_CHARS:
        raise PasswordPolicyError(
            f"Password must contain at least {MIN_DISTINCT_CHARS} different characters"
        )
    if _is_common(password):
        raise PasswordPolicyError("Password is too common")
    if username and len(username) >= 3 and username.lower() in password.lower():
        raise PasswordPolicyError("Password must not contain the username")


def password_problem(password: str, *, username: str | None = None) -> str | None:
    """Return why ``password`` is rejected, or ``None`` if it is acceptable."""
    try:
        validate_password(password, username=username)
    except PasswordPolicyError as exc:
        return str(exc)
    return None
