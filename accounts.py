"""Comptes utilisateurs : mot de passe haché et cookie signé."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def data_root() -> Path:
    path = Path(os.environ.get("DATA_DIR", ROOT))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _secret() -> bytes:
    configured = os.environ.get("SESSION_SECRET", "").strip()
    if configured:
        return configured.encode()
    secret_path = data_root() / ".session_secret"
    if secret_path.exists():
        return secret_path.read_bytes()
    value = secrets.token_hex(32).encode()
    secret_path.write_bytes(value)
    return value


def _db() -> sqlite3.Connection:
    connection = sqlite3.connect(data_root() / "users.db")
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
        """
    )
    return connection


def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 200_000)
    return f"{salt}${digest.hex()}"


def _check_password(password: str, stored: str) -> bool:
    salt, digest = stored.split("$", 1)
    check = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 200_000)
    return hmac.compare_digest(check.hex(), digest)


def sign_user(user_id: str) -> str:
    mac = hmac.new(_secret(), user_id.encode(), hashlib.sha256).hexdigest()
    return f"{user_id}.{mac}"


def user_id_from_cookie(value: str | None) -> str | None:
    if not value or "." not in value:
        return None
    user_id, mac = value.split(".", 1)
    if not re.fullmatch(r"[0-9a-f]{32}", user_id):
        return None
    expected = hmac.new(_secret(), user_id.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(mac, expected):
        return None
    if get_user(user_id) is None:
        return None
    return user_id


def get_user(user_id: str) -> dict[str, str] | None:
    with _db() as connection:
        row = connection.execute(
            "SELECT id, name, email FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
    if row is None:
        return None
    return {"id": row["id"], "name": row["name"], "email": row["email"]}


def register(name: str, email: str, password: str) -> tuple[dict[str, str] | None, str | None]:
    name = " ".join(name.split())
    email = email.strip().lower()
    if len(name) < 2:
        return None, "Indiquez votre nom."
    if not EMAIL.fullmatch(email):
        return None, "Indiquez une adresse e-mail valide."
    if len(password) < 8:
        return None, "Le mot de passe doit contenir au moins 8 caractères."
    user_id = secrets.token_hex(16)
    try:
        with _db() as connection:
            connection.execute(
                "INSERT INTO users (id, name, email, password_hash) VALUES (?, ?, ?, ?)",
                (user_id, name, email, _hash_password(password)),
            )
    except sqlite3.IntegrityError:
        return None, "Un compte existe déjà avec cette adresse e-mail."
    return {"id": user_id, "name": name, "email": email}, None


def authenticate(email: str, password: str) -> dict[str, str] | None:
    email = email.strip().lower()
    with _db() as connection:
        row = connection.execute(
            "SELECT id, name, email, password_hash FROM users WHERE email = ?",
            (email,),
        ).fetchone()
    if row is None or not _check_password(password, row["password_hash"]):
        return None
    return {"id": row["id"], "name": row["name"], "email": row["email"]}
