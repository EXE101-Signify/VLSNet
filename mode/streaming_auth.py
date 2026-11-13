"""Verify short-lived HS256 VSL session tokens issued by Signify backend."""
import base64
import hashlib
import hmac
import json
import time
import uuid
from dataclasses import dataclass


class SessionTokenError(ValueError):
    pass


@dataclass(frozen=True)
class SessionClaims:
    session_id: str
    call_id: int
    signer_id: int
    expires_at: int


def _decode(value: str) -> bytes:
    if not value or len(value) > 4096:
        raise SessionTokenError('Invalid token encoding')
    try:
        return base64.urlsafe_b64decode(value + '=' * (-len(value) % 4))
    except (ValueError, base64.binascii.Error) as exc:
        raise SessionTokenError('Invalid token encoding') from exc


def verify_session_token(token: str, secret: str, now: int | None = None) -> SessionClaims:
    if len(secret.encode()) < 32 or not token or len(token) > 8192:
        raise SessionTokenError('Streaming token unavailable')
    parts = token.split('.')
    if len(parts) != 3:
        raise SessionTokenError('Malformed token')
    try:
        signed = f'{parts[0]}.{parts[1]}'.encode('ascii')
    except UnicodeError as exc:
        raise SessionTokenError('Malformed token') from exc
    expected = hmac.new(secret.encode(), signed, hashlib.sha256).digest()
    if not hmac.compare_digest(expected, _decode(parts[2])):
        raise SessionTokenError('Invalid signature')
    try:
        header = json.loads(_decode(parts[0]))
        claims = json.loads(_decode(parts[1]))
        current = int(time.time()) if now is None else now
        issued, expires = claims['iat'], claims['exp']
        call_id, signer_id = claims['callId'], int(claims['sub'])
        session_id = str(uuid.UUID(claims['sessionId']))
        if (header.get('alg') != 'HS256' or claims.get('iss') != 'signify-be-vsl'
                or claims.get('aud') not in ('vsl-net', ['vsl-net'])
                or claims.get('scope') != 'vsl:stream'
                or type(issued) is not int or type(expires) is not int
                or type(call_id) is not int or call_id <= 0 or signer_id <= 0
                or issued > current + 10 or expires <= current
                or expires - issued > 300 or expires <= issued):
            raise SessionTokenError('Invalid token claims')
        return SessionClaims(session_id, call_id, signer_id, expires)
    except (KeyError, ValueError, TypeError, UnicodeError) as exc:
        raise SessionTokenError('Invalid token claims') from exc
