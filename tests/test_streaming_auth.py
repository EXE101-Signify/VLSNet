import base64
import hashlib
import hmac
import json
import unittest

from mode.streaming_auth import SessionTokenError, verify_session_token


SECRET = 'streaming-test-secret-with-at-least-32-bytes'
SESSION = 'd2e2187c-e9d2-45a4-b420-77f0a0ac2075'


def token(**changes):
    header = {'alg': 'HS256', 'typ': 'JWT'}
    claims = {'iss': 'signify-be-vsl', 'aud': ['vsl-net'], 'scope': 'vsl:stream',
              'sub': '20', 'callId': 42, 'sessionId': SESSION, 'iat': 1000, 'exp': 1300}
    claims.update(changes)
    encode = lambda value: base64.urlsafe_b64encode(json.dumps(value).encode()).rstrip(b'=').decode()
    signed = f'{encode(header)}.{encode(claims)}'
    signature = base64.urlsafe_b64encode(hmac.new(SECRET.encode(), signed.encode(), hashlib.sha256).digest()).rstrip(b'=').decode()
    return f'{signed}.{signature}'


class StreamingTokenTests(unittest.TestCase):
    def test_valid_token_is_bound_to_call_signer_and_session(self):
        claims = verify_session_token(token(), SECRET, now=1100)
        self.assertEqual((claims.call_id, claims.signer_id, claims.session_id), (42, 20, SESSION))

    def test_tampered_expired_or_wrong_scope_token_is_rejected(self):
        for value, now in [(token() + 'a', 1100), (token(), 1300),
                           (token(scope='vsl:prediction'), 1100), (token(callId=0), 1100)]:
            with self.assertRaises(SessionTokenError):
                verify_session_token(value, SECRET, now=now)


if __name__ == '__main__':
    unittest.main()
