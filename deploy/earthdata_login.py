"""Private, interactive Earthdata login for the research worker, never the website.

NASA's documented find_or_create_token endpoint avoids retaining a password.
Only a revocable token is stored, outside the repository and web service paths.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
from urllib.parse import urljoin, urlsplit

import requests

CREDENTIAL_PATH = Path('/var/lib/lst-data/earthdata/credential.json')
TOKEN_URL = 'https://urs.earthdata.nasa.gov/api/users/find_or_create_token'
DATA_HOST = 'data.lpdaac.earthdatacloud.nasa.gov'
ASSETS = {
    'ECOSTRESS': 'https://' + DATA_HOST + '/lp-prod-protected/ECO_L2T_LSTE.003/'
    'ECOv003_L2T_LSTE_42449_008_30UXC_20251231T030706_03/'
    'ECOv003_L2T_LSTE_42449_008_30UXC_20251231T030706_03_LST.tif',
    'ASTER': 'https://' + DATA_HOST + '/lp-prod-protected/AST_08.004/'
    'AST_08_00408012024205527_20251114030125/'
    'AST_08_00408012024205527_20251114030125_SKT.tif',
}


class LoginError(Exception):
    """Only fixed, non-secret messages may be passed to this exception."""


def obtain_token(session, username: str, password: str) -> dict:
    # Never follow a redirect carrying a Basic password, or print response bodies.
    with session.post(TOKEN_URL, auth=(username, password), timeout=(10, 30),
                      allow_redirects=False) as response:
        if response.status_code == 401:
            raise LoginError('NASA rejected the login. Check your username, password and email activation.')
        if response.status_code != 200:
            raise LoginError('NASA could not issue a token. Try again later, or use --token with a token from your Earthdata profile.')
        payload = response.json()
    token = payload.get('access_token')
    if not isinstance(token, str) or not token or any(c.isspace() for c in token):
        raise LoginError('NASA returned an unusable token. Nothing was saved.')
    return {'access_token': token, 'expiration_date': payload.get('expiration_date'),
            'saved_at_utc': datetime.now(timezone.utc).isoformat()}


def save_credential(credential: dict, path: Path = CREDENTIAL_PATH) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    fd, temporary = tempfile.mkstemp(prefix='.login-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            os.fchmod(handle.fileno(), 0o600)
            json.dump(credential, handle)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def probe_asset(session, url: str, token: str) -> str:
    """Read at most 16 TIFF header bytes; never read a training label or full file."""
    for _ in range(4):
        parsed = urlsplit(url)
        host = parsed.hostname or ''
        if (parsed.scheme != 'https' or parsed.port not in (None, 443)
                or parsed.username or parsed.password):
            return 'unverified: unexpected download destination'
        # A protected NASA asset may redirect to a signed S3 URL. The bearer
        # token is sent only to the exact LP DAAC host, never to that redirect.
        if host != DATA_HOST and not (host.endswith('.s3.us-west-2.amazonaws.com')
                                     or host.endswith('.s3.amazonaws.com')):
            return 'unverified: authorize LP DAAC in your Earthdata account'
        headers = {'Range': 'bytes=0-15', 'Accept-Encoding': 'identity'}
        if host == DATA_HOST:
            headers['Authorization'] = 'Bearer ' + token
        with session.get(url, headers=headers, stream=True, timeout=(10, 30),
                         allow_redirects=False) as response:
            if response.status_code in (301, 302, 303, 307, 308):
                url = urljoin(url, response.headers.get('Location', ''))
                continue
            if response.status_code in (401, 403):
                return 'unverified: token or LP DAAC authorization was rejected'
            if response.status_code not in (200, 206):
                return f'unverified: dataset returned HTTP {response.status_code}'
            signature = response.raw.read(16)
            if signature[:4] in (b'II*\x00', b'MM\x00*', b'II+\x00', b'MM\x00+'):
                return 'verified (TIFF header only; no thermal pixels downloaded)'
            return 'unverified: response was not a TIFF file'
    return 'unverified: too many download redirects'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='Check saved access without showing credentials.')
    mode.add_argument('--token', action='store_true', help='Paste a profile-generated token into a hidden prompt.')
    args = parser.parse_args()
    os.umask(0o077)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    with requests.Session() as session:
        # Do not pick up unrelated .netrc credentials or shell proxies.
        session.trust_env = False
        if args.check:
            if not CREDENTIAL_PATH.is_file():
                print('Earthdata is not connected yet. Run lst-earthdata-login in your own terminal.')
                return 2
            credential = json.loads(CREDENTIAL_PATH.read_text())
        else:
            if not sys.stdin.isatty() or not sys.stdout.isatty():
                raise LoginError('Run this in your own terminal: ssh -t hetzner2 lst-earthdata-login')
            print('Private NASA Earthdata login for LST research on Hetzner.')
            print('Your password is sent only to NASA and is never saved. Only a revocable token is stored.')
            if args.token:
                token = getpass.getpass('Earthdata token (hidden): ').strip()
                if not token or any(c.isspace() for c in token):
                    raise LoginError('The token is empty or contains whitespace. Nothing was saved.')
                credential = {'access_token': token, 'expiration_date': None,
                              'saved_at_utc': datetime.now(timezone.utc).isoformat()}
            else:
                username = input('Earthdata username: ').strip()
                password = getpass.getpass('Earthdata password (hidden): ')
                credential = obtain_token(session, username, password)
                del password
            save_credential(credential)
            print('Earthdata token saved privately for the research worker.')
        expiry = credential.get('expiration_date')
        if expiry and isinstance(expiry, str) and len(expiry) <= 32:
            print('NASA token expiration: ' + expiry)
        results = {}
        for label, asset in ASSETS.items():
            try:
                results[label] = probe_asset(session, asset, credential['access_token'])
            except (requests.RequestException, OSError, ValueError):
                results[label] = 'unverified: temporary network or response error; retry --check'
            print(label + ': ' + results[label])
        if all(value.startswith('verified') for value in results.values()):
            print('Both dataset connections are ready. Tell Codex: connected.')
            return 0
        print('Login is saved, but dataset access still needs checking. Share only these status messages, never your token.')
        return 3


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except LoginError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
    except (KeyboardInterrupt, EOFError):
        print('\nLogin cancelled.', file=sys.stderr)
        raise SystemExit(1)
    except Exception:
        # A traceback or server response could contain a request header/secret.
        print('Earthdata setup failed. No credentials are shown. Retry, or report this message.', file=sys.stderr)
        raise SystemExit(1)
