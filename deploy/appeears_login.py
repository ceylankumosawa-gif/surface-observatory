"""Interactive, private AppEEARS login; no subset submission or data downloads."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import getpass
import json
import logging
import os
from pathlib import Path
import pwd
import re
import resource
import secrets
import stat
import sys
import warnings

import requests

TOKEN_URL = 'https://appeears.earthdatacloud.nasa.gov/api/login'
CREDENTIAL_PATH = Path('/var/lib/lst-data/appeears/credential.json')
DATA_USER = 'lstdata'
MAX_RESPONSE_BYTES = 65536
COMMAND = 'ssh -t hetzner2 lst-appeears-login'


class LoginError(Exception):
    """Only fixed, non-secret messages may be used here."""


def validate_token(payload: object) -> dict:
    if not isinstance(payload, dict):
        raise LoginError('NASA returned an unusable AppEEARS token. Nothing was saved.')
    token = payload.get('token')
    if (payload.get('token_type') != 'Bearer' or not isinstance(token, str)
            or not 1 <= len(token) <= 16384
            or re.fullmatch(r'[A-Za-z0-9._~+/-]+=*', token) is None):
        raise LoginError('NASA returned an unusable AppEEARS token. Nothing was saved.')
    expiry = payload.get('expiration')
    try:
        if not isinstance(expiry, str) or len(expiry) > 40:
            raise ValueError
        expiry_time = datetime.fromisoformat(expiry.replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        if expiry_time.tzinfo is None or expiry_time <= now:
            raise ValueError
    except (ValueError, TypeError, OverflowError):
        raise LoginError('NASA returned an invalid or expired AppEEARS token. Nothing was saved.') from None
    # Whitelist fields: never persist the username, password, or response body.
    return {'service': 'AppEEARS', 'token_type': 'Bearer', 'token': token,
            'expiration': expiry_time.astimezone(timezone.utc).isoformat(),
            'saved_at_utc': now.isoformat()}


def obtain_token(session, username: str, password: str) -> dict:
    if not username or ':' in username or any(c.isspace() for c in username) or not password:
        raise LoginError('Enter a nonempty Earthdata username and password. Nothing was saved.')
    try:
        # No redirects, proxies/netrc, retries, or credentials in URL/query/logs.
        with session.post(TOKEN_URL, auth=(username, password), timeout=(10, 30),
                          allow_redirects=False, verify=True, stream=True,
                          headers={'Accept': 'application/json', 'Accept-Encoding': 'identity'}) as response:
            if response.status_code in (401, 403):
                raise LoginError('NASA rejected the AppEEARS login. Check your Earthdata account details.')
            if response.status_code != 200:
                raise LoginError('NASA could not issue an AppEEARS token. Try again later.')
            body = bytearray()
            for chunk in response.iter_content(chunk_size=4096):
                if len(body) + len(chunk) > MAX_RESPONSE_BYTES:
                    raise LoginError('NASA returned an oversized login response. Nothing was saved.')
                body.extend(chunk)
            payload = json.loads(body)
    except (requests.RequestException, ValueError, UnicodeError):
        raise LoginError('AppEEARS login failed because of a network or response error. Nothing was saved.') from None
    return validate_token(payload)


def save_credential(credential: dict, path: Path = CREDENTIAL_PATH) -> None:
    """Atomically replace only this private token; never read an existing token."""
    path = Path(path)
    # Reject a misconfigured symlink path, including existing ancestor links.
    for ancestor in (path.parent, *path.parent.parents):
        if ancestor.is_symlink():
            raise LoginError('The private token directory is unsafe. Nothing was saved.')
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    temporary = '.login-' + secrets.token_hex(16)
    try:
        if os.fstat(directory).st_uid != os.geteuid():
            raise LoginError('The private token directory has the wrong owner. Nothing was saved.')
        os.fchmod(directory, 0o700)
        try:
            existing = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None and (not stat.S_ISREG(existing.st_mode)
                                     or existing.st_uid != os.geteuid() or existing.st_nlink != 1):
            raise LoginError('The private token destination is unsafe. Nothing was saved.')
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=directory)
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            os.fchmod(handle.fileno(), 0o600)
            json.dump(credential, handle)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path.name, src_dir_fd=directory, dst_dir_fd=directory)
        os.fsync(directory)
    finally:
        try:
            os.unlink(temporary, dir_fd=directory)
        except FileNotFoundError:
            pass
        os.close(directory)


def check_saved_credential(path: Path = CREDENTIAL_PATH) -> dict:
    """Check only this AppEEARS token locally; never make an authentication call."""
    path = Path(path)
    for ancestor in (path.parent, *path.parent.parents):
        if ancestor.is_symlink():
            raise LoginError('The private token directory is unsafe.')
    try:
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except FileNotFoundError:
        raise LoginError('AppEEARS is not connected yet. Run: ' + COMMAND) from None
    try:
        parent = os.fstat(directory)
        if parent.st_uid != os.geteuid() or stat.S_IMODE(parent.st_mode) != 0o700:
            raise LoginError('The AppEEARS token directory has unsafe ownership or permissions.')
        try:
            descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        except FileNotFoundError:
            raise LoginError('AppEEARS is not connected yet. Run: ' + COMMAND) from None
        with os.fdopen(descriptor, 'rb') as handle:
            saved = os.fstat(handle.fileno())
            if (not stat.S_ISREG(saved.st_mode) or saved.st_uid != os.geteuid()
                    or stat.S_IMODE(saved.st_mode) != 0o600 or saved.st_nlink != 1
                    or saved.st_size > MAX_RESPONSE_BYTES):
                raise LoginError('The AppEEARS token file has unsafe ownership, permissions or format.')
            body = handle.read(MAX_RESPONSE_BYTES + 1)
            if len(body) > MAX_RESPONSE_BYTES:
                raise LoginError('The saved AppEEARS token has an invalid format.')
    finally:
        os.close(directory)
    try:
        payload = json.loads(body)
        if not isinstance(payload, dict) or payload.get('service') != 'AppEEARS':
            raise ValueError
        return validate_token(payload)
    except (ValueError, UnicodeError, LoginError):
        raise LoginError('The saved AppEEARS token is invalid or expired. Run: ' + COMMAND) from None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true',
                        help='Check the saved AppEEARS token and expiry locally; no network call.')
    args = parser.parse_args(argv)
    os.umask(0o077)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    logging.disable(logging.CRITICAL)
    if pwd.getpwuid(os.geteuid()).pw_name != DATA_USER:
        raise LoginError('Use the installed private launcher: ' + COMMAND)
    if args.check:
        credential = check_saved_credential()
        print('A private, unexpired AppEEARS token is saved.')
        print('Expires at: ' + credential['expiration'])
        print('Local check only; NASA acceptance was not tested. No subset task was submitted.')
        return 0
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise LoginError('Run this in your own terminal: ' + COMMAND)
    print('Private NASA AppEEARS login for LST research on Hetzner.')
    print('Your password is sent only to NASA, never saved. Only the AppEEARS token is stored.')
    username = input('Earthdata username: ').strip()
    with warnings.catch_warnings():
        # Fail instead of falling back to a possibly echoed password prompt.
        warnings.simplefilter('error', getpass.GetPassWarning)
        try:
            password = getpass.getpass('Earthdata password (hidden): ')
        except getpass.GetPassWarning:
            raise LoginError('A hidden password prompt is unavailable. Use your own terminal.') from None
    try:
        with requests.Session() as session:
            session.trust_env = False
            credential = obtain_token(session, username, password)
    finally:
        del password
    save_credential(credential)
    print('AppEEARS token saved privately for the research worker.')
    print('Expires at: ' + credential['expiration'])
    print('The existing Earthdata connection is unchanged. No subset task was submitted.')
    return 0


def run(argv=None) -> int:
    try:
        return main(argv)
    except LoginError as error:
        print(str(error), file=sys.stderr)
    except (KeyboardInterrupt, EOFError):
        print('\nAppEEARS login cancelled.', file=sys.stderr)
    except Exception:
        # Never emit a traceback, response body, exception text or request header.
        print('AppEEARS setup failed. No credentials are shown. Retry or report this message.', file=sys.stderr)
    return 1


if __name__ == '__main__':
    raise SystemExit(run())
