"""Connect the research worker to Copernicus without exposing a key to the website."""
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

import requests

CREDENTIAL_PATH = Path('/var/lib/lst-data/copernicus/credential.json')
PROBE_URL = ('https://arco.datastores.ecmwf.int/cadl-arco-geo-043/arco/'
             'reanalysis_era5_land/sfc-skin-temperature/geoChunked.zarr/.zmetadata')


class LoginError(Exception):
    """Use fixed, non-secret messages only."""


def probe(session, token):
    # A fixed destination and disabled redirects prevent forwarding a secret.
    with session.get(PROBE_URL, headers={'Authorization': 'Bearer ' + token},
                     timeout=(10, 40), allow_redirects=False, stream=True) as response:
        if response.status_code == 401:
            raise LoginError('Copernicus rejected the key. Copy the API key from your CDS profile and retry.')
        if response.status_code == 403:
            raise LoginError('Copernicus denied access (HTTP 403). Check that you accepted the CDS terms and ERA5-Land licence at https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=download . If already accepted, report this status; licence consent is not the only possible cause. No new key was saved.')
        if response.status_code != 200:
            raise LoginError('The Copernicus metadata service is unavailable or returned an unexpected response. Retry later.')
        # Only store metadata is inspected; no weather arrays or thermal labels.
        payload = response.raw.read(1_048_577, decode_content=True)
        if len(payload) > 1_048_576:
            raise LoginError('The metadata response exceeded the verification limit. Nothing was saved.')
        metadata = json.loads(payload)
        if metadata.get('zarr_consolidated_format') != 1 or not isinstance(metadata.get('metadata'), dict):
            raise LoginError('The service did not return expected ERA5-Land metadata. Nothing was saved.')


def save_credential(token, path=CREDENTIAL_PATH):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    fd, temporary = tempfile.mkstemp(prefix='.login-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            os.fchmod(handle.fileno(), 0o600)
            json.dump({'token': token, 'saved_at_utc': datetime.now(timezone.utc).isoformat()}, handle)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if args.check:
        if not CREDENTIAL_PATH.is_file():
            print('Copernicus is not connected yet. Run ssh -t hetzner2 lst-cds-login in your own terminal.')
            return 2
        token = json.loads(CREDENTIAL_PATH.read_text())['token']
    else:
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            raise LoginError('Run this in your own terminal: ssh -t hetzner2 lst-cds-login')
        print('Private Copernicus connection for LST research on Hetzner.')
        print('Copy the API key under the Your profile tab at https://cds.climate.copernicus.eu/profile')
        print('Input is hidden. The key is stored outside the repository and website, accessible only to the research worker and administrator.')
        token = getpass.getpass('CDS API key (hidden): ').strip()
    if not isinstance(token, str) or not token or len(token) > 4096 or any(c.isspace() for c in token):
        raise LoginError('The key is empty or contains unexpected characters. Nothing was saved.')
    with requests.Session() as session:
        session.trust_env = False
        probe(session, token)
    if not args.check:
        save_credential(token)
    print('Copernicus connected. ERA5-Land metadata access verified; no thermal labels downloaded.')
    print('Tell Codex: connected.')
    return 0


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
        print('Copernicus setup failed. No credentials are shown. Retry or share only this status message.', file=sys.stderr)
        raise SystemExit(1)
