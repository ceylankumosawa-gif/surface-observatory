import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('earthdata_login', Path(__file__).parents[1] / 'deploy/earthdata_login.py')
login = importlib.util.module_from_spec(spec)
spec.loader.exec_module(login)


class Response:
    def __init__(self, status, payload=None, headers=None, body=b'II*\x00' + b'\x00' * 20):
        self.status_code = status
        self.payload = payload
        self.headers = headers or {}
        self.raw = io.BytesIO(body)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def json(self):
        return self.payload


def test_password_only_sent_to_nasa_and_never_persisted(tmp_path):
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return Response(200, {'access_token': 'example-token', 'expiration_date': '11/01/2026'})

    credential = login.obtain_token(SimpleNamespace(post=post), 'example-user', 'example-password')
    path = tmp_path / 'private' / 'credential.json'
    login.save_credential(credential, path)
    assert calls[0][0] == login.TOKEN_URL
    assert calls[0][1]['allow_redirects'] is False
    assert 'example-password' not in path.read_text()
    assert 'example-user' not in path.read_text()
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700


def test_download_redirect_does_not_forward_bearer_or_read_pixels():
    calls = []
    data = Response(206)
    responses = iter([Response(302, headers={'Location': 'https://lp-prod-protected.s3.us-west-2.amazonaws.com/file.tif?signature=private'}), data])

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return next(responses)

    status = login.probe_asset(SimpleNamespace(get=get), login.ASSETS['ECOSTRESS'], 'example-token')
    assert status.startswith('verified')
    assert calls[0][1]['headers']['Authorization'] == 'Bearer example-token'
    assert 'Authorization' not in calls[1][1]['headers']
    assert data.raw.tell() == 16


@pytest.mark.parametrize('destination', ['https://evil.invalid/x', 'http://data.lpdaac.earthdatacloud.nasa.gov/x', 'https://urs.earthdata.nasa.gov/oauth/authorize', 'https://data.lpdaac.earthdatacloud.nasa.gov:444/x'])
def test_unexpected_redirect_never_contacted(destination):
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        return Response(302, headers={'Location': destination})

    assert login.probe_asset(SimpleNamespace(get=get), login.ASSETS['ASTER'], 'example-token').startswith('unverified')
    assert len(calls) == 1


def test_failed_nasa_response_does_not_echo_body_or_save(tmp_path):
    session = SimpleNamespace(post=lambda *a, **k: Response(401, {'secret': 'example-password'}))
    with pytest.raises(login.LoginError) as error:
        login.obtain_token(session, 'example-user', 'example-password')
    assert 'example-password' not in str(error.value)
