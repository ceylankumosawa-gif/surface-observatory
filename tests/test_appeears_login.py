import importlib.util
import io
import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace
import warnings

import pytest

spec = importlib.util.spec_from_file_location('appeears_login', Path(__file__).parents[1] / 'deploy/appeears_login.py')
login = importlib.util.module_from_spec(spec)
spec.loader.exec_module(login)

SECRET = 'mock-sensitive-password'
TOKEN = 'mock.token-private'


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError('Network is forbidden in these mock tests')
    monkeypatch.setattr(socket.socket, 'connect', deny)
    monkeypatch.setattr(socket.socket, 'connect_ex', deny)
    monkeypatch.setattr(socket, 'getaddrinfo', deny)


def payload(**updates):
    return {'token_type': 'Bearer', 'token': TOKEN, 'expiration': '2099-01-01T00:00:00Z', **updates}


class Response:
    def __init__(self, status=200, data=None, body=None):
        self.status_code = status
        self.body = body if body is not None else json.dumps(payload() if data is None else data).encode()
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True

    def iter_content(self, chunk_size):
        for i in range(0, len(self.body), chunk_size):
            yield self.body[i:i + chunk_size]


def session_for(response, calls=None):
    def post(url, **kwargs):
        if calls is not None:
            calls.append((url, kwargs))
        return response
    return SimpleNamespace(post=post)


def test_host_transport_and_only_token_fields_saved(tmp_path):
    calls = []
    response = Response(data=payload(username='mock-user', password=SECRET, arbitrary='discard-me'))
    result = login.obtain_token(session_for(response, calls), 'mock-user', SECRET)
    path = tmp_path / 'private' / 'credential.json'
    login.save_credential(result, path)
    assert len(calls) == 1 and calls[0][0] == 'https://appeears.earthdatacloud.nasa.gov/api/login'
    kwargs = calls[0][1]
    assert kwargs['auth'] == ('mock-user', SECRET)
    assert kwargs['allow_redirects'] is False and kwargs['verify'] is True
    assert kwargs['stream'] is True and kwargs['timeout'] == (10, 30)
    saved = path.read_text()
    assert all(value not in saved for value in (SECRET, 'mock-user', 'discard-me'))
    assert json.loads(saved)['token'] == TOKEN
    assert set(json.loads(saved)) == {'service', 'token_type', 'token', 'expiration', 'saved_at_utc'}
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert path.stat().st_uid == os.geteuid() and response.closed


@pytest.mark.parametrize('status', [301, 302, 307, 401, 403, 429, 500])
def test_status_error_redacts_body_and_never_follows(status):
    calls = []
    response = Response(status, body=(SECRET + TOKEN).encode())
    with pytest.raises(login.LoginError) as error:
        login.obtain_token(session_for(response, calls), 'mock-user', SECRET)
    assert SECRET not in str(error.value) and TOKEN not in str(error.value)
    assert len(calls) == 1 and response.closed


@pytest.mark.parametrize('body', [b'not-json-private', b'"mock-sensitive-password"', b'\xff', b'x' * 65537])
def test_invalid_or_oversized_response_redacted(body):
    with pytest.raises(login.LoginError) as error:
        login.obtain_token(session_for(Response(body=body)), 'mock-user', SECRET)
    assert SECRET not in str(error.value) and 'not-json-private' not in str(error.value)


@pytest.mark.parametrize('change', [
    {'token': ''}, {'token': 'with space'}, {'token': 'injected\nvalue'}, {'token': 'é'},
    {'token': 'x' * 16385}, {'token_type': 'Other'}, {'expiration': None},
    {'expiration': '2099-01-01T00:00:00'}, {'expiration': '2000-01-01T00:00:00Z'},
    {'expiration': SECRET},
])
def test_invalid_token_or_expiry_never_accepted(change):
    with pytest.raises(login.LoginError) as error:
        login.validate_token(payload(**change))
    assert SECRET not in str(error.value)


def test_network_exception_redacted():
    def post(*args, **kwargs):
        raise login.requests.ConnectionError(SECRET + TOKEN)
    with pytest.raises(login.LoginError) as error:
        login.obtain_token(SimpleNamespace(post=post), 'mock-user', SECRET)
    assert SECRET not in str(error.value) and TOKEN not in str(error.value)


def test_atomic_replacement_failure_preserves_old_token(tmp_path, monkeypatch):
    path = tmp_path / 'private' / 'credential.json'
    login.save_credential({'token': 'old-mock-token'}, path)
    original = path.read_bytes()
    def fail(*args, **kwargs):
        raise OSError(SECRET)
    monkeypatch.setattr(login.os, 'replace', fail)
    with pytest.raises(OSError):
        login.save_credential({'token': 'new-mock-token'}, path)
    assert path.read_bytes() == original
    assert list(path.parent.iterdir()) == [path]
    assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize('kind', ['symlink_file', 'symlink_directory', 'hardlink_file'])
def test_unsafe_storage_rejected(tmp_path, kind):
    other = tmp_path / 'unrelated-nasa-credential'
    other.write_text('untouched')
    directory = tmp_path / 'private'
    directory.mkdir()
    path = directory / 'credential.json'
    if kind == 'symlink_file':
        path.symlink_to(other)
    elif kind == 'hardlink_file':
        os.link(other, path)
    else:
        alias = tmp_path / 'alias'
        alias.symlink_to(directory, target_is_directory=True)
        path = alias / 'credential.json'
    with pytest.raises(login.LoginError):
        login.save_credential({'token': TOKEN}, path)
    assert other.read_text() == 'untouched'


def configure_interactive(monkeypatch):
    monkeypatch.setattr(login.pwd, 'getpwuid', lambda _: SimpleNamespace(pw_name='lstdata'))
    monkeypatch.setattr(login.resource, 'setrlimit', lambda *args: None)
    monkeypatch.setattr(login.os, 'umask', lambda *args: None)
    monkeypatch.setattr(login.logging, 'disable', lambda *args: None)
    monkeypatch.setattr(login.sys, 'stdin', SimpleNamespace(isatty=lambda: True))
    out = io.StringIO()
    out.isatty = lambda: True
    monkeypatch.setattr(login.sys, 'stdout', out)
    monkeypatch.setattr('builtins.input', lambda _: 'mock-user')
    monkeypatch.setattr(login.getpass, 'getpass', lambda _: SECRET)
    return out


def test_main_disables_environment_and_does_not_probe_or_print_secrets(monkeypatch, tmp_path, capsys):
    out = configure_interactive(monkeypatch)
    fake = session_for(Response())
    class Session:
        trust_env = True
        post = staticmethod(fake.post)
        def __enter__(self): return self
        def __exit__(self, *_): pass
    instance = Session()
    monkeypatch.setattr(login.requests, 'Session', lambda: instance)
    saved = []
    monkeypatch.setattr(login, 'save_credential', saved.append)
    assert login.run([]) == 0
    assert instance.trust_env is False and len(saved) == 1
    assert TOKEN not in out.getvalue() and SECRET not in out.getvalue()


def test_hidden_prompt_failure_aborts_before_network(monkeypatch):
    configure_interactive(monkeypatch)
    def unsafe_prompt(_):
        warnings.warn('echo unavailable', login.getpass.GetPassWarning)
    monkeypatch.setattr(login.getpass, 'getpass', unsafe_prompt)
    assert login.run([]) == 1


def test_unexpected_exception_output_is_redacted(monkeypatch, capsys):
    def fail(_):
        raise RuntimeError(SECRET + TOKEN)
    monkeypatch.setattr(login, 'main', fail)
    assert login.run([]) == 1
    output = capsys.readouterr()
    assert SECRET not in output.err and TOKEN not in output.err and 'Traceback' not in output.err


def test_noninteractive_refused_before_session_or_password(monkeypatch, capsys):
    configure_interactive(monkeypatch)
    monkeypatch.setattr(login.sys, 'stdin', SimpleNamespace(isatty=lambda: False))
    with pytest.raises(login.LoginError, match='own terminal'):
        login.main([])


def test_local_check_is_private_redacted_and_read_only(tmp_path):
    path = tmp_path / 'private' / 'credential.json'
    login.save_credential(login.validate_token(payload()), path)
    original = path.read_bytes()
    result = login.check_saved_credential(path)
    assert result['expiration'] == '2099-01-01T00:00:00+00:00'
    assert path.read_bytes() == original


@pytest.mark.parametrize('mode', [0o644, 0o640, 0o666])
def test_check_rejects_exposed_saved_token(tmp_path, mode):
    path = tmp_path / 'private' / 'credential.json'
    login.save_credential(login.validate_token(payload()), path)
    path.chmod(mode)
    with pytest.raises(login.LoginError, match='unsafe'):
        login.check_saved_credential(path)


def test_check_invalid_saved_token_redacts_payload(tmp_path):
    path = tmp_path / 'private' / 'credential.json'
    login.save_credential({'service': 'AppEEARS', 'token': TOKEN, 'expiration': SECRET}, path)
    with pytest.raises(login.LoginError) as error:
        login.check_saved_credential(path)
    assert SECRET not in str(error.value) and TOKEN not in str(error.value)


def test_check_does_not_need_tty_or_session(monkeypatch):
    out = configure_interactive(monkeypatch)
    monkeypatch.setattr(login.sys, 'stdin', SimpleNamespace(isatty=lambda: False))
    monkeypatch.setattr(login, 'check_saved_credential', lambda: login.validate_token(payload()))
    def fail():
        raise AssertionError('No authentication is permitted for --check')
    monkeypatch.setattr(login.requests, 'Session', fail)
    assert login.run(['--check']) == 0
    assert TOKEN not in out.getvalue() and SECRET not in out.getvalue()
