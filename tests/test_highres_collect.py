import json
from pathlib import Path

import pytest

from lst_pilot import highres_collect as c


def test_protected_urls_are_collection_scoped():
    nasa = 'https://data.lpdaac.earthdatacloud.nasa.gov'
    assert c.allowed(nasa + '/lp-prod-protected/AST_08.004/a/a_SKT.tif', 'aster_cog')
    assert not c.allowed(nasa + '/lp-prod-protected/AST_08.004/a/a_SKT.tif?token=secret', 'aster_cog')
    assert not c.allowed(nasa + '/lp-prod-protected/AST_08.003/a/a_SKT.tif', 'aster_cog')
    assert not c.allowed('https://evil.example/lp-prod-protected/AST_08.004/a/a.tif', 'aster_cog', True)
    assert not c.allowed(nasa + '/lp-prod-protected/ECO_L2T_LSTE.002/a/a.tif', 'aster_cog')


def test_ledger_survives_restart_and_prevents_new_request(tmp_path):
    state = {'charged_bytes': c.CAP, 'requests': 0, 'completed': []}
    (tmp_path / 'download_ledger.json').write_text(json.dumps(state))
    client = c.Downloader(tmp_path, token='not-a-real-token')
    with pytest.raises(c.eco.AcquisitionError, match='ceiling'):
        client.download('https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/AST_08.004/a/a.tif', tmp_path/'a.tif', 'aster_cog')
    assert client.state['requests'] == 0


def test_tampered_cache_is_rejected(tmp_path):
    path = tmp_path / 'a.tif'
    path.write_bytes(b'bad')
    url = 'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/AST_08.004/a/a.tif'
    path.with_suffix('.download.json').write_text(json.dumps({'url': url, 'sha256': 'wrong'}))
    client = c.Downloader(tmp_path, token='not-a-real-token')
    with pytest.raises(c.eco.AcquisitionError, match='checksum'):
        client.download(url, path, 'aster_cog')
    assert client.state['requests'] == 0
