from pathlib import Path
import pytest
from lst_pilot.coarse_download import allowed, Downloader, DownloadError, authorized_years


def record(product='MOD21'):
    version='002' if product=='VNP21' else '061'
    stem=f'{product}.A2021196.1105.{version}.2021196201238'
    suffix='.nc' if product=='VNP21' else '.hdf'
    prefix='https://data.laadsdaac.earthdatacloud.nasa.gov/prod-lads/MOD03/' if product=='MOD03' else f'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/{product}.{version}/{stem}/'
    return {'product':product,'version':version,'stem':stem,'granule_start_utc':'2021-07-15T11:05:00Z','asset_url':prefix+stem+suffix}


def test_exact_hosts_and_product_paths():
    r=record()
    assert allowed(r['asset_url'],r)
    assert not allowed(r['asset_url'].replace('data.lpdaac.earthdatacloud.nasa.gov','evil.example'),r)
    assert not allowed(r['asset_url'].replace('/MOD21.061/','/ECO_L2T_LSTE.002/'),r)
    assert not allowed(r['asset_url']+'?secret=x',r)
    assert allowed(r['asset_url']+'?signed=x',r,redirect=True)
    assert not allowed(r['asset_url'].replace('https://','http://'),r)
    assert not allowed(r['asset_url'].replace('https://','https://user@'),r)
    assert allowed(record('MOD03')['asset_url'],record('MOD03'))


def test_cdns_require_exact_observed_origin_and_identity():
    r=record(); url='https://d1nklfio7vscoe.cloudfront.net/s3-'+'a'*32+'/lp-prod-protected.s3.us-west-2.amazonaws.com/MOD21.061/'+r['stem']+'/'+r['stem']+'.hdf?signature=x'
    assert allowed(url,r,True)
    assert not allowed(url,r,False)
    assert not allowed(url.replace('d1nklfio7vscoe','other'),r,True)
    assert not allowed(url,record('MOD03'),True)


def test_budget_survives_restart_and_heldout_year_rejected_before_network(tmp_path):
    plan={'max_total_protected_bytes':10}
    d=Downloader(tmp_path,plan,token='test');d.account(size=7)
    next_d=Downloader(tmp_path,plan,token='test')
    assert next_d.budget['bytes_received']==7
    with pytest.raises(DownloadError,match='cap'):next_d.account(size=4)
    r=record();r['granule_start_utc']='2023-07-15T11:05:00Z'
    with pytest.raises(DownloadError,match='2021'):next_d.download(r)


def test_auth_does_not_follow_cdn_redirect(monkeypatch,tmp_path):
    r=record();r.update(cmr_metadata={'umm':{'DataGranule':{}}},granule_id='G-test',cmr_revision=1)
    cdn='https://d1nklfio7vscoe.cloudfront.net'+r['asset_url'].split('.nasa.gov')[1]+'?signature=test'
    requests=[]
    class Response:
        def __init__(self,n):self.status_code=302 if n==0 else 200;self.headers={'Location':cdn} if n==0 else {'Content-Length':'8'}
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def iter_content(self,*args):yield b'\x0e\x03\x13\x01test'
    class Session:
        cookies=type('Cookies',(),{'clear':lambda self:None})()
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def get(self,url,**kwargs):requests.append((url,kwargs));return Response(len(requests)-1)
    monkeypatch.setattr('lst_pilot.coarse_download.requests.Session',Session)
    d=Downloader(tmp_path,{'max_total_protected_bytes':100,'max_file_bytes':{'MOD21':50}},token='private-test')
    assert d.download(r)['bytes']==8
    assert requests[0][1]['headers']['Authorization']=='Bearer private-test'
    assert 'Authorization' not in requests[1][1]['headers']
    assert requests[1][1]['allow_redirects'] is False
    assert 'signature' not in Path(tmp_path/'assets'/(r['stem']+'.hdf.json')).read_text()


def test_2023_requires_predeclared_protocol_and_2024_never_enabled(tmp_path):
    assert authorized_years({})=={2021,2022}
    with pytest.raises(DownloadError,match='protocol'):authorized_years({'thermal_years':[2023]})
    fake=tmp_path/'fake.md';fake.write_text('not the frozen protocol')
    with pytest.raises(DownloadError,match='protocol'):
        authorized_years({'thermal_years':[2023],'H_protocol':{'path':str(fake),'sha256':'fake'}})
    with pytest.raises(DownloadError,match='2021'):authorized_years({'thermal_years':[2024]})


def test_shared_cumulative_budget_keeps_symlink_and_prior_bytes(tmp_path):
    ledger=tmp_path/'ledger.json';ledger.write_text('{"bytes_received":7,"requests":1}')
    phase=tmp_path/'phase';phase.mkdir();(phase/'transfer_budget.json').symlink_to(ledger)
    d=Downloader(phase,{'max_total_protected_bytes':100},token='test');d.account(size=5)
    assert (phase/'transfer_budget.json').is_symlink()
    import json
    assert json.loads(ledger.read_text())['bytes_received']==12
