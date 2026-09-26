"""Isolated, capped NASA native-swath download; no serving changes."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import threading
from urllib.parse import urlsplit, urljoin
import requests
from .coarse_inventory import PRODUCTS, identity, sha, save, utc

CREDENTIAL = Path('/var/lib/lst-data/earthdata/credential.json')
AUTH_HOSTS = {'data.lpdaac.earthdatacloud.nasa.gov', 'data.laadsdaac.earthdatacloud.nasa.gov'}
CDN = 'd1nklfio7vscoe.cloudfront.net'
LAADS_CDN = 'd13j1jds5ybppo.cloudfront.net'  # Exact MOD03 NASA 303 redirect, verified 2026-09-10.
H_PROTOCOL_SHA256 = '66fdd4deac2a55da6cee38c03a66cb84154d7595b1d90f7694da183a84848cee'


class DownloadError(ValueError):
    """Only safe, fixed diagnostics; never response contents or signed URLs."""


def authorized_years(plan):
    years=set(plan.get('thermal_years',[2021,2022]))
    if not years or not years<={2021,2022,2023}:raise DownloadError('Only explicitly frozen 2021–2023 thermal sources may be enabled.')
    if 2023 in years:
        protocol=plan.get('H_protocol',{})
        path=Path(protocol.get('path',''))
        if protocol.get('sha256')!=H_PROTOCOL_SHA256 or not path.is_file() or sha(path)!=H_PROTOCOL_SHA256:
            raise DownloadError('2023 requires the exact predeclared H protocol before thermal access.')
    return years


def allowed(url, record, redirect=False):
    p = urlsplit(url)
    if p.scheme != 'https' or p.port not in (None, 443) or p.username or p.password or p.fragment:
        return False
    filename = record['stem']+PRODUCTS[record['product']]['suffix']
    if not p.path.endswith('/'+filename) or (p.query and not redirect): return False
    product = record['product']; collection = product+'.'+record['version']
    if p.hostname == 'data.lpdaac.earthdatacloud.nasa.gov':
        return product in ('MOD21', 'VNP21') and p.path.startswith('/lp-prod-protected/'+collection+'/')
    if p.hostname == 'data.laadsdaac.earthdatacloud.nasa.gov':
        return product == 'MOD03' and p.path.startswith('/prod-lads/MOD03/')
    # This exact CDN is observed in NASA's LP DAAC redirect chain. Unknown LAADS
    # destinations are stopped and reported by hostname only for explicit review.
    if redirect and p.hostname == CDN and product in ('MOD21', 'VNP21'):
        return bool(p.path.startswith('/lp-prod-protected/'+collection+'/') or re.match(
            r'^/s3-[0-9a-f]{32}/lp-prod-protected\.s3\.us-west-2\.amazonaws\.com/'+re.escape(collection)+'/', p.path))
    if redirect and p.hostname == LAADS_CDN and product == 'MOD03':
        return bool(re.match(r'^/s3-[0-9a-f]{32}/prod-lads\.s3\.us-west-2\.amazonaws\.com/MOD03/', p.path))
    return False


class Downloader:
    def __init__(self, root, plan, token=None):
        self.root=Path(root); self.plan=plan; self.lock=threading.Lock()
        self.years=authorized_years(plan)
        # Resolve an explicitly shared budget symlink before atomic JSON writes.
        # Only one bounded acquisition service may own this ledger at a time.
        self.budget_path=(self.root/'transfer_budget.json').resolve()
        self.budget=json.loads(self.budget_path.read_text()) if self.budget_path.exists() else {'bytes_received':0, 'requests':0}
        self.token=token if token is not None else json.loads(CREDENTIAL.read_text())['access_token']

    def account(self, size=0, request=False):
        with self.lock:
            self.budget['bytes_received']+=size; self.budget['requests']+=int(request)
            save(self.budget_path, self.budget)
            if self.budget['bytes_received']>self.plan['max_total_protected_bytes'] or self.budget['requests']>self.plan.get('max_requests',180):
                raise DownloadError('Persistent protected transfer/request cap reached.')

    def download(self, record):
        if utc(record['granule_start_utc']).year not in self.years:
            raise DownloadError('This frozen plan does not enable the requested thermal year (default 2021–2022).')
        url=record['asset_url']
        if not allowed(url, record): raise DownloadError('Initial NASA asset identity/origin rejected.')
        filename=record['stem']+PRODUCTS[record['product']]['suffix']
        destination=self.root/'assets'/filename; sidecar=destination.with_suffix(destination.suffix+'.json')
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and sidecar.exists():
            previous=json.loads(sidecar.read_text())
            if previous['url']!=url or previous['sha256']!=sha(destination):
                raise DownloadError('Cached source identity/checksum mismatch.')
            return previous
        temp=destination.with_suffix(destination.suffix+'.partial')
        cap=self.plan['max_file_bytes'][record['product']]; total=0
        try:
            with requests.Session() as session:
                session.trust_env=False; current=url
                for _ in range(6):
                    if not allowed(current, record, redirect=True):
                        raise DownloadError('Unapproved NASA redirect host: '+str(urlsplit(current).hostname))
                    self.account(request=True); headers={'Accept-Encoding':'identity'}
                    if urlsplit(current).hostname in AUTH_HOSTS: headers['Authorization']='Bearer '+self.token
                    session.cookies.clear()
                    with session.get(current, headers=headers, stream=True, allow_redirects=False, timeout=(10,60)) as response:
                        if response.status_code in (301,302,303,307,308):
                            current=urljoin(current,response.headers.get('Location','')); continue
                        if response.status_code!=200: raise DownloadError(f'NASA native asset HTTP {response.status_code}; response content suppressed.')
                        length=int(response.headers.get('Content-Length','0'))
                        if length>cap: raise DownloadError('Native asset Content-Length exceeds frozen per-file cap.')
                        with temp.open('wb') as handle:
                            for chunk in response.iter_content(1024*1024):
                                self.account(size=len(chunk)); total+=len(chunk)
                                if total>cap: raise DownloadError('Native asset exceeded frozen per-file cap.')
                                handle.write(chunk)
                        with temp.open('rb') as handle: magic=handle.read(8)
                        expected=b'\x89HDF\r\n\x1a\n' if record['product']=='VNP21' else b'\x0e\x03\x13\x01'
                        if not magic.startswith(expected): raise DownloadError('Native asset container signature failed.')
                        if length and length!=total: raise DownloadError('Native asset response was truncated.')
                        # Validate source MD5 where CMR provides it (MOD03 exposes one).
                        infos=record['cmr_metadata']['umm']['DataGranule'].get('ArchiveAndDistributionInformation',[])
                        checks=[x['Checksum'] for x in infos if x.get('Checksum',{}).get('Algorithm','').upper()=='MD5']
                        if checks:
                            md5=hashlib.md5(temp.read_bytes()).hexdigest()
                            if any(x['Value'].lower()!=md5 for x in checks): raise DownloadError('Native source CMR MD5 mismatch.')
                        temp.replace(destination)
                        result={'url':url,'path':str(destination.resolve()),'bytes':total,'sha256':sha(destination),
                                'cmr_granule_id':record['granule_id'],'cmr_revision':record['cmr_revision'],'cmr_md5_checked':bool(checks)}
                        save(sidecar,result); return result
                raise DownloadError('Native asset exceeded redirect count cap.')
        except requests.RequestException:
            raise DownloadError('NASA native network request failed; private request details suppressed.') from None
        finally:
            if temp.exists(): temp.unlink()


def acquire(root):
    root=Path(root); plan_path=root/'engineering_plan.json'; plan=json.loads(plan_path.read_text())
    signature={'plan_sha256':sha(plan_path),'downloader_sha256':sha(__file__),
               'max_workers':2,'years':sorted(authorized_years(plan)),'protected_cap_bytes':plan['max_total_protected_bytes'],
               'canonical_budget_path':str((root/'transfer_budget.json').resolve())}
    signature_path=root/'acquisition_signature.json'
    if signature_path.exists():
        if json.loads(signature_path.read_text())!=signature: raise ValueError('Frozen acquisition signature changed.')
    else: save(signature_path,signature)
    downloader=Downloader(root,plan)
    def one(record):
        key=record['region_id']+'_'+record['stem']; path=root/'downloads'/f'{key}.json'
        try:
            asset=downloader.download(record); geo=None
            if 'geolocation_companion' in record: geo=downloader.download(record['geolocation_companion'])
            result={'status':'downloaded','record':record,'thermal_asset':asset,'geolocation_asset':geo}
        except DownloadError as exc:
            result={'status':'failed','record':record,'error':str(exc)}
        save(path,result); print(json.dumps({'key':key,'status':result['status'],'error':result.get('error')}),flush=True)
        return result
    with ThreadPoolExecutor(max_workers=2) as pool: records=list(pool.map(one,plan['records']))
    save(root/'download_manifest.json',{'signature':signature,'budget':downloader.budget,'records':records})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    acquire(p.parse_args().root)
