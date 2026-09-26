import json
from pathlib import Path

import numpy as np
import pytest
from rasterio.transform import from_origin

from lst_pilot.ecostress import AcquisitionError, NasaDownloader, aggregate, asset_links, asset_url_ok, native_valid, LAYERS

URL='https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/ECO_L2T_LSTE.002/scene/scene_LST.tif'


def clean(shape=(7,7)):
    return {'LST':np.full(shape,300.,dtype='float32'),'LST_err':np.ones(shape,dtype='float32'),
            'cloud':np.zeros(shape,dtype='uint8'),'water':np.zeros(shape,dtype='uint8'),
            'QC':np.full(shape,3 << 14,dtype='uint16'),'view_zenith':np.zeros(shape,dtype='float32')}


def test_cloud_layer_is_required_even_when_qc_says_good():
    raw=clean((15,15));raw['cloud'][7,7]=1
    valid,_=native_valid(raw)
    assert not valid[7,12]  # 350 m exactly is buffered.
    assert valid[7,13]      # 420 m is outside the buffer.
    assert valid[11,11]     # Euclidean distance, not a square dilation.
    raw['cloud'][7,7]=255
    assert np.array_equal(native_valid(raw)[0],valid)


def test_qc_good_low_flags_and_missing_uncertainty_water_bad_radiance_rejected():
    raw=clean()
    assert native_valid(raw)[0].all()
    raw['LST_err'][0,0]=np.nan;raw['LST_err'][0,1]=0;raw['LST_err'][0,2]=2.01
    raw['QC'][0,3]=4;raw['water'][0,4]=1;raw['view_zenith'][0,5]=31
    raw['LST'][0,6]=np.nan
    assert not native_valid(raw)[0][0].any()


def test_qc_zero_is_decoded_not_nodata_but_accuracy_code_is_ineligible():
    raw=clean(); raw['QC'][0,0]=0
    valid, checks=native_valid(raw)
    assert checks['qc'][0,0]  # Low mandatory/radiance flags still have valid meaning.
    assert not checks['qc_lst_accuracy'][0,0]  # High 00 means error >2 K.
    assert not valid[0,0]


def test_tiled_kelvin_not_double_scaled_and_partial_source_edges_rejected():
    raw=clean((2,2))
    values,_,_=aggregate(raw,from_origin(0,200,70,70),'EPSG:32614',[0,0,200,200],32614)
    assert values['lst_c'][0,0]==pytest.approx(26.85,abs=1e-4)
    assert np.isfinite(values['lst_c']).sum()==1


def test_native_mask_precedes_temperature_averaging():
    raw=clean((10,10));raw['LST'][0,0]=390;raw['water'][0,0]=1
    values,_,_=aggregate(raw,from_origin(0,100,10,10),'EPSG:32614',[0,0,100,100],32614)
    assert values['valid_fraction'][0,0]==pytest.approx(.99)
    assert values['lst_c'][0,0]==pytest.approx(26.85,abs=1e-4)


def test_target_support_threshold_and_maximum_source_error():
    raw=clean((10,10));raw['water'][0,:]=1
    raw['LST_err'][2,2]=1.9; raw['QC'][2,2]=1 << 14
    values,_,_=aggregate(raw,from_origin(0,100,10,10),'EPSG:32614',[0,0,100,100],32614)
    assert np.isfinite(values['lst_c'][0,0])
    assert values['max_source_lst_error_k'][0,0]==pytest.approx(1.9)
    raw['water'][1,0]=1
    values,_,_=aggregate(raw,from_origin(0,100,10,10),'EPSG:32614',[0,0,100,100],32614)
    assert values['valid_fraction'][0,0]==pytest.approx(.89)
    assert np.isnan(values['lst_c'][0,0])


class Response:
    def __init__(self,status=200,headers=None,body=b'II*\x00' + b'fake-data'):
        self.status_code=status;self.headers=headers or {};self.body=body
    def __enter__(self):return self
    def __exit__(self,*_):pass
    def iter_content(self,*_):yield self.body


def test_bearer_is_stripped_from_signed_s3_redirect_and_cache_verified(tmp_path):
    client=NasaDownloader(token='fake-secret');calls=[]
    answers=iter([Response(302,{'Location':'https://bucket.s3.us-west-2.amazonaws.com/file?signature=hidden'}),Response()])
    def get(url,**kw):
        calls.append((url,kw));return next(answers)
    client.session.get=get
    path=tmp_path/'LST.tif'
    info=client.download(URL,path)
    assert calls[0][1]['headers']['Authorization']=='Bearer fake-secret'
    assert 'Authorization' not in calls[1][1]['headers']
    assert 'hidden' not in json.dumps(info) and 'fake-secret' not in json.dumps(info)
    assert client.download(URL,path)==info and len(calls)==2
    path.write_bytes(b'corrupted')
    with pytest.raises(AcquisitionError,match='checksum'):
        client.download(URL,path)


@pytest.mark.parametrize('url',['http://data.lpdaac.earthdatacloud.nasa.gov/x','https://evil.invalid/x','https://data.lpdaac.earthdatacloud.nasa.gov:444/lp-prod-protected/ECO_L2T_LSTE.002/x','https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/OTHER/x'])
def test_unapproved_destination_rejected(url):
    assert not asset_url_ok(url,redirect=True)


def test_http_body_and_token_not_in_failed_download_error(tmp_path):
    client=NasaDownloader(token='fake-secret')
    client.session.get=lambda *a,**k:Response(401,body=b'fake-secret')
    with pytest.raises(AcquisitionError) as error:
        client.download(URL,tmp_path/'bad.tif')
    assert 'fake-secret' not in str(error.value)
    assert not list(tmp_path.iterdir())


def test_download_byte_cap_removes_partial_file(tmp_path):
    client=NasaDownloader(token='fake-secret',max_bytes=4)
    client.session.get=lambda *a,**k:Response()
    with pytest.raises(AcquisitionError,match='cap'):
        client.download(URL,tmp_path/'too-large.tif')
    assert not list(tmp_path.iterdir())


def test_geolocation_metadata_has_separate_path_type_and_budget(tmp_path):
    url='https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/ECO_L1B_GEO.002/scene/scene.h5.dmrpp'
    assert not asset_url_ok(url)
    assert asset_url_ok(url,kind='geolocation_metadata')
    assert not asset_url_ok(url.removesuffix('.dmrpp'),kind='geolocation_metadata')
    assert not asset_url_ok(URL,kind='geolocation_metadata')
    client=NasaDownloader(token='fake-secret')
    client.session.get=lambda *a,**k:Response(body=b'<?xml version="1.0"?><Dataset/>')
    result=client.download(url,tmp_path/'scene.dmrpp',kind='geolocation_metadata')
    assert result['bytes']<100
    assert 'fake-secret' not in json.dumps(result)


def test_verified_nasa_cdn_redirect_receives_no_bearer(tmp_path):
    client=NasaDownloader(token='fake-secret'); calls=[]
    redirect=URL.replace('data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected',
                        'd1nklfio7vscoe.cloudfront.net/s3-2d2df3a34830d5223d1e9547cd713408/lp-prod-protected.s3.us-west-2.amazonaws.com')+'?signed=hidden'
    answers=iter([Response(303,{'Location':redirect}),Response()])
    def get(url,**kwargs):calls.append(kwargs);return next(answers)
    client.session.get=get
    info=client.download(URL,tmp_path/'LST.tif')
    assert 'Authorization' not in calls[1]['headers']
    assert not asset_url_ok(redirect)
    assert not asset_url_ok(redirect.replace('d1nklfio7vscoe','unrecognized'),redirect=True)
    assert 'hidden' not in json.dumps(info)


def test_asset_pairing_requires_all_layers_same_granule_and_collection():
    title='scene'
    umm={'CollectionReference':{'ShortName':'ECO_L2T_LSTE','Version':'002'},'GranuleUR':title,
         'RelatedUrls':[{'Type':'GET DATA','URL':URL.replace('scene_LST.tif',f'scene_{k}.tif')} for k in LAYERS]}
    assert set(asset_links(umm))==set(LAYERS)
    umm['RelatedUrls'].pop()
    with pytest.raises(AcquisitionError,match='missing'):
        asset_links(umm)
