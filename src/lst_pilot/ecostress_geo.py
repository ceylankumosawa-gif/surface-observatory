"""Independent ECOSTRESS V002 geolocation metadata QA; never fetches labels.

Only anonymous CMR and the official public obstruction list can be fetched by
this module. Protected GEO HDF/DMR++ content must be supplied by the caller.
Unknown/missing QA is never a positive quality decision.
"""
from __future__ import annotations

import base64
import binascii
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET

import requests

CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
GEO_COLLECTION = "C2076087338-LPCLOUD"
GEO_HOST = "data.lpdaac.earthdatacloud.nasa.gov"
GEO_PREFIX = "/lp-prod-protected/ECO_L1B_GEO.002/"
OBSTRUCTION_URL = "https://lpdaac.usgs.gov/documents/2249/obst_all_sort.txt"
OFFICIAL_EXAMPLE = "https://github.com/nasa/ECOSTRESS-Data-Resources/blob/main/python/scripts/extract_geolocation_flag/ECOSTRESS_geolocation.py"
MAX_DMRPP_BYTES = 8 * 1024**2
MAX_OBSTRUCTION_BYTES = 16 * 1024**2
NAME = re.compile(r"^ECOv(?P<version>\d{3})_(?P<product>L1B_GEO|L2T_LSTE)_"
                  r"(?P<orbit>\d{5})_(?P<scene>\d{3})_"
                  r"(?:(?P<tile>\d{2}[C-X][A-Z]{2})_)?(?P<time>\d{8}T\d{6})_"
                  r"(?P<build>\d{4})_(?P<counter>\d{2})$")
OBSTRUCTION_LINE = re.compile(r'^ORB=(?P<orbit>\d+)\s+SCN=(?P<scene>\d+)\s+'
                             r't1=(?P<start>\S+)\s+t2=(?P<end>\S+)\s+'
                             r'FOV_OBST=(?P<flag>YES|NO)(?:\s+GeolocationAccuracyQA="(?P<qa>[^"\r\n]*)")?\s*$')


class GeoMetadataError(ValueError):
    pass


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _utc(value, *, allow_naive=False):
    value = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if value.tzinfo is None:
        if not allow_naive:
            raise GeoMetadataError("Metadata timestamp has no timezone.")
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def parse_identity(name):
    """Parse a granule UR or HDF filename; do not accept arbitrary URL contents."""
    name = str(name)
    if name.endswith(".h5.dmrpp"):
        name = name[:-9]
    elif name.endswith(".h5"):
        name = name[:-3]
    match = NAME.fullmatch(name)
    if not match or match["version"] != "002":
        raise GeoMetadataError("Expected an ECOSTRESS V002 GEO or tiled LSTE granule name.")
    parts = match.groupdict()
    if bool(parts["tile"]) != (parts["product"] == "L2T_LSTE"):
        raise GeoMetadataError("Granule tile/product naming mismatch.")
    stamp = datetime.strptime(parts["time"], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    return {**parts, "granule_name": name, "acquisition_utc": stamp.isoformat()}


def discovery_params(target_granule_name):
    target = parse_identity(target_granule_name)
    return {"collection_concept_id": GEO_COLLECTION, "short_name": "ECO_L1B_GEO", "version": "002",
            "readable_granule_name": f"ECOv002_L1B_GEO_{target['orbit']}_{target['scene']}_{target['time']}_*",
            "options[readable_granule_name][pattern]": "true", "page_size": 100}


def _unknown(reason, **extra):
    return {"status": "Unknown", "accepted": False, "reason": reason, **extra}


def select_geo_candidate(cmr_umm_items, target_granule_name):
    """Match orbit, scene and named acquisition; select revision before reading QA.

    Highest processing build/counter wins. Within that granule concept, the
    highest CMR revision-id wins. An unusable latest revision never falls back
    to an older candidate because it has a more convenient quality flag.
    """
    target = parse_identity(target_granule_name)
    candidates = []
    for item in cmr_umm_items:
        try:
            identity = parse_identity(item["umm"]["GranuleUR"])
        except (KeyError, TypeError, ValueError):
            continue
        if identity["product"] == "L1B_GEO" and all(identity[key] == target[key] for key in ("orbit", "scene", "time")):
            candidates.append((identity, item))
    if not candidates:
        return _unknown("No matching GEO orbit/scene/acquisition metadata.")
    latest_processing = max((int(identity["build"]), int(identity["counter"])) for identity, _ in candidates)
    candidates = [(identity, item) for identity, item in candidates if (int(identity["build"]), int(identity["counter"])) == latest_processing]
    try:
        concepts = {item["meta"]["concept-id"] for _, item in candidates}
        if len(concepts) != 1:
            return _unknown("Ambiguous GEO concepts at the latest processing revision.")
        revision = max(int(item["meta"]["revision-id"]) for _, item in candidates)
        newest = [(identity, item) for identity, item in candidates if int(item["meta"]["revision-id"]) == revision]
        fingerprints = {_sha(json.dumps(item, sort_keys=True).encode()) for _, item in newest}
        if len(fingerprints) != 1:
            return _unknown("Conflicting metadata for the latest GEO revision.")
        identity, item = newest[0]
        meta, umm = item["meta"], item["umm"]
        collection = umm["CollectionReference"]
        if collection["ShortName"] != "ECO_L1B_GEO" or collection["Version"] != "002" or meta.get("collection-concept-id") != GEO_COLLECTION:
            return _unknown("GEO collection identity mismatch.")
        if meta.get("deleted") or not re.fullmatch(r"G\d+-LPCLOUD", meta["concept-id"]) or revision < 1:
            return _unknown("GEO concept/revision is invalid or deleted.")
        actual = _utc(umm["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"])
        if abs((actual - _utc(identity["acquisition_utc"])).total_seconds()) >= 2:
            return _unknown("GEO CMR acquisition time differs from its granule name.")
        name = identity["granule_name"]
        path = GEO_PREFIX + name + "/" + name + ".h5"
        links = []
        for record in umm.get("RelatedUrls", []):
            url = record.get("URL", "")
            parsed = urlsplit(url)
            if (record.get("Type") == "GET DATA" and parsed.scheme == "https" and parsed.hostname == GEO_HOST
                    and parsed.port in (None, 443) and not parsed.username and not parsed.password
                    and not parsed.query and not parsed.fragment and parsed.path == path):
                links.append(url)
        if len(set(links)) != 1:
            return _unknown("Latest GEO revision lacks one exact unsigned HDF metadata link.")
        return {"status": "Matched", "accepted": False, "qa_not_read": True,
                "identity": identity, "granule_concept_id": meta["concept-id"], "cmr_revision_id": revision,
                "cmr_revision_date": meta.get("revision-date"), "cmr_start_utc": actual.isoformat(),
                "selection_rule": "highest processing build/counter, then highest CMR revision of that concept",
                "metadata_sha256": next(iter(fingerprints)), "h5_url": links[0], "dmrpp_url": links[0] + ".dmrpp"}
    except (KeyError, TypeError, ValueError):
        return _unknown("Latest GEO revision has incomplete or malformed identity metadata.")


def parse_geolocation_dmrpp(payload: bytes, expected_geo_name: str):
    """Read only the exact L1GEOMetadata/GeolocationAccuracyQA String value.

    Supports the base64 compact/missingdata scalar representation used by the
    official example. Conflicting values, opaque chunks and malformed XML are
    Unknown. Dataset identity is mandatory and bound to the requested GEO UR.
    """
    expected = parse_identity(expected_geo_name)
    if expected["product"] != "L1B_GEO":
        raise GeoMetadataError("DMR++ must be bound to a GEO granule.")
    audit = {"source_sha256": _sha(payload), "source_bytes": len(payload), "expected_geo_name": expected["granule_name"],
             "qa_label": "Unknown", "official_example": OFFICIAL_EXAMPLE}
    if len(payload) > MAX_DMRPP_BYTES:
        return _unknown("DMR++ metadata exceeds the bounded parser size.", **audit)
    if re.search(br"<!\s*(?:DOCTYPE|ENTITY)\b", payload, re.I):
        return _unknown("DTD/entity declarations are not accepted.", **audit)
    try:
        root = ET.fromstring(payload)
    except (ET.ParseError, ValueError):
        return _unknown("Malformed DMR++ XML.", **audit)
    local_tag = lambda tag: str(tag).rsplit("}", 1)[-1]
    if local_tag(root.tag) != "Dataset":
        return _unknown("DMR++ root is not a Dataset.", **audit)
    try:
        dataset = parse_identity(Path(root.attrib.get("name", "")).name)
    except GeoMetadataError:
        return _unknown("DMR++ Dataset lacks a verifiable GEO filename.", **audit)
    if dataset["granule_name"] != expected["granule_name"]:
        return _unknown("DMR++ Dataset identity differs from the selected GEO revision.", **audit)
    strings = []
    for group in root.iter():
        if local_tag(group.tag) == "Group" and group.attrib.get("name") == "L1GEOMetadata":
            strings.extend(element for element in list(group)
                           if local_tag(element.tag) == "String" and element.attrib.get("name") == "GeolocationAccuracyQA")
    if len(strings) != 1:
        return _unknown("Expected exactly one geolocation QA scalar in L1GEOMetadata.", **audit)
    scalar = strings[0]
    if any(local_tag(element.tag) == "Dim" for element in scalar.iter()):
        return _unknown("Geolocation QA is not a scalar String.", **audit)
    decoded = []
    for element in scalar.iter():
        text = (element.text or "").strip()
        if not text:
            continue
        if len(text) > 1024:
            return _unknown("Geolocation QA scalar is unexpectedly large.", **audit)
        try:
            value = base64.b64decode("".join(text.split()), validate=True).decode("utf-8").strip("\x00 \t\r\n")
        except (ValueError, UnicodeDecodeError, binascii.Error):
            return _unknown("Geolocation QA is not valid base64 UTF-8.", **audit)
        decoded.append(value)
    if len(decoded) != 1:
        return _unknown("Geolocation QA has missing or ambiguous inline content.", **audit)
    normalized = decoded[0].casefold()
    labels = {"best": "Best", "good": "Good", "suspect": "Suspect", "poor": "Poor"}
    if normalized not in labels:
        return _unknown("Geolocation QA is unrecognized or missing.", **audit)
    label = labels[normalized]
    return {**audit, "qa_label": label, "accepted": label in ("Best", "Good"),
            "status": "Accepted" if label in ("Best", "Good") else "Rejected",
            "reason": "Only positive Best/Good GEO flags satisfy the geolocation screen."}


def parse_obstruction_list(payload: bytes, *, complete=False, source_url=OBSTRUCTION_URL):
    """Parse the official ORB/SCN/t1/t2 list. A prefix is not a complete list.

    The list's embedded geolocation label is retained for audit only; it cannot
    replace a matched latest GEO product's GeolocationAccuracyQA.
    """
    audit = {"source_url": source_url, "source_sha256": _sha(payload), "source_bytes": len(payload),
             "complete_download": bool(complete), "records": [], "errors": [], "status": "Unknown"}
    if source_url != OBSTRUCTION_URL or len(payload) > MAX_OBSTRUCTION_BYTES:
        audit["errors"].append("Unexpected source or oversized obstruction list.")
        return audit
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        audit["errors"].append("Obstruction list is not UTF-8 text.")
        return audit
    records = {}
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = OBSTRUCTION_LINE.fullmatch(line.strip())
        if not match:
            audit["errors"].append(f"Unrecognized obstruction record on line {number}.")
            continue
        fields = match.groupdict()
        try:
            start, end = _utc(fields["start"], allow_naive=True), _utc(fields["end"], allow_naive=True)
            orbit, scene = int(fields["orbit"]), int(fields["scene"])
            # The official historical list includes long merged intervals,
            # optional QA text, and reprocessing records with subsecond shifts.
            # Preserve its reported intervals instead of imposing a swath length.
            if not (0 < orbit < 100000 and 0 < scene < 1000 and end >= start):
                raise ValueError("Invalid obstruction acquisition interval.")
            record = {"orbit": orbit, "scene": scene, "start_utc": start.isoformat(), "end_utc": end.isoformat(),
                      "obstructed": fields["flag"] == "YES", "listed_geolocation_qa": fields["qa"] or "Unknown"}
            # Keep distinct timing/flag variants; exact duplicates alone collapse.
            # Opposite flags are resolved as Unknown for that identity at lookup.
            key = (orbit, scene, start, end, record["obstructed"], record["listed_geolocation_qa"])
            records[key] = record
        except ValueError:
            audit["errors"].append(f"Malformed obstruction timing on line {number}.")
    audit["records"] = list(records.values())
    if complete and records and not audit["errors"]:
        audit["status"] = "Parsed"
    return audit


def obstruction_status(index, target_granule_name):
    target = parse_identity(target_granule_name)
    source = {key: index.get(key) for key in ("source_url", "source_sha256", "source_bytes")}
    if index.get("status") != "Parsed" or not index.get("complete_download"):
        return _unknown("Official obstruction list is missing, partial or unparsed.", **source)
    matches = [record for record in index["records"] if record["orbit"] == int(target["orbit"]) and record["scene"] == int(target["scene"])]
    if not matches:
        return {"status": "NotListed", "obstructed": None, "proves_unobstructed": False,
                "reason": "No matching entry in this complete official list; absence is not proof of unobstructed pixels.", **source}
    matching_time = [record for record in matches
                     if abs((_utc(record["start_utc"]) - _utc(target["acquisition_utc"])).total_seconds()) < 2]
    if not matching_time:
        return _unknown("Obstruction orbit/scene exists but its acquisition timestamp differs.", **source)
    if len({record["obstructed"] for record in matching_time}) != 1:
        return _unknown("Conflicting obstruction flags for the same acquisition.", **source)
    record = matching_time[0]
    return {"status": "Listed" if record["obstructed"] else "ListedNoObstruction", "obstructed": record["obstructed"],
            "proves_unobstructed": False, "record": record, "matching_variants": matching_time, **source}


def _anonymous_bytes(url, *, params=None, max_bytes):
    """Two exact public endpoints only; no auth/netrc, cookies or redirects."""
    if url not in (CMR_URL, OBSTRUCTION_URL):
        raise GeoMetadataError("Anonymous fetch endpoint is not approved public metadata.")
    try:
        with requests.Session() as session:
            session.trust_env = False
            with session.get(url, params=params, headers={"Accept-Encoding": "identity"}, stream=True,
                             allow_redirects=False, timeout=(10, 40)) as response:
                if response.status_code != 200:
                    raise GeoMetadataError(f"Public metadata returned HTTP {response.status_code}.")
                expected = response.headers.get("Content-Length")
                if expected is not None and int(expected) > max_bytes:
                    raise GeoMetadataError("Public metadata exceeds the byte budget.")
                chunks, size = [], 0
                for chunk in response.iter_content(65536):
                    size += len(chunk)
                    if size > max_bytes:
                        raise GeoMetadataError("Public metadata exceeds the streaming byte budget.")
                    chunks.append(chunk)
                payload = b"".join(chunks)
                if expected is not None and len(payload) != int(expected):
                    raise GeoMetadataError("Public metadata was truncated.")
                return payload, dict(response.headers)
    except requests.RequestException:
        raise GeoMetadataError("Anonymous public metadata request failed.") from None


def discover_geo(target_granule_name, cache):
    """One bounded public CMR snapshot; cache immutable response bytes and SHA."""
    params = discovery_params(target_granule_name)
    key = _sha(json.dumps(params, sort_keys=True).encode())[:24]
    directory = Path(cache)
    directory.mkdir(parents=True, exist_ok=True)
    path, sidecar = directory / ("geo_" + key + ".json"), directory / ("geo_" + key + ".source.json")
    if path.exists() and sidecar.exists():
        payload, audit = path.read_bytes(), json.loads(sidecar.read_text())
        if _sha(payload) != audit["source_sha256"]:
            raise GeoMetadataError("Cached CMR response hash changed.")
    else:
        payload, headers = _anonymous_bytes(CMR_URL, params=params, max_bytes=2 * 1024**2)
        audit = {"source_url": CMR_URL, "request": params, "source_sha256": _sha(payload), "source_bytes": len(payload),
                 "retrieved_utc": datetime.now(timezone.utc).isoformat(), "cmr_hits": headers.get("CMR-Hits", headers.get("cmr-hits")),
                 "revision_scope": "Latest within this frozen CMR snapshot; use a new cache directory to refresh."}
        path.write_bytes(payload)
        sidecar.write_text(json.dumps(audit, indent=2) + "\n")
    body = json.loads(payload)
    items = body.get("items", [])
    count = body.get("hits", audit.get("cmr_hits"))
    if count is None or int(count) > len(items):
        return _unknown("CMR snapshot is incomplete; latest GEO revision is not established.", discovery=audit)
    return {**select_geo_candidate(items, target_granule_name), "discovery": audit}


def cache_obstruction_list(payload, cache, *, complete=False):
    index = parse_obstruction_list(payload, complete=complete)
    directory = Path(cache)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("obstruction_" + index["source_sha256"] + ".txt")
    if path.exists() and _sha(path.read_bytes()) != index["source_sha256"]:
        raise GeoMetadataError("Cached obstruction source hash changed.")
    if not path.exists():
        path.write_bytes(payload)
    index["cache_file"] = path.name
    return index


def fetch_obstruction_list(cache):
    """Bounded anonymous public list download; raw SHA-named source is retained."""
    payload, _ = _anonymous_bytes(OBSTRUCTION_URL, max_bytes=MAX_OBSTRUCTION_BYTES)
    index = cache_obstruction_list(payload, cache, complete=True)
    index["retrieved_utc"] = datetime.now(timezone.utc).isoformat()
    return index
