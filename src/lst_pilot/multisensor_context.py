"""Verify hash-bound causal native context before residual-model use.

This only reads completed join/native tables, never satellite assets or labels
from the target file. Invalid claimed matches fail; absent context stays absent.
"""
from __future__ import annotations
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import from_wkb
from shapely.geometry import Point, box
from shapely.ops import unary_union

from . import option_b_train as old

TARGET_COLUMNS = ["sample_id", "region_id", "datetime_utc", "latitude", "longitude", "epsg"]
MODEL_CONTEXT = ["coarse_lst_c", "coarse_age_hours", "coarse_context_eligible"]


def missing_context(frame):
    result = frame.copy()
    result["coarse_lst_c"] = np.nan; result["coarse_age_hours"] = np.nan
    result["coarse_context_eligible"] = False
    return result


def check_intervals(joined, targets):
    flags = old.strict_bool(joined.coarse_context_eligible, "coarse_context_eligible")
    missing=joined.loc[~flags]
    if missing[["source_granule_start","source_granule_end"]].notna().any().any():
        raise ValueError("Missing context must not contain hidden source timestamps before thermal columns are read.")
    data = joined.loc[flags]
    if data.empty: return flags
    start = pd.to_datetime(data.source_granule_start, utc=True, errors="raise")
    end = pd.to_datetime(data.source_granule_end, utc=True, errors="raise")
    target = pd.to_datetime(targets.set_index("sample_id").loc[data.sample_id, "datetime_utc"], utc=True).reset_index(drop=True)
    start = start.reset_index(drop=True); end = end.reset_index(drop=True)
    if (start.isna().any() or end.isna().any() or (start > end).any() or (end > target).any()
            or (start < target-pd.Timedelta(hours=24)).any()):
        raise ValueError("Native context interval is future, stale, missing or inverted; context values were not accepted.")
    return flags


@lru_cache(maxsize=32)
def _reserved_union(area_json):
    area=json.loads(area_json)
    if area["id"] not in ("greater_london", "sioux_falls"): return None
    left, bottom, right, top = area["extent_m"]
    polygons = [box(left+c*10000,bottom+r*10000,min(right,left+(c+1)*10000),min(top,bottom+(r+1)*10000)).buffer(1000)
                for r in range(int(np.ceil((top-bottom)/10000))) for c in range(int(np.ceil((right-left)/10000))) if (r+2*c)%5 == 0]
    return unary_union(polygons)


def reserved_union(area):
    return _reserved_union(json.dumps(area,sort_keys=True))


@lru_cache(maxsize=32)
def transformer(epsg):
    return Transformer.from_crs(4326,int(epsg),always_xy=True)


def verify_native_row(row, joined, target, area, for_fitting):
    for flag in ("native_qa_valid", "context_eligible") + (("context_fit_eligible",) if for_fitting else ()):
        if not old.strict_bool(pd.Series([row[flag]]), flag).iloc[0]:
            raise ValueError("A matched native cell failed QA or geographic eligibility.")
    for field,shift in (("qc_mandatory",0),("qc_data",2),("qc_cloud",4),("qc_lst_error",14)):
        if int(row[field]) != ((int(row["qc_raw"])>>shift)&3): raise ValueError("Decoded QA disagrees with raw native quality bits.")
    if not 150 <= float(row["lst_c"])+273.15 <= 400: raise ValueError("Native LST is outside the source QA range.")
    product = row["product"]
    if product not in ("MOD21", "VNP21") or str(row["version"]).zfill(3) != {"MOD21":"061","VNP21":"002"}[product]:
        raise ValueError("Native product/version is outside the fixed H specification.")
    if (any(int(row[k]) != 0 for k in ("qc_mandatory", "qc_data", "qc_cloud", "oceanpix_raw"))
            or int(row["qc_lst_error"]) not in (2,3) or not 0 < float(row["lst_error_k"]) <= 1.5
            or not 0 <= float(row["view_zenith_deg"]) <= 30):
        raise ValueError("Claimed native context does not satisfy the fixed raw QA gates.")
    pairs = {"coarse_native_id":"native_cell_id", "coarse_acquisition_id":"acquisition_id",
             "coarse_product":"product", "coarse_source_sha256":"source_sha256"}
    if any(joined[a] != row[b] for a,b in pairs.items()) or row["region_id"] != target["region_id"]:
        raise ValueError("Joined context identity differs from its audited native row.")
    for a,b in (("coarse_lst_c","lst_c"),("coarse_lst_error_k","lst_error_k"),
                ("coarse_native_area_m2","native_footprint_area_m2"),("coarse_support_margin_m","context_support_margin_m")):
        if not np.isfinite(float(joined[a])) or not np.isclose(float(joined[a]),float(row[b]),rtol=0,atol=1e-8):
            raise ValueError("Joined context values differ from their native source.")
    for a,b in (("source_granule_start","granule_start_utc"),("source_granule_end","granule_end_utc")):
        if pd.Timestamp(joined[a]) != pd.Timestamp(row[b]):
            raise ValueError("Joined source timestamps differ from the native acquisition.")
    oldest=(pd.Timestamp(target["datetime_utc"])-pd.Timestamp(joined["source_granule_start"])).total_seconds()/3600
    newest=(pd.Timestamp(target["datetime_utc"])-pd.Timestamp(joined["source_granule_end"])).total_seconds()/3600
    if not np.allclose([joined["coarse_age_hours"],joined["coarse_age_min_hours"]],[oldest,newest],rtol=0,atol=1e-9):
        raise ValueError("Reported context ages disagree with exact acquisition timestamps.")
    if int(row["footprint_epsg"]) != int(area["epsg"]) or int(target["epsg"]) != int(area["epsg"]):
        raise ValueError("Context/sample footprint CRS does not match its pilot.")
    polygon, support = from_wkb(row["native_footprint_wkb"]), from_wkb(row["context_support_wkb"])
    margin = float(row["context_support_margin_m"])
    if not np.isclose(polygon.area,float(row["native_footprint_area_m2"]),rtol=0,atol=1e-6): raise ValueError("Native footprint area differs from its geometry.")
    edge=np.linalg.norm(np.diff(np.asarray(polygon.exterior.coords),axis=0),axis=1).max()
    minimum=.5*edge+(150 if product=="MOD21" else 375)/np.cos(np.radians(row["view_zenith_deg"]))**2
    if not np.isfinite(margin) or margin < minimum-1e-6 or not polygon.is_valid or not support.is_valid or polygon.is_empty:
        raise ValueError("Invalid native context geometry or support margin.")
    if (not polygon.equals_exact(from_wkb(joined["coarse_footprint_wkb"]),1e-8) or
            not support.equals_exact(from_wkb(joined["coarse_support_wkb"]),1e-8) or
            not support.buffer(1e-6).covers(polygon.buffer(margin)) or not box(*area["extent_m"]).covers(support)):
        raise ValueError("Native whole-support geometry is inconsistent or leaves the pilot.")
    x,y=transformer(area["epsg"]).transform(target["longitude"],target["latitude"])
    if not polygon.covers(Point(x,y)):
        raise ValueError("Sample does not lie within the claimed native footprint.")
    exclusion=reserved_union(area) if for_fitting else None
    if for_fitting and (area["id"] == "cabauw" or (exclusion is not None and support.intersects(exclusion))):
        raise ValueError("Expanded native support crosses a withheld block or buffer.")


def attach_one(frame, specification, areas, *, for_fitting):
    join_dir=Path(specification["join_dir"]); target_path=Path(specification["target_path"])
    context_path=Path(specification["context_manifest_path"])
    manifest=json.loads((join_dir/"manifest.json").read_text()); table=join_dir/"coarse_context.parquet"
    if (manifest["input_sha256"] != old.sha(target_path) or manifest["context_manifest_sha256"] != old.sha(context_path)
            or manifest["output_sha256"] != old.sha(table) or manifest.get("for_fitting") is not for_fitting
            or manifest.get("maximum_oldest_possible_age_hours") != 24 or manifest.get("input_columns_read") != TARGET_COLUMNS):
        raise ValueError("Native context join is not bound to the correct target/source/scope.")
    if manifest.get("join_source_sha256") != old.sha(Path(__file__).with_name("coarse_join.py")):
        raise ValueError("Native context join implementation changed.")
    targets=pd.read_parquet(target_path,columns=TARGET_COLUMNS)
    # Full table is decoded below: guard unused QA rows as well as model IDs.
    allowed=[2021,2022] if for_fitting else [2021,2022,2023]
    stamps=pd.to_datetime(targets.datetime_utc,utc=True,errors="raise")
    if stamps.isna().any() or not stamps.dt.year.isin(allowed).all():
        raise ValueError("H context is not admitted in this target year; context values were not read.")
    all_targets=targets.copy()
    if targets.sample_id.duplicated().any() or not set(frame.sample_id).issubset(set(targets.sample_id)):
        raise ValueError("Native context target identities differ from the exact fitting/evaluation cohort.")
    targets=targets.set_index("sample_id").loc[frame.sample_id].reset_index()
    for field in ("region_id","latitude","longitude"):
        if not np.array_equal(targets[field].to_numpy(),frame[field].to_numpy()):
            raise ValueError("Native context target location differs from the model observation.")
    if not np.array_equal(pd.to_datetime(targets.datetime_utc,utc=True).to_numpy(),pd.to_datetime(frame.datetime_utc,utc=True).to_numpy()):
        raise ValueError("Native context target timestamps differ from the model observation.")
    metadata=pd.read_parquet(table,columns=["sample_id","coarse_context_eligible","source_granule_start","source_granule_end"])
    if metadata.sample_id.duplicated().any() or set(metadata.sample_id) != set(all_targets.sample_id):
        raise ValueError("Context join lost, duplicated or added sample identities.")
    if len(metadata)!=manifest["rows"] or int(old.strict_bool(metadata.coarse_context_eligible,"coarse_context_eligible").sum())!=manifest["matched_rows"]:
        raise ValueError("Context join coverage differs from its complete audit.")
    check_intervals(metadata,all_targets)
    joined=pd.read_parquet(table).set_index("sample_id").loc[frame.sample_id].reset_index()
    flags=old.strict_bool(joined.coarse_context_eligible,"coarse_context_eligible")
    for field in ("coarse_lst_c","coarse_age_hours"):
        if joined.loc[~flags,field].notna().any():
            raise ValueError("Missing context must not contain imputed native values.")
    context=json.loads(context_path.read_text())
    lookup={r["table_sha256"]:r for r in context["records"] if "table_path" in r}
    selected=joined.loc[flags]
    for table_sha, part in selected.groupby("coarse_context_table_sha256"):
        if table_sha not in lookup: raise ValueError("Matched native table is absent from the frozen context manifest.")
        record=lookup[table_sha]; path=Path(record["table_path"])
        if old.sha(path) != table_sha: raise ValueError("Audited native context table changed.")
        if record.get("native_table_path") and old.sha(record["native_table_path"]) != record["native_table_sha256"]:
            raise ValueError("Original native table changed after its audited context was built.")
        native_meta=pd.read_parquet(path,columns=["native_cell_id","granule_start_utc","granule_end_utc"])
        if native_meta.native_cell_id.duplicated().any(): raise ValueError("Native context table duplicates a cell identity.")
        # Native tables are single granules in real inputs; reject malformed
        # mixed-year extras before decoding any unselected thermal rows too.
        native_start=pd.to_datetime(native_meta.granule_start_utc,utc=True,errors="raise")
        native_end=pd.to_datetime(native_meta.granule_end_utc,utc=True,errors="raise")
        if (native_start.isna().any() or native_end.isna().any()
                or not native_start.dt.year.isin(allowed).all() or not native_end.dt.year.isin(allowed).all()
                or (native_start>native_end).any()):
            raise ValueError("Native context table has future, stale or reserved-year metadata; thermal values were not read.")
        selected_meta=native_meta.set_index("native_cell_id").loc[part.coarse_native_id]
        # Inspect native timestamps independently before decoding native LST values.
        expected_meta=part[["sample_id","coarse_context_eligible","source_granule_start","source_granule_end"]].copy()
        expected_meta["source_granule_start"]=selected_meta.granule_start_utc.to_numpy()
        expected_meta["source_granule_end"]=selected_meta.granule_end_utc.to_numpy()
        check_intervals(expected_meta,targets)
        native=pd.read_parquet(path)
        if record.get("native_summary_path"):
            summary=json.loads(Path(record["native_summary_path"]).read_text())
            if not native.source_sha256.eq(summary["source_assets"]["thermal"]["sha256"]).all():
                raise ValueError("Native source identity disagrees with its source-asset audit.")
            try: rms=float(summary["metadata"]["geolocation_rms_error_m"])
            except (KeyError,TypeError,ValueError): rms=None
            for source in native.loc[native.native_cell_id.isin(part.coarse_native_id)].itertuples():
                if source.product!="MOD21": continue
                if rms is None or not np.isfinite(rms) or rms<0: raise ValueError("Missing MOD03 geolocation uncertainty evidence.")
                polygon=from_wkb(source.native_footprint_wkb)
                edge=np.linalg.norm(np.diff(np.asarray(polygon.exterior.coords),axis=0),axis=1).max()
                expected=.5*edge+max(150,3*rms)/np.cos(np.radians(source.view_zenith_deg))**2
                if not np.isclose(expected,source.context_support_margin_m,rtol=0,atol=1e-6):
                    raise ValueError("Native support margin differs from the recorded MOD03 uncertainty policy.")
        if native.native_cell_id.duplicated().any(): raise ValueError("Native context table duplicates a cell identity.")
        native=native.set_index("native_cell_id",drop=False)
        for index, row in part.iterrows():
            if row.coarse_native_id not in native.index: raise ValueError("Native cell identity is absent from its source.")
            source=native.loc[row.coarse_native_id]
            if not isinstance(row.coarse_source_sha256,str) or not re.fullmatch(r"[0-9a-f]{64}",row.coarse_source_sha256):
                raise ValueError("Missing native source hash.")
            verify_native_row(source,row,targets.iloc[index],areas[targets.iloc[index].region_id],for_fitting)
    result=frame.copy().reset_index(drop=True)
    for field in joined.columns:
        if field != "sample_id": result[field]=joined[field].to_numpy()
    audit={"join_manifest_sha256":old.sha(join_dir/"manifest.json"),"target_sha256":old.sha(target_path),
        "context_manifest_sha256":old.sha(context_path),"join_table_sha256":old.sha(table),
        "rows":len(result),"eligible_rows":int(flags.sum()),"for_fitting":for_fitting,
        "eligible_pilot_dates":len(result.loc[flags.to_numpy(),["region_id","datetime_utc"]].assign(
            utc_day=pd.to_datetime(result.loc[flags.to_numpy(),"datetime_utc"],utc=True).dt.floor("D"))[["region_id","utc_day"]].drop_duplicates()),
        "by_product":joined.loc[flags,"coarse_product"].value_counts().to_dict()}
    return result,audit


def batches(specification):
    if specification is None: return []
    return specification if isinstance(specification,list) else [specification]


def freeze_inputs(specification):
    """Hash files without decoding evaluation context values before model freeze."""
    hashes={}
    for section in ("fit","old_evaluation","new_evaluation"):
        for batch in batches(specification.get(section)):
            directory=Path(batch["join_dir"])
            paths=[directory/"manifest.json",directory/"coarse_context.parquet",Path(batch["target_path"]),Path(batch["context_manifest_path"])]
            context=json.loads(Path(batch["context_manifest_path"]).read_text())
            for record in context["records"]:
                if record.get("status") not in (None,"processed"):
                    raise ValueError("A join requires an explicit processed-only context manifest.")
                paths.append(Path(record["table_path"]))
                if record.get("native_table_path"): paths.append(Path(record["native_table_path"]))
                if record.get("native_summary_path"): paths.append(Path(record["native_summary_path"]))
            for original_record in context.get("source_manifests",[]):
                original_path=Path(original_record["path"])
                if old.sha(original_path)!=original_record["sha256"]: raise ValueError("Original mixed-status native audit changed.")
                paths.append(original_path)
            if context.get("original_manifest_path"):
                original=Path(context["original_manifest_path"])
                if old.sha(original)!=context["original_manifest_sha256"]: raise ValueError("Original mixed-status native audit changed.")
                paths.append(original)
            for path in paths: hashes[str(path.resolve())]=old.sha(path)
    return hashes


def verify_inputs(hashes):
    for path,digest in hashes.items():
        if old.sha(path)!=digest: raise ValueError("Frozen H context input changed before evaluation context was read.")


def attach(frame,specification,areas,*,for_fitting):
    """Composite joins may contain extra QA rows; never drop a final model row."""
    frame=frame.reset_index(drop=True)
    if frame.sample_id.duplicated().any(): raise ValueError("Duplicate model identity for context joining.")
    result=missing_context(frame); records=[]; covered=set(); parts=[]
    for batch in batches(specification):
        metadata=pd.read_parquet(batch["target_path"],columns=["sample_id"])
        if metadata.sample_id.duplicated().any(): raise ValueError("Duplicate context target identity.")
        overlap=frame.sample_id.isin(metadata.sample_id)
        selected=frame.loc[overlap].reset_index(drop=True)
        if selected.empty: continue
        attached,audit=attach_one(selected,batch,areas,for_fitting=for_fitting)
        context_fields=[x for x in attached if x.startswith("coarse_") or x in ("source_granule_start","source_granule_end")]
        parts.append(attached[["sample_id",*context_fields]])
        covered.update(selected.sample_id);records.append(audit)
    if parts:
        combined=pd.concat(parts,ignore_index=True)
        # Duplicate batches are permitted only with identical missingness/source proof.
        for _,group in combined.loc[combined.sample_id.duplicated(False)].groupby("sample_id"):
            for _,row in group.iloc[1:].iterrows():
                pd.testing.assert_series_equal(row,group.iloc[0],check_names=False)
        combined=combined.drop_duplicates("sample_id").set_index("sample_id")
        for field in combined:
            if field not in result: result[field]=pd.Series(None,index=result.index,dtype=object)
            mask=result.sample_id.isin(combined.index)
            result.loc[mask,field]=combined.loc[result.loc[mask,"sample_id"],field].to_numpy()
    result["coarse_context_eligible"]=old.strict_bool(result.coarse_context_eligible,"coarse_context_eligible")
    return result,{"rows":len(frame),"join_covered_rows":len(covered),"rows_absent_from_joins":len(frame)-len(covered),
                   "eligible_rows":int(result.coarse_context_eligible.sum()),"batches":records,
                   "row_sha256":old.row_hash(frame),"no_model_rows_removed":True}
