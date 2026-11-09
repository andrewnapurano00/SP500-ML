"""Explicit, immutable Gold export adapter. No vendor or PostgreSQL reads."""
import hashlib
import json
import re
from importlib.resources import files
from pathlib import Path
import numpy as np
import pandas as pd

GROUPS=json.loads(files('sp500_ml').joinpath('feature_groups.json').read_text(encoding='utf8'))
FILENAMES={'research':'research_with_fundamentals.parquet','latest':'latest_with_fundamentals.parquet','population':'population.parquet','candidates':'candidates.parquet'}

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_frame(frame):
    required={'security_id','signal_date','ticker','sector','next_month_return','benchmark_return','target_top20'}
    required.update(sum((GROUPS[k] for k in ['technical_features','macro_features','sector_features','fundamental_raw_features','fundamental_score_features','fundamental_composites']),[]))
    required.update(f+'_applicable' for f in GROUPS['fundamental_factors'])
    required.update(['debt_free','nonpositive_equity','no_interest_expense','growth_unavailable_negative_base','growth_unavailable_small_base'])
    missing=required-set(frame)
    if missing:raise ValueError('Missing contract columns: '+', '.join(sorted(missing)))
    if frame[['security_id','signal_date','ticker','sector']].isna().any().any():raise ValueError('Null identity/date/sector')
    if frame.duplicated(['signal_date','security_id']).any() or frame.duplicated(['signal_date','ticker']).any():raise ValueError('Duplicate security or ticker within month')
    dates=pd.to_datetime(frame.signal_date,errors='raise')
    if dates.dt.tz is not None:raise ValueError('Timezone-naive exchange signal dates required')
    months=dates.dt.to_period('M')
    if pd.DataFrame({'month':months,'date':dates}).groupby('month').date.nunique().gt(1).any():raise ValueError('One signal date per month required')
    for column in required-{'security_id','signal_date','ticker','sector'}:
        values=pd.to_numeric(frame[column],errors='raise')
        if np.isinf(values.to_numpy(dtype=float)).any():raise ValueError('Infinite feature/outcome: '+column)
    flags=[f+'_applicable' for f in GROUPS['fundamental_factors']]+['debt_free','nonpositive_equity','no_interest_expense','growth_unavailable_negative_base','growth_unavailable_small_base']
    if not frame[flags].isin([0,1,False,True]).all().all():raise ValueError('Explicit binary non-null flags required')
    if not frame.target_top20.dropna().isin([0,1]).all():raise ValueError('Binary top-quintile labels required')
    if (frame[['next_month_return','benchmark_return']].dropna() < -1).any().any():raise ValueError('Simple return below -100%')
    if frame.groupby('signal_date').benchmark_return.nunique().gt(1).any():raise ValueError('Benchmark differs within month')
    return dates

def prepare(frame,output_dir,source):
    if source.get('provider')!='databricks_gold' or not re.fullmatch('[a-f0-9]{64}',source.get('source_binding_sha256','')):raise ValueError('Explicit immutable Gold provenance required')
    frame=frame.copy();frame['signal_date']=validate_frame(frame)
    frame=frame.sort_values(['signal_date','ticker']).reset_index(drop=True)
    readiness=GROUPS['technical_features']+GROUPS['macro_features']+GROUPS['sector_features']
    candidates=frame.loc[frame[readiness].notna().all(axis=1)].copy()
    latest=candidates.loc[candidates.signal_date.eq(frame.signal_date.max())].copy()
    if latest.empty or latest[['next_month_return','benchmark_return','target_top20']].notna().any().any():raise ValueError('Latest cohort must be nonempty and feature-only, with unknown outcomes')
    history=candidates.loc[candidates[['next_month_return','benchmark_return','target_top20']].notna().all(axis=1)].copy()
    if history.empty:raise ValueError('No mature research history')
    out=Path(output_dir)
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    frames={'research':history,'latest':latest,'population':frame,'candidates':candidates}
    for key,value in frames.items():value.to_parquet(out/FILENAMES[key],index=False)
    hashes={key:digest(out/name) for key,name in FILENAMES.items()}
    manifest={**GROUPS,'source':source,'file_sha256':hashes,'fundamental_dataset_sha256':hashes['research'],'fundamental_live_sha256':hashes['latest']}
    (out/'feature_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    return {key:len(value) for key,value in frames.items()}

def validate_prepared(directory):
    root=Path(directory);manifest=json.loads((root/'feature_manifest.json').read_text(encoding='utf8'))
    for key,name in FILENAMES.items():
        if digest(root/name)!=manifest['file_sha256'][key]:raise ValueError('Dataset hash mismatch: '+key)
    for key,value in GROUPS.items():
        if manifest.get(key)!=value:raise ValueError('Feature group contract mismatch: '+key)
    history=pd.read_parquet(root/FILENAMES['research'])
    validate_frame(history)
    months=sorted(history.signal_date.dt.to_period('M').unique())
    if list(pd.period_range(months[0],months[-1],freq='M'))!=months:raise ValueError('Contiguous monthly research history required')
    return manifest

def export_gold(binding_file,profile,warehouse_id,output_dir):
    from databricks.sdk import WorkspaceClient
    from sp500_lakehouse.reader import GoldReader,StatementQuery
    binding=json.loads(Path(binding_file).read_text(encoding='utf8'))
    reader=GoldReader(StatementQuery(WorkspaceClient(profile=profile),warehouse_id),binding)
    frame=reader.read(include_targets=True)
    return prepare(frame,output_dir,{'provider':'databricks_gold','source_binding_sha256':reader.sha256,'snapshot_id':binding['snapshot_id']})
