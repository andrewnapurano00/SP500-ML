"""Private batch candidate, not the frozen M03 model or a hosted service.

Scoring requires an entire model-eligible monthly cohort for cross-sectional ranks.
Only balance-sheet sector scores are supported; raw scaling is research-only here.
"""
import json
import hashlib
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .constants import CONTEXT_FEATURES,RELATIVE_FEATURES,SECTOR_FEATURES
from .dataset import GROUPS,digest,validate_prepared
from .modeling import fundamental_design,fit_model

SPEC={'representation':'score','family':'balance_sheet'}

def technical_columns():
    return [f for f in GROUPS['technical_features'] if not f.startswith('reversal_')]

def transform(frame,state):
    technical=technical_columns()
    ranks=frame.groupby('signal_date')[technical+RELATIVE_FEATURES].rank(pct=True)-.5
    context=((frame[CONTEXT_FEATURES]-pd.Series(state['center']))/pd.Series(state['scale'])).clip(-5,5)
    fundamental,_,names,_=fundamental_design(frame,frame,'score','balance_sheet')
    matrix=np.column_stack([ranks[technical],context,ranks[RELATIVE_FEATURES],fundamental])
    names=technical+SECTOR_FEATURES+names
    if not np.isfinite(matrix).all():raise ValueError('Nonfinite transformed inputs')
    return matrix,names

def cohort_digest(frame):
    identities=sorted(zip(frame.security_id.astype(str),frame.ticker.astype(str)))
    return hashlib.sha256(json.dumps(identities,separators=(',',':')).encode()).hexdigest()

def train(dataset_dir,signal_month,bundle_dir):
    manifest=validate_prepared(dataset_dir)
    data=pd.read_parquet(Path(dataset_dir)/'research_with_fundamentals.parquet')
    month=pd.Period(signal_month,freq='M')
    candidates=pd.read_parquet(Path(dataset_dir)/'candidates.parquet')
    cohort=candidates.loc[candidates.signal_date.dt.to_period('M').eq(month)]
    if cohort.empty:raise ValueError('Explicit eligible scoring cohort required in prepared snapshot')
    train_months=pd.period_range(month-37,month-2,freq='M')
    train=data.loc[data.signal_date.dt.to_period('M').isin(train_months)].sort_values(['signal_date','ticker'])
    if set(train.signal_date.dt.to_period('M'))!=set(train_months):raise ValueError('36 contiguous mature training months required, followed by one purge month')
    if not train[['next_month_return','benchmark_return','target_top20']].notna().all().all():raise ValueError('Only known outcomes may train')
    history=train[['signal_date','sector']+CONTEXT_FEATURES].drop_duplicates(['signal_date','sector'])
    state={'center':history[CONTEXT_FEATURES].mean().to_dict(),'scale':history[CONTEXT_FEATURES].std(ddof=0).replace(0,1.).to_dict()}
    matrix,names=transform(train,state)
    model=fit_model(matrix,train.target_top20.astype(int))
    out=Path(bundle_dir)
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new empty private bundle directory')
    out.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,out/'estimator.joblib')
    import importlib.metadata
    metadata={'status':'LOCAL_RESEARCH_CANDIDATE','signal_month':str(month),'cohort_sha256':cohort_digest(cohort),'train_start':str(train_months[0]),'train_end':str(train_months[-1]),'training_months':36,'purge_months':1,'source':manifest['source'],'features':names,'state':state,'model_sha256':digest(out/'estimator.joblib'),'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','joblib']},'source_sha256':{p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('modeling.py'),Path(__file__).with_name('constants.py')]}}
    # Test persistence before making a candidate available.
    if not np.array_equal(model.predict_proba(matrix),joblib.load(out/'estimator.joblib').predict_proba(matrix)):raise ValueError('Reload probability mismatch')
    (out/'manifest.json').write_text(json.dumps(metadata,indent=2,allow_nan=False),encoding='utf8')
    return metadata

def score(bundle_dir,frame,expected_manifest_sha256):
    out=Path(bundle_dir)
    if digest(out/'manifest.json')!=expected_manifest_sha256:raise ValueError('Trusted manifest digest required')
    metadata=json.loads((out/'manifest.json').read_text(encoding='utf8'))
    if digest(out/'estimator.joblib')!=metadata['model_sha256']:raise ValueError('Estimator digest mismatch')
    import importlib.metadata
    for package,version in metadata['packages'].items():
        if importlib.metadata.version(package)!=version:raise ValueError('Candidate dependency mismatch: '+package)
    for name,sha in metadata['source_sha256'].items():
        if digest(Path(__file__).with_name(name))!=sha:raise ValueError('Candidate source mismatch: '+name)
    if frame.empty or frame[['security_id','ticker','signal_date','sector']].isna().any().any():raise ValueError('Complete identity required')
    if frame.duplicated('security_id').any() or frame.duplicated('ticker').any():raise ValueError('Duplicate scoring identity')
    dates=pd.to_datetime(frame.signal_date)
    if dates.nunique()!=1 or set(dates.dt.to_period('M').astype(str))!={metadata['signal_month']}:raise ValueError('Exact candidate signal month required')
    if cohort_digest(frame)!=metadata['cohort_sha256']:raise ValueError('Scoring cohort is incomplete or differs from pinned snapshot')
    matrix,names=transform(frame,metadata['state'])
    if names!=metadata['features']:raise ValueError('Estimator feature order mismatch')
    model=joblib.load(out/'estimator.joblib')  # trusted local artifacts only
    result=frame[['security_id','ticker','signal_date','sector']].copy()
    result['probability']=model.predict_proba(matrix)[:,1]
    return result.sort_values(['probability','ticker'],ascending=[False,True]).reset_index(drop=True)
