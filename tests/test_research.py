import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sp500_ml.dataset import GROUPS,prepare,validate_prepared,digest
from sp500_ml.constants import CONTEXT_FEATURES,RELATIVE_FEATURES
from sp500_ml.modeling import design,SPECS
from sp500_ml.evaluation import paired_test,performance
from sp500_ml.eda import adjust,block_indices
from sp500_ml.candidate import train,score,transform

@pytest.fixture
def panel():
    rng=np.random.default_rng(42)
    dates=pd.date_range('2020-01-31',periods=39,freq='ME')
    rows=[]
    columns=set(sum(GROUPS.values(),[]))
    for date in dates:
        for i in range(40):
            row={c:float(rng.normal()) for c in columns}
            row.update(security_id=str(i),ticker='S'+str(i).zfill(3),sector='Industrials' if i<20 else 'Energy',signal_date=date,next_month_return=float(rng.normal(0,.05)),benchmark_return=.01,target_top20=int(i<8))
            for c in GROUPS['fundamental_score_features']+GROUPS['fundamental_composites']:row[c]=float(rng.uniform(0,100))
            for c in GROUPS['fundamental_factors']:row[c+'_applicable']=1
            for c in ['debt_free','nonpositive_equity','no_interest_expense','growth_unavailable_negative_base','growth_unavailable_small_base']:row[c]=0
            rows.append(row)
    result=pd.DataFrame(rows)
    # Shared context really is at sector-month / market-month grain.
    for c in CONTEXT_FEATURES:result[c]=result.groupby(['signal_date','sector'])[c].transform('first')
    for c in GROUPS['macro_features']:result[c]=result.groupby('signal_date')[c].transform('first')
    result.loc[result.signal_date.eq(dates[-1]),['next_month_return','benchmark_return','target_top20']]=np.nan
    return result

@pytest.fixture
def prepared(tmp_path,panel):
    out=tmp_path/'data'
    prepare(panel,out,{'provider':'databricks_gold','source_binding_sha256':'a'*64})
    return out

def test_contract_keeps_missing_fundamentals(tmp_path,panel):
    panel.loc[0,'debt_to_assets_sector_score']=np.nan
    counts=prepare(panel,tmp_path/'data',{'provider':'databricks_gold','source_binding_sha256':'a'*64})
    assert counts['research']==38*40 and counts['latest']==40
    validate_prepared(tmp_path/'data')

def test_duplicate_identity_fails(tmp_path,panel):
    panel.loc[1,'security_id']=panel.loc[0,'security_id']
    with pytest.raises(ValueError,match='Duplicate'):prepare(panel,tmp_path/'data',{'provider':'databricks_gold','source_binding_sha256':'a'*64})

def test_dataset_tamper_fails(prepared):
    with (prepared/'research_with_fundamentals.parquet').open('ab') as f:f.write(b'x')
    with pytest.raises(ValueError,match='hash mismatch'):validate_prepared(prepared)

def test_latest_outcomes_rejected(tmp_path,panel):
    panel.loc[panel.signal_date.eq(panel.signal_date.max()),'next_month_return']=0
    with pytest.raises(ValueError,match='feature-only'):prepare(panel,tmp_path/'data',{'provider':'databricks_gold','source_binding_sha256':'a'*64})

@pytest.mark.parametrize('spec',list(SPECS.values()))
def test_design_outcome_invariance(panel,spec):
    train=panel.loc[panel.signal_date.lt('2023-01-01')].copy()
    test=panel.loc[panel.signal_date.eq(panel.signal_date.max())].copy()
    technical=[c for c in GROUPS['technical_features'] if not c.startswith('reversal_')]
    a,b,names,audit=design(train,test,technical,spec)
    poisoned=test.copy();poisoned['next_month_return']=999;poisoned['benchmark_return']=-999;poisoned['target_top20']=1
    aa,bb,nn,_=design(train,poisoned,technical,spec)
    np.testing.assert_array_equal(a,aa);np.testing.assert_array_equal(b,bb);assert names==nn
    # Test-context changes cannot alter training-fitted values/scales.
    changed=test.copy();changed[CONTEXT_FEATURES]+=100000
    aaa,_,_,new_audit=design(train,changed,technical,spec)
    np.testing.assert_array_equal(a,aaa);assert audit==new_audit

def test_candidate_matches_reference_and_persists(prepared,tmp_path):
    bundle=tmp_path/'bundle'
    metadata=train(prepared,'2023-03',bundle)
    assert metadata['train_start']=='2020-02' and metadata['train_end']=='2023-01'
    frame=pd.read_parquet(prepared/'latest_with_fundamentals.parquet')
    history=pd.read_parquet(prepared/'research_with_fundamentals.parquet')
    history=history.loc[history.signal_date.dt.to_period('M').between('2020-02','2023-01')]
    technical=[c for c in GROUPS['technical_features'] if not c.startswith('reversal_')]
    _,reference,names,_=design(history,frame,technical,{'representation':'score','family':'balance_sheet'})
    matrix,actual_names=transform(frame,metadata['state'])
    assert names==actual_names and len(names)==74
    np.testing.assert_allclose(matrix,reference,atol=1e-12)
    trusted=digest(bundle/'manifest.json')
    first=score(bundle,frame,trusted);second=score(bundle,frame,trusted)
    np.testing.assert_array_equal(first.probability,second.probability)
    poisoned=frame.copy();poisoned['target_top20']=1;poisoned['next_month_return']=999
    np.testing.assert_array_equal(first.probability,score(bundle,poisoned,trusted).probability)
    with pytest.raises(ValueError,match='cohort'):score(bundle,frame.iloc[:-1],trusted)
    with pytest.raises(ValueError,match='manifest'):score(bundle,frame,'b'*64)
    with (bundle/'estimator.joblib').open('ab') as f:f.write(b'x')
    with pytest.raises(ValueError,match='Estimator digest'):score(bundle,frame,trusted)

def test_gap_blocks_training(prepared,tmp_path):
    p=prepared/'research_with_fundamentals.parquet';d=pd.read_parquet(p)
    d=d.loc[~d.signal_date.dt.to_period('M').eq('2021-04')];d.to_parquet(p,index=False)
    manifest=json.loads((prepared/'feature_manifest.json').read_text())
    manifest['file_sha256']['research']=digest(p)
    (prepared/'feature_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='Contiguous'):train(prepared,'2023-03',tmp_path/'bundle')

def test_bootstrap_and_multiplicity():
    assert paired_test(np.zeros(68))['p']==1
    assert paired_test(np.zeros(68))['ci_low']==0
    adjusted=adjust([.001,.02,.9,np.nan])
    np.testing.assert_allclose(adjusted[:3],[.003,.03,.9]);assert np.isnan(adjusted[3])
    ix=block_indices(10,block=3,draws=20)
    assert ix.shape==(20,10) and ((ix[:,1]-ix[:,0])%10==1).all()

def test_drawdown_includes_initial_capital():
    report=performance(pd.DataFrame({'loss':[-.2,.1]}))
    assert report.loc['loss','max_drawdown']==pytest.approx(-.2)


def test_research_entry_points_smoke(prepared,tmp_path,monkeypatch):
    from sp500_ml import eda,modeling
    monkeypatch.setattr(eda,'RESAMPLES',50)
    report=eda.build_report(input_dir=prepared,output_dir=tmp_path/'eda')
    assert report['validation']['rows']==38*40
    assert (tmp_path/'eda/research_overview.html').exists()
    result=modeling.run(research_dir=prepared,output_dir=tmp_path/'models')
    assert len(result)==14  # 12 fitted variants plus two benchmark series
    validation=json.loads((tmp_path/'models/validation_report.json').read_text())
    assert validation['models']==12 and validation['months']==1
    assert validation['baseline_reproduction']=='not_checked'
