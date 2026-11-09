from pathlib import Path
from .provenance import compatible_source_digest, compatible_source_identity
"""Date-purged, matched-universe fundamental ablations against the sector model."""
import hashlib
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss,brier_score_loss,confusion_matrix
from sklearn.exceptions import ConvergenceWarning
from xgboost import XGBClassifier
from threadpoolctl import threadpool_limits
ROOT=Path.cwd()
from .constants import ALL_FACTORS,FACTORS,FAMILY
RESEARCH=ROOT/'data/prepared'
from .evaluation import design_pair,performance,paired_test,bh

OUT=ROOT/'outputs/fundamental_model_comparison'

SPECS={
 'Sector technical baseline':dict(representation=None),
 'Fundamentals only':dict(representation='score',technical=False),
 'Technicals + fundamental scores':dict(representation='score'),
 'Technicals + fundamental raw':dict(representation='raw'),
 'Technicals + raw and scores':dict(representation='both'),
 'Boosted technicals + fundamentals':dict(representation='both',boost=True),
 **{f'Technicals + {family}':dict(representation='score',family=family) for family in FACTORS},
 'Beat VOO technical baseline':dict(representation=None,target='outperform'),
 'Beat VOO technicals + fundamentals':dict(representation='score',target='outperform'),
}

def fundamental_design(train,test,representation,family=None,native_missing=False):
    factors=[x for x in ALL_FACTORS if family is None or FAMILY[x]==family]
    left=[];right=[];names=[];audit=[]
    for factor in factors:
        observed={}
        for kind in (['raw','sector_score'] if representation=='both' else ['raw'] if representation=='raw' else ['sector_score']):
            field=factor+'_'+kind
            a=pd.to_numeric(train[field],errors='coerce').to_numpy(dtype=float,copy=True)
            b=pd.to_numeric(test[field],errors='coerce').to_numpy(dtype=float,copy=True)
            observed[kind]=(np.isfinite(a),np.isfinite(b))
            if kind=='sector_score':a=a/100-.5;b=b/100-.5
            else:
                # Fold-only sector cleaning/scaling. No population-wide future fit.
                for sector in sorted(train.sector.dropna().unique()):
                    mask=train.sector.eq(sector).to_numpy();other=test.sector.eq(sector).to_numpy()
                    valid=a[mask&np.isfinite(a)]
                    if len(valid)<15:continue
                    low,high=np.quantile(valid,[.01,.99]);center=np.median(valid);q25,q75=np.quantile(valid,[.25,.75]);scale=q75-q25
                    if scale<=1e-12:scale=max(np.std(valid),1.)
                    a[mask]=(np.clip(a[mask],low,high)-center)/scale
                    b[other]=(np.clip(b[other],low,high)-center)/scale
                    audit.append(dict(feature=field,sector=sector,low=low,high=high,center=center,scale=scale,training_observations=len(valid)))
                a=np.clip(a,-10,10);b=np.clip(b,-10,10)
            # Gated representation for logistic: missing contribution is disabled,
            # with separate observed/applicable flags. No financial value is imputed
            # into the warehouse or a structurally inapplicable company.
            left.append(a if native_missing else np.nan_to_num(a,nan=0.));right.append(b if native_missing else np.nan_to_num(b,nan=0.));names.append(field)
        for kind,(a,b) in observed.items():
            left.append(a.astype(float));right.append(b.astype(float));names.append(factor+'_'+kind+'_observed')
        left.append(train[factor+'_applicable'].astype(float).to_numpy());right.append(test[factor+'_applicable'].astype(float).to_numpy());names.append(factor+'_applicable')
    for group in ([family] if family else FACTORS):
        field=group+'_composite';a=train[field].to_numpy(dtype=float)/100-.5;b=test[field].to_numpy(dtype=float)/100-.5
        left.extend([np.nan_to_num(a,nan=0.),np.isfinite(a).astype(float)])
        right.extend([np.nan_to_num(b,nan=0.),np.isfinite(b).astype(float)]);names.extend([field,field+'_observed'])
    quality_flags={
        'growth':['growth_unavailable_negative_base','growth_unavailable_small_base'],
        'profitability':['nonpositive_equity'],
        'valuation':['nonpositive_equity'],
        'balance_sheet':['debt_free','nonpositive_equity','no_interest_expense'],
    }
    for field in (quality_flags[family] if family else ['debt_free','nonpositive_equity','no_interest_expense','growth_unavailable_negative_base','growth_unavailable_small_base']):
        left.append(train[field].astype(float).to_numpy());right.append(test[field].astype(float).to_numpy());names.append(field)
    return np.column_stack(left),np.column_stack(right),names,audit

def design(train,test,technical,spec):
    if spec.get('technical',True):
        a,b,names,scaling=design_pair(train,test,technical,sectors=True)
    else:a=np.empty((len(train),0));b=np.empty((len(test),0));names=[];scaling=[]
    if spec['representation']:
        c,d,extra,audit=fundamental_design(train,test,spec['representation'],spec.get('family'),spec.get('boost',False))
        a=np.column_stack([a,c]);b=np.column_stack([b,d]);names+=extra;scaling+=audit
    return a,b,names,scaling

def labels(frame,spec):
    return frame.next_month_return.gt(frame.benchmark_return).astype(int) if spec.get('target')=='outperform' else frame.target_top20.astype(int)

@threadpool_limits.wrap(limits=1)
def fit_model(a,y,boost=False,rounds=80):
    if boost:
        return XGBClassifier(n_estimators=rounds,max_depth=2,min_child_weight=50,learning_rate=.05,reg_lambda=20,
            reg_alpha=0.,subsample=1.,colsample_bytree=1.,tree_method='hist',n_jobs=1,random_state=42,eval_metric='logloss').fit(a,y)
    with warnings.catch_warnings():
        warnings.filterwarnings('error',category=ConvergenceWarning)
        return LogisticRegression(C=1.,max_iter=2000,random_state=42).fit(a,y)

def run(*, research_dir=None, output_dir=None, reference_baseline=None):
    from pathlib import Path
    research=Path(research_dir) if research_dir is not None else RESEARCH
    out=Path(output_dir) if output_dir is not None else OUT
    out.mkdir(parents=True,exist_ok=True)
    from .dataset import validate_prepared
    validate_prepared(research)
    manifest=json.loads((research/'feature_manifest.json').read_text(encoding='utf8'))
    data=pd.read_parquet(research/'research_with_fundamentals.parquet').sort_values(['signal_date','ticker']).reset_index(drop=True)
    live=pd.read_parquet(research/'latest_with_fundamentals.parquet').sort_values('ticker').reset_index(drop=True)
    if 'target_top20' not in live:live['target_top20']=np.nan
    assert hashlib.sha256((research/'research_with_fundamentals.parquet').read_bytes()).hexdigest()==manifest['fundamental_dataset_sha256']
    assert live.next_month_return.isna().all()
    technical=[x for x in manifest['technical_features'] if not x.startswith('reversal_')]
    dates=sorted(data.signal_date.unique());latest_date=live.signal_date.iloc[0]
    output=[];windows=[];scales=[];coefficients=[];latest=[];importance=[]
    import importlib.metadata
    cache_contract={'manifest':manifest,'source':{p.name:compatible_source_digest(p) for p in Path(__file__).parent.glob('*.py')},'packages':{name:importlib.metadata.version(name) for name in ['numpy','pandas','scikit-learn','xgboost']}}
    fingerprint=hashlib.sha256(json.dumps(cache_contract,sort_keys=True).encode()).hexdigest()
    cache=out/'cache'/fingerprint[:16];cache.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):
        for name,spec in SPECS.items():
            cached=cache/(name.replace(' ','_').replace('+','plus')+'.parquet')
            if cached.exists():
                part=pd.read_parquet(cached);output.append(part.loc[part.signal_date.ne(latest_date)]);latest.append(part.loc[part.signal_date.eq(latest_date)])
                details=json.loads(cached.with_suffix('.json').read_text());windows+=details['windows'];scales+=details['scaling'];coefficients+=details['coefficients'];importance+=details['importance']
                print('Cached',name,flush=True);continue
            parts=[];audit_rows=[];scale_rows=[];coef_rows=[];importance_rows=[]
            for count,date in enumerate(list(dates[37:])+[latest_date],1):
                index=dates.index(date) if date!=latest_date else len(dates)
                train_dates=dates[index-37:index-1]
                assert len(train_dates)==36 and pd.Timestamp(train_dates[-1]).to_period('M')+1<pd.Timestamp(date).to_period('M')
                train=data.loc[data.signal_date.isin(train_dates)];test=live if date==latest_date else data.loc[data.signal_date.eq(date)]
                a,b,names,scaling=design(train,test,technical,spec);rounds=0
                if spec.get('boost'):
                    inner_dates=train_dates[:-7];validation_dates=train_dates[-6:]
                    assert pd.Timestamp(inner_dates[-1]).to_period('M')+1<pd.Timestamp(validation_dates[0]).to_period('M')
                    inner=train.loc[train.signal_date.isin(inner_dates)];validation=train.loc[train.signal_date.isin(validation_dates)]
                    ia,ib,_,_=design(inner,validation,technical,spec)
                    inner_model=fit_model(ia,labels(inner,spec),True,120)
                    losses={}
                    for rounds_candidate in [40,80,120]:
                        probability=inner_model.predict_proba(ib,iteration_range=(0,rounds_candidate))[:,1]
                        losses[rounds_candidate]=np.mean([log_loss(labels(validation.loc[validation.signal_date.eq(v)],spec),probability[validation.signal_date.eq(v).to_numpy()],labels=[0,1]) for v in validation_dates])
                    rounds=min(losses,key=losses.get)
                model=fit_model(a,labels(train,spec),spec.get('boost',False),rounds or 80)
                result=test[['signal_date','ticker','sector','next_month_return','benchmark_return','target_top20']].copy()
                result['target_top20']=result.target_top20.astype(float)
                result['probability']=model.predict_proba(b)[:,1]
                result=result.sort_values(['probability','ticker'],ascending=[False,True]);result['selected']=np.arange(len(result))<30
                result['model']=name;result['training_target']='outperform' if spec.get('target')=='outperform' else 'top20'
                parts.append(result)
                audit_rows.append(dict(model=name,test_date=date,train_start=train_dates[0],train_end=train_dates[-1],
                    training_months=36,purge_months=1,predictors=len(names),boosting_rounds=rounds,
                    inner_train_months=29 if spec.get('boost') else 0,inner_validation_months=6 if spec.get('boost') else 0,
                    inner_train_end=train_dates[-8] if spec.get('boost') else None,
                    inner_validation_start=train_dates[-6] if spec.get('boost') else None,
                    inner_validation_end=train_dates[-1] if spec.get('boost') else None))
                scale_rows.extend(dict(model=name,test_date=date,train_end=train_dates[-1],**r) for r in scaling)
                if spec.get('boost'):
                    importance_rows.extend(dict(model=name,test_date=date,feature=f,importance=float(v)) for f,v in zip(names,model.feature_importances_))
                else:
                    deviations=np.std(a,axis=0)
                    coef_rows.extend(dict(model=name,test_date=date,feature=f,coefficient=float(v),training_std=float(s),standardized_coefficient=float(v*s)) for f,v,s in zip(names,model.coef_[0],deviations))
                if count%20==0:print(name,count,'fits',flush=True)
            part=pd.concat(parts,ignore_index=True);part.to_parquet(cached,index=False)
            cached.with_suffix('.json').write_text(json.dumps(dict(windows=audit_rows,scaling=scale_rows,coefficients=coef_rows,importance=importance_rows),default=str))
            output.append(part.loc[part.signal_date.ne(latest_date)]);latest.append(part.loc[part.signal_date.eq(latest_date)])
            windows+=audit_rows;scales+=scale_rows;coefficients+=coef_rows;importance+=importance_rows
    predictions=pd.concat(output,ignore_index=True);live_predictions=pd.concat(latest,ignore_index=True)
    assert predictions.groupby(['model','signal_date']).selected.sum().eq(30).all()
    assert live_predictions.next_month_return.isna().all()
    predictions.to_parquet(out/'all_predictions.parquet',index=False);predictions.loc[predictions.selected].to_csv(out/'selected_top30.csv',index=False)
    live_predictions.to_parquet(out/'latest_unscored_predictions.parquet',index=False);live_predictions.loc[live_predictions.selected].to_csv(out/'latest_unscored_top30.csv',index=False)
    pd.DataFrame(windows).to_csv(out/'training_windows.csv',index=False);pd.DataFrame(scales).to_parquet(out/'scaling_audit.parquet',index=False)
    coef=pd.DataFrame(coefficients);coef.to_parquet(out/'coefficient_audit.parquet',index=False)
    coef['year']=pd.to_datetime(coef.test_date).dt.year
    coef.groupby(['model','year','feature']).agg(mean_effect=('standardized_coefficient','mean'),effect_std=('standardized_coefficient','std'),positive_fraction=('coefficient',lambda s:float(s.gt(0).mean()))).to_csv(out/'feature_stability_by_year.csv')
    pd.DataFrame(importance).to_csv(out/'boosted_feature_importance.csv',index=False)
    monthly=[]
    for (name,date),g in predictions.groupby(['model','signal_date']):
        picks=g.loc[g.selected]
        if len(picks)!=30 or not np.isfinite(picks.next_month_return).all():raise ValueError('All 30 selected outcomes must be known')
        top=g.target_top20.astype(int);beat=g.next_month_return.gt(g.benchmark_return).astype(int)
        actual=beat if g.training_target.iloc[0]=='outperform' else top
        monthly.append(dict(model=name,signal_date=date,training_target=g.training_target.iloc[0],portfolio=picks.next_month_return.mean(),
            VOO=g.benchmark_return.iloc[0],AUC_top20=roc_auc_score(top,g.probability),AUC_outperform=roc_auc_score(beat,g.probability) if beat.nunique()>1 else np.nan,
            average_precision=average_precision_score(actual,g.probability),log_loss=log_loss(actual,g.probability,labels=[0,1]),
            brier=brier_score_loss(actual,g.probability),top20_hit_rate=picks.target_top20.mean(),
            outperform_hit_rate=picks.next_month_return.gt(picks.benchmark_return).mean(),rank_IC=g.probability.corr(g.next_month_return,method='spearman')))
    monthly=pd.DataFrame(monthly);monthly['return_month']=(monthly.signal_date.dt.to_period('M')+1).dt.to_timestamp('M');monthly.to_csv(out/'monthly_results.csv',index=False)
    selected=predictions.loc[predictions.selected].copy()
    sector=selected.groupby(['model','signal_date','sector']).agg(holdings=('ticker','size'),mean_holding_return=('next_month_return','mean'),return_sum=('next_month_return','sum')).reset_index()
    sector['portfolio_weight']=sector.holdings/30;sector['portfolio_return_contribution']=sector.return_sum/30
    sector.to_csv(out/'sector_allocations_and_contributions.csv',index=False)
    returns=monthly.pivot(index='signal_date',columns='model',values='portfolio');returns['VOO']=data.groupby('signal_date').benchmark_return.first().reindex(returns.index)
    returns['Eligible universe equal weight']=data.groupby('signal_date').next_month_return.mean().reindex(returns.index)
    baseline_matches=None
    if reference_baseline is not None:
        old=pd.read_csv(reference_baseline,parse_dates=['signal_date'])
        old=old.loc[old.model.eq('Sector technical baseline')].set_index('signal_date').portfolio
        baseline_matches=bool(np.allclose(returns['Sector technical baseline'],old.reindex(returns.index),atol=1e-10))
        if not baseline_matches:raise ValueError('Explicit reference baseline failed to reproduce')
    summary=performance(returns).reset_index(names='model')
    diagnostics=monthly.groupby('model')[['AUC_top20','AUC_outperform','average_precision','log_loss','brier','top20_hit_rate','outperform_hit_rate','rank_IC']].mean()
    summary=summary.merge(diagnostics,on='model',how='left');summary.to_csv(out/'performance_summary.csv',index=False);returns.to_csv(out/'matched_monthly_returns.csv')
    calendar=returns.copy();calendar.index=(calendar.index.to_period('M')+1).to_timestamp('M')
    annual=(1+calendar).groupby(calendar.index.year).prod()-1;annual.insert(0,'months',calendar.groupby(calendar.index.year).size());annual.to_csv(out/'calendar_year_returns.csv')
    tests=[]
    for name in SPECS:
        if name=='Sector technical baseline':continue
        for comparator in ['Sector technical baseline','VOO']:
            delta=returns[name]-returns[comparator]
            tests.append(dict(model=name,comparator=comparator,**paired_test(delta),p_6month_blocks=paired_test(delta,6)['p']))
    tests=pd.DataFrame(tests);tests['BH_q']=bh(tests.p);tests['BY_q']=np.minimum(1,tests.BH_q*sum(1/np.arange(1,len(tests)+1)));tests.to_csv(out/'paired_return_tests.csv',index=False)
    plot_results(returns,monthly,predictions,output_dir=out)
    report=dict(status='passed',models=len(SPECS),months=len(returns),first_return_month=str(calendar.index.min().date()),
        last_return_month=str(calendar.index.max().date()),dataset_sha256=manifest['fundamental_dataset_sha256'],cache_fingerprint=fingerprint,
        all_models_same_universe=True,no_fundamental_complete_case_filter=True,trading_cost_bps=0,training_months=36,purge_months=1,
        predictor_policy='Explicit values + observed/applicable gates; warehouse NULL remains NULL; fold-only raw cleaning/scaling',
        raw_missing_handling='logistic zero contribution plus observed/applicable flags; boosted values retain NaN',
        primary_target='top20',secondary_target='outperform (separately named)',baseline_reproduction='not_checked' if baseline_matches is None else 'passed',source=manifest.get('source',{}),
        no_model_auto_promoted=True,pit_revision_uncertain=True,
        limitations='Current vendor revisions, retrospective sectors/industries, observed-outcome universe, same-close execution, reused history.')
    (out/'validation_report.json').write_text(json.dumps(report,indent=2))
    (out/'findings.md').write_text('# Fundamental walk-forward comparison\n\n'+json.dumps(report,indent=2)+'\n\n'+summary.to_string(index=False,float_format=lambda v:f'{v:.4f}')+'\n\n'
        'All 28 baseline factors and the two financial substitutes are tested without EDA-based selection. '
        'Family ablations isolate growth, profitability, valuation and financial strength. '
        'All models retain the existing technical complete-case universe; missing fundamentals do not remove stocks. '
        'AUC_top20 and AUC_outperform evaluate every strategy against separately named common labels. '
        'Boosting rounds use six inner validation months with a one-month inner purge and 29 inner training months. '
        'The existing technical-sector model remains primary. No model is promoted automatically.\n',encoding='utf-8')
    print(summary.to_string(index=False),flush=True);return summary

def plot_results(returns,monthly,predictions,*,output_dir=None):
    from pathlib import Path
    out=Path(output_dir) if output_dir is not None else OUT
    plt.style.use('seaborn-v0_8-whitegrid');dates=(returns.index.to_period('M')+1).to_timestamp('M')
    for filename,names in [('01_fundamental_performance.png',['Sector technical baseline','Fundamentals only','Technicals + fundamental scores','Technicals + fundamental raw','Technicals + raw and scores','Boosted technicals + fundamentals','VOO']),
                           ('02_family_ablations.png',['Sector technical baseline']+[f'Technicals + {x}' for x in FACTORS]+['VOO']),
                           ('03_target_comparison.png',['Sector technical baseline','Technicals + fundamental scores','Beat VOO technical baseline','Beat VOO technicals + fundamentals','VOO'])]:
        fig,axes=plt.subplots(2,1,figsize=(14,9),sharex=True,gridspec_kw={'height_ratios':[2,1]})
        for name in names:
            w=(1+returns[name]).cumprod();axes[0].plot([dates.min()-pd.offsets.MonthEnd(1)]+list(dates),[100]+list(100*w),label=name)
            axes[1].plot(dates,100*(w/w.cummax().clip(lower=1)-1))
        axes[0].set(title='Fundamental ablations | matched walk-forward months | zero trading costs',ylabel='Growth of $100');axes[0].legend(ncol=2,fontsize=8)
        axes[1].set(ylabel='Month-end drawdown (%)',xlabel='Realized return month');fig.tight_layout();fig.savefig(out/filename,dpi=150);plt.close(fig)
    chosen=['Sector technical baseline','Technicals + fundamental scores','Boosted technicals + fundamentals']
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,name in zip(axes,chosen):
        g=predictions.loc[predictions.model.eq(name)];cm=confusion_matrix(g.target_top20,g.selected,labels=[False,True]);ax.imshow(cm,cmap='Blues')
        for y in range(2):
            for x in range(2):ax.text(x,y,f'{cm[y,x]:,}',ha='center',va='center',color='white' if cm[y,x]>cm.max()/2 else 'black')
        ax.set(xticks=[0,1],xticklabels=['Not selected','Top 30'],yticks=[0,1],yticklabels=['Not top quintile','Top quintile'],title=name)
    fig.suptitle('Top-30 selection confusion matrices | actual next-month top quintile');fig.tight_layout();fig.savefig(out/'04_confusion_matrices.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,5))
    for name in chosen:
        g=monthly.loc[monthly.model.eq(name)].set_index('return_month');g.AUC_top20.rolling(6).mean().plot(ax=ax,label=name)
    ax.axhline(.5,color='black',ls='--');ax.set(title='Six-month mean AUC | next-month top-quintile label',ylabel='AUC',xlabel='Return month');ax.legend();fig.tight_layout();fig.savefig(out/'05_auc_over_time.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,6))
    for name in chosen:
        g=predictions.loc[predictions.model.eq(name)].copy();g['bucket']=pd.cut(g.probability,bins=np.linspace(0,1,11),include_lowest=True)
        bins=g.groupby('bucket',observed=True).agg(predicted=('probability','mean'),actual=('target_top20','mean'))
        ax.plot(bins.predicted,bins.actual,'o-',label=name)
    ax.plot([0,1],[0,1],ls='--',color='black');ax.set(title='Probability calibration | retrospective diagnostic',xlabel='Mean predicted probability',ylabel='Observed top-quintile frequency');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'06_calibration.png',dpi=150);plt.close(fig)

def show_results():
    from IPython.display import display,Image,Markdown
    display(Markdown((OUT/'findings.md').read_text(encoding='utf-8').split('\n\n')[0]))
    display(pd.read_csv(OUT/'performance_summary.csv'))
    display(pd.read_csv(OUT/'calendar_year_returns.csv'))
    for name in ['01_fundamental_performance.png','02_family_ablations.png','03_target_comparison.png','04_confusion_matrices.png','05_auc_over_time.png','06_calibration.png']:
        display(Image(filename=str(OUT/name)))
    display(pd.read_csv(OUT/'paired_return_tests.csv'))
    display(Markdown('**Limits:** vendor revisions and historical classifications are uncertain. Same universe, zero costs, 36-month training and one-month purges. Existing model remains primary.'))

if __name__=='__main__':run()
