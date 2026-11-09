"""Readable, cached-data EDA of features versus forward returns; no SQL writes or fitting."""
from pathlib import Path
import hashlib
import html
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
ROOT = Path.cwd()

OUT = ROOT / 'outputs/research_overview'
SEED = 42
RESAMPLES = 5000
BLOCK = 3
MIN_MONTHS = 24
COLORS = ['#156b83', '#db9b42', '#668578', '#856799', '#7c8795']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def adjust(pvalues, dependent=False):
    p = np.asarray(pvalues, float)
    result = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    m = len(order)
    if m:
        multiplier = sum(1/np.arange(1, m+1)) if dependent else 1.
        result[order] = np.minimum(1, np.minimum.accumulate((p[order]*m*multiplier/np.arange(1,m+1))[::-1])[::-1])
    return result


def block_indices(length, block=BLOCK, draws=None):
    draws=RESAMPLES if draws is None else draws
    rng = np.random.default_rng(SEED)
    start = rng.integers(length, size=(draws, int(np.ceil(length/block))))
    return ((start[..., None]+np.arange(block)) % length).reshape(draws,-1)[:, :length]


def summarize_draws(estimate, draws, n):
    draws = np.asarray(draws)
    draws = draws[np.isfinite(draws)]
    if n < MIN_MONTHS or not len(draws):
        return dict(estimate=estimate, low=np.nan, high=np.nan, p=np.nan, months=n)
    errors = draws-estimate
    low, high = np.quantile(errors,[.025,.975])
    p = min(1., 2*min((1+sum(errors >= estimate))/(len(errors)+1),
                     (1+sum(errors <= estimate))/(len(errors)+1)))
    return dict(estimate=float(estimate), low=float(estimate-high), high=float(estimate-low), p=float(p), months=n)


def mean_test(series, indices):
    x = np.asarray(series,float)
    n = int(np.isfinite(x).sum())
    estimate = float(np.nanmean(x)) if n else np.nan
    sampled = x[indices]
    count = np.isfinite(sampled).sum(axis=1)
    draws = np.divide(np.nansum(sampled,axis=1),count,out=np.full(len(count),np.nan),where=count>0)
    return summarize_draws(estimate,draws,n)


def correlation_test(x,y,indices):
    # Re-rank paired monthly observations within every resampled chronology.
    x,y = np.asarray(x,float),np.asarray(y,float)
    valid = np.isfinite(x)&np.isfinite(y)
    x,y = np.where(valid,x,np.nan),np.where(valid,y,np.nan)
    def correlate(a,b,axis):
        a=rankdata(a,axis=axis,nan_policy='omit');b=rankdata(b,axis=axis,nan_policy='omit')
        a=a-np.nanmean(a,axis=axis,keepdims=True);b=b-np.nanmean(b,axis=axis,keepdims=True)
        numerator=np.nansum(a*b,axis=axis)
        denominator=np.sqrt(np.nansum(a*a,axis=axis)*np.nansum(b*b,axis=axis))
        return np.divide(numerator,denominator,out=np.full_like(numerator,np.nan,dtype=float),where=denominator>0)
    estimate=float(correlate(x,y,0))
    return summarize_draws(estimate,correlate(x[indices],y[indices],1),int(valid.sum()))


def monthly_evidence(frame,feature,min_rows=30):
    rows=[];buckets=[]
    for date,g in frame.groupby('signal_date',sort=True):
        g=g[[feature,'next_month_return','benchmark_return']].replace([np.inf,-np.inf],np.nan).dropna()
        if len(g)<min_rows or g[feature].nunique()<5:
            continue
        x=g[feature].rank(method='average');y=g.next_month_return.rank(method='average')
        ic=x.corr(y)
        # Ties remain together, rather than assigning identical values arbitrary ranks.
        bucket=np.ceil(5*x/len(g)).clip(1,5).astype(int)
        excess=g.next_month_return-g.benchmark_return
        returns=excess.groupby(bucket).mean()
        spread=returns.get(5,np.nan)-returns.get(1,np.nan)
        rows.append(dict(signal_date=date,ic=ic,spread=spread,observations=len(g)))
        buckets.extend(dict(signal_date=date,bucket=int(q),excess_return=float(v)) for q,v in returns.items())
    index=pd.DatetimeIndex(sorted(frame.signal_date.unique()),name='signal_date')
    series=pd.DataFrame(rows,columns=['signal_date','ic','spread','observations']).set_index('signal_date').reindex(index)
    return series,pd.DataFrame(buckets,columns=['signal_date','bucket','excess_return'])


def evidence_table(tests,block,limit=10):
    t=tests.loc[tests.block.eq(block)].copy()
    t=t.sort_values('effect',key=lambda x:x.abs(),ascending=False).head(limit)
    return pd.DataFrame({
        'Feature':t.label, 'Association':t.effect.round(3),
        '95% interval':[f'{a:.3f} to {b:.3f}' for a,b in zip(t.low,t.high)],
        'Q5 − Q1 (pp/month)':(100*t.spread).round(2),
        'Adjusted q':t.q.map(lambda x:f'{x:.3g}' if pd.notna(x) else '—'),
        'Valid months':t.months.astype(int),'Coverage %':(100*t.coverage).round(1),
    }).reset_index(drop=True)


def build_report(labels=None, *, input_dir="data/prepared", output_dir=None):
    from .dataset import validate_prepared
    input_dir=Path(input_dir)
    validate_prepared(input_dir)
    INPUTS={k:input_dir/name for k,name in dict(research='research_with_fundamentals.parquet',latest='latest_with_fundamentals.parquet',manifest='feature_manifest.json',population='population.parquet',candidates='candidates.parquet').items()}
    out=Path(output_dir) if output_dir is not None else OUT
    out.mkdir(parents=True,exist_ok=True)
    labels=labels or {}
    manifest=json.loads(INPUTS['manifest'].read_text(encoding='utf8'))
    source_evidence=manifest.get('source',{})
    d=pd.read_parquet(INPUTS['research']);latest=pd.read_parquet(INPUTS['latest'])
    pop=pd.read_parquet(INPUTS['population']);candidates=pd.read_parquet(INPUTS['candidates'])
    assert not d.duplicated(['signal_date','ticker']).any()
    assert not pop.duplicated(['signal_date','ticker']).any()
    assert d[['next_month_return','benchmark_return','target_top20']].notna().all().all()
    assert latest.next_month_return.isna().all()
    matched=d[['signal_date','ticker','security_id']].merge(pop[['signal_date','ticker','security_id']],on=['signal_date','ticker'],validate='one_to_one',suffixes=('_research','_population'))
    assert len(matched)==len(d) and matched.security_id_research.eq(matched.security_id_population).all()
    dates=pd.DatetimeIndex(sorted(d.signal_date.unique()),name='signal_date')
    ix=block_indices(len(dates));ix6=block_indices(len(dates),block=6)
    market=d.groupby('signal_date').agg(next_month_return=('next_month_return','mean'),benchmark_return=('benchmark_return','first'))
    assert d.groupby('signal_date').benchmark_return.nunique().eq(1).all()
    sector=d.groupby(['signal_date','sector'],as_index=False).agg(next_month_return=('next_month_return','mean'),benchmark_return=('benchmark_return','first'))
    for feature in manifest['sector_context_features']:
        assert d.groupby(['signal_date','sector'])[feature].nunique().le(1).all()
        sector=sector.merge(d.groupby(['signal_date','sector'],as_index=False)[feature].first(),on=['signal_date','sector'],validate='one_to_one')
    feature_specs=[]
    # Exact negative reversals add no information to their return counterparts.
    for f in manifest['technical_features']:
        if not f.startswith('reversal_'):feature_specs.append((f,'Stock technicals','stock'))
    feature_specs += [(f,'Stock vs sector','stock') for f in manifest['sector_relative_features']]
    feature_specs += [(f+'_sector_score','Fundamentals','stock') for f in manifest['fundamental_factors']]
    feature_specs += [(f,'Fundamental composites','stock') for f in manifest['fundamental_composites']]
    feature_specs += [(f,'Sector context','sector') for f in manifest['sector_context_features']]
    feature_specs += [(f,'Macro context','macro') for f in manifest['macro_features']]
    flag_names=['debt_free','nonpositive_equity','no_interest_expense','growth_unavailable_negative_base','growth_unavailable_small_base']
    names={}
    def label(f):
        base=f.removesuffix('_sector_score')
        return labels.get(base,base.replace('_',' ').replace('yoy','YoY').replace('ocf','operating cash flow').replace('fcf','free cash flow').replace('ebitda','EBITDA').replace('roic','ROIC').replace('roe','ROE').replace('roa','ROA').title())
    tests=[];series_rows=[];bucket_rows=[]
    for i,(f,block,grain) in enumerate(feature_specs):
        names[f]=label(f)
        source=sector if grain=='sector' else d
        applicable=f.removesuffix('_sector_score')+'_applicable'
        eligible=d[applicable].astype(bool) if applicable in d else pd.Series(True,index=d.index)
        coverage=float(d.loc[eligible,f].notna().mean()) if eligible.any() else np.nan
        if grain=='macro':
            assert d.groupby('signal_date')[f].nunique().le(1).all()
            x=d.groupby('signal_date')[f].first().reindex(dates)
            a=correlation_test(x,market.next_month_return.reindex(dates),ix)
            sensitivity=correlation_test(x,market.next_month_return.reindex(dates),ix6)
            b=dict(estimate=np.nan,low=np.nan,high=np.nan,p=np.nan,months=0)
            early=late=same=np.nan
        else:
            series,buckets=monthly_evidence(source,f,min_rows=8 if grain=='sector' else 30)
            series=series.reindex(dates)
            a=mean_test(series.ic,ix);b=mean_test(series.spread,ix);sensitivity=mean_test(series.ic,ix6)
            same=float((series.ic.dropna()*np.sign(a['estimate'])>0).mean())
            early=float(series.loc[series.index<'2024-01-01','ic'].mean());late=float(series.loc[series.index>='2024-01-01','ic'].mean())
            series_rows.append(series.reset_index().assign(feature=f,block=block))
            bucket_rows.append(buckets.assign(feature=f,block=block))
        tests.append(dict(feature=f,label=names[f],block=block,grain=grain,effect=a['estimate'],low=a['low'],high=a['high'],p=a['p'],months=a['months'],
            spread=b['estimate'],spread_low=b['low'],spread_high=b['high'],spread_p=b['p'],coverage=coverage,
            same_sign_months=same,early_ic=early,later_ic=late,p_six_month_blocks=sensitivity['p']))
        if (i+1)%30==0:print(f'Analyzed {i+1}/{len(feature_specs)} features',flush=True)
    t=pd.DataFrame(tests)
    # One multiplicity family across ALL screened blocks and both tested endpoints.
    p=np.r_[t.p.to_numpy(),t.spread_p.to_numpy()]
    q=adjust(p);by=adjust(p,dependent=True)
    t['q']=q[:len(t)];t['spread_q']=q[len(t):];t['q_BY']=by[:len(t)];t['spread_q_BY']=by[len(t):]
    t['q_six_month_blocks']=adjust(t.p_six_month_blocks)
    t.to_csv(out/'feature_tests.csv',index=False)
    monthly=pd.concat(series_rows,ignore_index=True);monthly.to_parquet(out/'monthly_feature_evidence.parquet',index=False)
    buckets=pd.concat(bucket_rows,ignore_index=True);buckets.to_csv(out/'feature_quintiles.csv',index=False)
    coverage=pd.DataFrame({'Historical population':pop.groupby('signal_date').ticker.nunique(),
        'Technical/macro ready':candidates.groupby('signal_date').ticker.nunique(),
        'Observed-return EDA':d.groupby('signal_date').ticker.nunique()}).fillna(0).astype(int)
    coverage.to_csv(out/'monthly_population.csv')
    funnel=pd.DataFrame({'Stage':['All historical membership snapshots','Technical/macro-ready snapshots','Observed-return EDA rows','Ready but not in observed-return EDA','Latest feature-only candidates'],
        'Stock-months':[len(pop),len(candidates),len(d),len(candidates)-len(d),len(latest)]})
    r=d.next_month_return;excess=r-d.benchmark_return
    target=pd.DataFrame({
        'Statistic':['Stock-months','Mean','Median','Standard deviation','1st percentile','5th percentile','25th percentile','75th percentile','95th percentile','99th percentile','Minimum','Maximum','Skewness','Positive return rate','Beats VOO rate','Top-20% label rate'],
        'Next-month stock return':[f'{len(d):,}',*[f'{x:.2%}' for x in [r.mean(),r.median(),r.std(),r.quantile(.01),r.quantile(.05),r.quantile(.25),r.quantile(.75),r.quantile(.95),r.quantile(.99),r.min(),r.max()]],f'{r.skew():.2f}',f'{r.gt(0).mean():.2%}',f'{excess.gt(0).mean():.2%}',f'{d.target_top20.mean():.2%}']})
    target.to_csv(out/'target_descriptive_statistics.csv',index=False)
    family=[]
    for block,g in t.groupby('block',sort=False):
        estimable=g.loc[g.effect.notna()]
        winner=estimable.loc[estimable.effect.abs().idxmax()] if len(estimable) else pd.Series({'label':'Not estimable','effect':np.nan})
        family.append(dict(Block=block,Features=len(g),**{'Median coverage %':round(100*g.coverage.median(),1),'q < 5%':int(g.q.lt(.05).sum()),'Strongest descriptive association':winner.label,'Association':round(winner.effect,3)}))
    family=pd.DataFrame(family)
    figures={}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15,'figure.facecolor':'white','axes.facecolor':'white','legend.frameon':False})
    def save(fig,name):
        fig.savefig(out/(name+'.png'),dpi=145,bbox_inches='tight');plt.close(fig);figures[name]=str(out/(name+'.png'))
    fig,axes=plt.subplots(1,2,figsize=(13,4.2),layout='constrained')
    coverage.replace(0,np.nan).plot(ax=axes[0],color=[COLORS[4],COLORS[1],COLORS[0]])
    positive=coverage.where(coverage.gt(0)).stack()
    axes[0].set_ylim(positive.min()-5,positive.max()+5)
    axes[0].set(title='Population to analysis sample',xlabel='Signal month',ylabel='Unique securities')
    last=pop.signal_date.max();counts=pop.loc[pop.signal_date.eq(last)].groupby('sector').size().sort_values()
    counts.plot.barh(ax=axes[1],color=COLORS[0]);axes[1].set(title=f'Historical members at {last:%Y-%m-%d}',xlabel='Securities',ylabel='')
    save(fig,'01_population')
    fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained')
    low,high=r.quantile([.01,.99]);central=r.between(low,high)
    axes[0].hist(r.loc[central]*100,bins=55,color=COLORS[0],alpha=.9)
    axes[0].axvline(100*r.median(),color=COLORS[1],label=f'Median {r.median():.2%}');axes[0].legend()
    axes[0].set(title='Next-month return: central 98% shown',xlabel='Stock return (%)',ylabel='Stock-months')
    calendar=market.copy();calendar.index=(calendar.index.to_period('M')+1).to_timestamp('M')
    calendar.rename(columns={'next_month_return':'Equal-weight observed sample','benchmark_return':'VOO'}).plot(ax=axes[1],color=COLORS[:2])
    axes[1].yaxis.set_major_formatter(PercentFormatter(1));axes[1].set(title='Monthly return context',xlabel='Realized return month',ylabel='Return')
    save(fig,'02_target')
    fig,ax=plt.subplots(figsize=(12,4),layout='constrained')
    for block in ['Stock technicals','Fundamentals','Sector context','Macro context']:
        fs=t.loc[t.block.eq(block),'feature'];available=d[fs].notna().mean(axis=1).groupby(d.signal_date).mean()
        ax.plot(available.index,available,label=block)
    ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set(ylim=(0,1.05),title='Feature availability in the observed-return sample',xlabel='Signal month',ylabel='Fraction available');ax.legend(ncol=2)
    save(fig,'03_feature_coverage')
    def forest(group,name,title,limit=10):
        g=group.dropna(subset=['effect']).sort_values('effect',key=lambda x:x.abs(),ascending=False).head(limit).sort_values('effect')
        fig,axes=plt.subplots(1,2,figsize=(13,max(3.5,len(g)*.34)),layout='constrained')
        for ax,value,lo,hi,scale in [(axes[0],'effect','low','high',1),(axes[1],'spread','spread_low','spread_high',100)]:
            for j,row in enumerate(g.itertuples()):
                v=getattr(row,value)*scale;a=getattr(row,lo)*scale;b=getattr(row,hi)*scale
                color=COLORS[0] if row.q<.05 else COLORS[4]
                if np.isfinite(v):ax.plot(v,j,'o',color=color)
                if np.isfinite(a) and np.isfinite(b):ax.plot([a,b],[j,j],color=color,lw=2)
            ax.axvline(0,color='#a0a0a0',lw=1);ax.set_yticks(range(len(g)),g.label if ax==axes[0] else []);ax.set_ylim(-.7,len(g)-.3)
        axes[0].set(title=title,xlabel='Mean monthly rank correlation (95% interval)')
        axes[1].set(title='Higher-value minus lower-value quintile',xlabel='Forward-return difference (percentage points/month)')
        save(fig,name)
    forest(t.loc[t.block.isin(['Stock technicals','Stock vs sector'])],'04_technicals','Stock signals: descriptive association',10)
    forest(t.loc[t.block.eq('Fundamental composites')],'05_fundamental_families','Fundamental families: higher score = more favorable',4)
    forest(t.loc[t.block.eq('Fundamentals')],'06_fundamental_factors','Individual sector-relative fundamental scores',8)
    macro_examples=['vix_level','treasury_10y_yield_pct','curve_10y_2y_pp','oil_return_63d','gold_return_63d','credit_hyg_minus_lqd_return_63d']
    fig,axes=plt.subplots(1,2,figsize=(14,4.6),layout='constrained')
    context_groups=[t.loc[t.feature.isin(macro_examples)].sort_values('effect'),t.loc[t.block.eq('Sector context')].sort_values('effect',key=lambda x:x.abs(),ascending=False).head(6).sort_values('effect')]
    for ax,g,title,xlabel in zip(axes,context_groups,['Macro examples: market timing association','Sector context: cross-sector association'],['Time-series rank correlation (95% interval)','Mean monthly sector rank correlation (95% interval)']):
        for j,row in enumerate(g.itertuples()):
            color=COLORS[0] if row.q<.05 else COLORS[4]
            ax.plot(row.effect,j,'o',color=color)
            ax.plot([row.low,row.high],[j,j],color=color,lw=2)
        ax.axvline(0,color='#a0a0a0',lw=1);ax.set_yticks(range(len(g)),g.label);ax.set(title=title,xlabel=xlabel)
    save(fig,'09_macro_sector_context')
    representatives=['momentum_12_1','return_21d','realized_vol_63d','max_drawdown_252d','price_to_sma200','log_dollar_volume_21d',*manifest['fundamental_composites']]
    corr=d.groupby('signal_date')[representatives].rank(pct=True).corr()
    corr.to_csv(out/'representative_rank_correlations.csv')
    d.groupby('signal_date')[t.loc[t.grain.eq('stock'),'feature'].tolist()].rank(pct=True).corr().to_csv(out/'all_stock_feature_rank_correlations.csv')
    d.groupby('signal_date')[manifest['macro_features']].first().corr(method='spearman').to_csv(out/'macro_monthly_correlations.csv')
    fig,ax=plt.subplots(figsize=(10,7),layout='constrained');im=ax.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r')
    ticks=[label(x) for x in representatives];ax.set_xticks(range(len(ticks)),ticks,rotation=55,ha='right');ax.set_yticks(range(len(ticks)),ticks)
    for i in range(len(corr)):
        for j in range(len(corr)):ax.text(j,i,f'{corr.iloc[i,j]:.1f}',ha='center',va='center',fontsize=8,color='white' if abs(corr.iloc[i,j])>.65 else '#273746')
    fig.colorbar(im,ax=ax,shrink=.7,label='Correlation of within-month feature ranks');ax.set_title('Where signals overlap');save(fig,'07_correlations')
    chosen=pd.concat([t.loc[t.block.eq('Stock technicals')].sort_values('effect',key=lambda x:x.abs(),ascending=False).head(4),t.loc[t.block.eq('Fundamentals')].sort_values('effect',key=lambda x:x.abs(),ascending=False).head(4)])
    annual=monthly.loc[monthly.feature.isin(chosen.feature)].assign(year=lambda x:pd.to_datetime(x.signal_date).dt.year).groupby(['feature','year']).ic.mean().unstack().reindex(chosen.feature)
    annual.to_csv(out/'annual_signal_stability.csv')
    fig,ax=plt.subplots(figsize=(11,4.7),layout='constrained');bound=max(.05,float(np.nanmax(abs(annual.to_numpy()))));im=ax.imshow(annual.to_numpy(dtype=float),vmin=-bound,vmax=bound,cmap='RdBu_r',aspect='auto')
    ax.set_yticks(range(len(annual)),[label(x) for x in annual.index]);ax.set_xticks(range(len(annual.columns)),annual.columns)
    for i in range(len(annual)):
        for j in range(len(annual.columns)):ax.text(j,i,f'{annual.iloc[i,j]:.2f}',ha='center',va='center',fontsize=8)
    ax.set_title('Do the strongest descriptive signals keep their direction?');fig.colorbar(im,ax=ax,shrink=.8,label='Mean monthly rank correlation');save(fig,'08_stability')
    # Missingness is shown separately, never mistaken for a favorable financial value.
    flags=[]
    for f in flag_names:
        g=d.assign(flag=d[f].astype(bool),excess=d.next_month_return-d.benchmark_return)
        contrast=g.groupby(['signal_date','flag']).excess.mean().unstack().reindex(dates)
        delta=contrast.get(True,pd.Series(np.nan,index=dates))-contrast.get(False,pd.Series(np.nan,index=dates))
        result=mean_test(delta,ix)
        flags.append({'Flag':label(f),'Prevalence %':round(100*g.flag.mean(),1),'Flagged − unflagged (pp/month)':round(100*result['estimate'],2),'Months with both groups':result['months']})
    flags=pd.DataFrame(flags);flags.to_csv(out/'quality_flag_descriptives.csv',index=False)
    strongest=t.loc[t.block.isin(['Stock technicals','Stock vs sector','Fundamentals'])].sort_values('effect',key=lambda x:x.abs(),ascending=False).head(8)
    conclusions=pd.DataFrame({'Feature':strongest.label,'Pattern':['Higher values → '+('higher' if x>0 else 'lower')+' forward return rank' for x in strongest.effect],
        'Adjusted q':strongest.q.map(lambda x:f'{x:.3g}'),'Same-direction months %':(100*strongest.same_sign_months).round(0),
        'Pre-2024 association':strongest.early_ic.round(3),'2024 onward':strongest.later_ic.round(3)})
    scope=pd.DataFrame({'Population / target':['Historical universe','Signal dates in EDA','Realized return months','Observed sample','Latest feature-only snapshot','Primary model label','Return outcome analyzed here','Known limitation'],
        'Definition':['Historical S&P 500 membership; latest ticker routes; distinct security IDs',f'{dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}',f'{dates.min().to_period("M")+1} to {dates.max().to_period("M")+1}',f'{len(d):,} stock-months; {d.ticker.nunique()} tickers; {len(dates)} months',f'{latest.signal_date.max():%Y-%m-%d}: {len(latest):,} candidates, excluded from outcome tests','Top 20% of observed stock returns within each signal month','Next-month adjusted close-to-close return; VOO-relative buckets','Historical outcome availability filters the EDA sample; current vendor revisions/sectors are retrospective']})
    discoveries=int(t.q.lt(.05).sum())
    discovery_text=(f'{discoveries} association tests passed the joint 5% BH false-discovery screen.' if discoveries else 'No individual association passed the joint 5% BH false-discovery screen. Some pointwise intervals exclude zero, but those results do not survive the full multiple-testing screen.')
    sections={
        'population':dict(title='1. Which stocks and dates are we studying?',text='Start with historical membership, then show technical/macro availability and the observed-return sample. These are monthly security counts, not a fixed list of today’s constituents. The newest snapshot has no known forward outcome and is excluded from return tests. The funnel spans all saved snapshots, including the newest; differences include warm-up/missing predictors and unknown outcomes.',tables=[scope,funnel],images=['01_population']),
        'target':dict(title='2. What does the target look like?',text='The continuous outcome is the next monthly adjusted stock return. The model’s top-20% label is a separate classification target; beating VOO is another separate label. Statistics retain all observed tails. The histogram zooms to the central 98% only; the context line is an equal-weight observed-stock average, not the deployed strategy.',tables=[target],images=['02_target']),
        'overview':dict(title='3. Technical and fundamental information: the big picture',text='Availability comes before signal strength. Fundamental scores have structural sector exclusions; the chart includes those exclusions, while test-table coverage uses applicable companies where defined. Association is mean monthly stock/sector rank correlation, except macro, which uses one time-series rank correlation. Compare blocks descriptively; this is not model feature importance.',tables=[family],images=['03_feature_coverage']),
        'technicals':dict(title='4. Which stock technicals describe forward returns?',text='Within each month, rank the feature and the next-month stock return. A positive correlation means higher feature values tended to precede higher return ranks. Q5 − Q1 compares the highest-value and lowest-value fifths using equal weights. Gray dots have association q ≥ 5%; teal dots pass that screen. Charts rank by absolute association over inspected history, not an untouched selection set. Reversal duplicates are omitted from inferential screening.',tables=[evidence_table(t,'Stock technicals',8),evidence_table(t,'Stock vs sector',3)],images=['04_technicals']),
        'fundamentals':dict(title='5. Which fundamental families and factors describe returns?',text='Begin with growth, profitability, valuation and balance-sheet composites, then individual sector-relative scores. A higher score follows the registry’s preferred direction, including reversed leverage ratios. Missing/inapplicable inputs are not filled with a financial value. Financial firms cannot meet the current two-factor minimum for the balance-sheet composite; their equity/assets factor remains available. Quality flags below are descriptive contrasts only and are not additional significance claims.',tables=[evidence_table(t,'Fundamental composites',4),evidence_table(t,'Fundamentals',8),flags],images=['05_fundamental_families','06_fundamental_factors']),
        'context':dict(title='6. What do macro and sector conditions tell us?',text='Macro is tested once per month against the following month’s equal-weight observed-universe return. Sector context is tested across equally weighted sector-months against their following stock-average returns; it does not treat hundreds of repeated ETF observations as independent. These answer different questions from choosing stocks within a sector. The macro chart shows fixed economic examples; the sector chart and tables show descriptively strongest associations. Sector scores and historical classifications retain the documented retrospective limitations.',tables=[evidence_table(t,'Macro context',6),evidence_table(t,'Sector context',6)],images=['09_macro_sector_context']),
        'relationships':dict(title='7. Are signals redundant or unstable?',text='The first chart compares within-month stock feature ranks, excluding repeated macro rows. A large correlation suggests overlap, not automatically a reason to delete a feature. The second shows annual rank associations for descriptively strong signals; changing color means changing direction. All feature and macro correlation matrices are exported separately.',tables=[],images=['07_correlations','08_stability']),
        'conclusions':dict(title='8. What is worth investigating next?',text=discovery_text+' Treat the descriptive leaders as candidate hypotheses: favor economic meaning, a material high-minus-low effect, adequate coverage and direction stability. Test incremental value with matched, training-only ablations in Rolling_Model.ipynb. Weak univariate evidence does not directly measure a multivariable model’s performance. Statistical association does not establish causality, independent information or portfolio benefit. No model, database table, feature selection or deployment policy is changed by this review.',tables=[conclusions],images=[]),
    }
    methods=f'Exploratory tests use {RESAMPLES:,} paired circular block resamples of {BLOCK} consecutive monthly dates (seed {SEED}), retaining the calendar and missing observations. Nulls: zero mean monthly rank correlation and zero mean Q5 − Q1 return spread; macro uses zero paired time-series Spearman correlation. Intervals are pointwise 95% basic-bootstrap intervals, not multiplicity-adjusted. Inference requires at least {MIN_MONTHS} valid months. BH q-values jointly adjust all screened features and both endpoints; dependence-conservative BY q-values and six-month-block association sensitivity are in feature_tests.csv. All dates have been inspected; there is no fresh confirmatory holdout here. Coverage and sample size differ across factors; these are univariate associations, not multivariable or randomized causal estimates.'
    input_hashes={'gold_source_binding_sha256':source_evidence['source_binding_sha256'],'feature_manifest':sha(INPUTS['manifest'])} if source_evidence else {k:sha(v) for k,v in INPUTS.items()}
    report=dict(status='passed',rows=len(d),months=len(dates),features_tested=len(t),tests_adjusted=int(np.isfinite(p).sum()),sections=list(sections),input_hashes=input_hashes,source=source_evidence,generator_sha256=sha(__file__),resamples=RESAMPLES,block_months=BLOCK,methods=methods,models_refit=False,database_written=False)
    (out/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    parts=['<!doctype html><html><head><meta charset="utf-8"><title>S&amp;P 500 Research</title><style>body{font:16px Arial;color:#243746;max-width:1200px;margin:40px auto;line-height:1.55}table{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0}th,td{padding:9px;border-bottom:1px solid #dce2e8;text-align:left}th{background:#f1f5f8}img{max-width:100%}p{max-width:1050px}</style></head><body><h1>S&amp;P 500: understanding the data before modeling</h1>']
    parts.append('<p>'+html.escape(methods)+'</p>')
    for section in sections.values():
        parts+=['<h2>'+html.escape(section['title'])+'</h2>','<p>'+html.escape(section['text'])+'</p>']
        parts += [x.to_html(index=False,escape=True) for x in section['tables']]
        parts += ['<img src="'+name+'.png" alt="'+name+'">' for name in section['images']]
    (out/'research_overview.html').write_text(''.join(parts)+'</body></html>',encoding='utf-8')
    return dict(output_dir=str(out),sections=sections,methods=methods,validation=report,tests=t)


def show_section(report,name):
    from IPython.display import display,Markdown,Image
    section=report['sections'][name]
    display(Markdown(section['text']))
    for table in section['tables']:display(table)
    for image in section['images']:display(Image(filename=str(Path(report.get('output_dir',OUT))/(image+'.png'))))
