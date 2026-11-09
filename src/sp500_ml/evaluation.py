import numpy as np
import pandas as pd
from .constants import PROXIES, CONTEXT_FEATURES, RELATIVE_FEATURES, SECTOR_FEATURES

STOCK_INTERACTIONS=['return_63d','realized_vol_63d','price_to_sma200']

BROAD_INTERACTIONS=['vix_level','treasury_10y_change_63d_bp','oil_return_63d','gold_return_63d','dollar_return_63d','market_return_63d']

def design_pair(train,test,technical,macros=(),interaction_macros=(),sectors=False,identity=False):
    """All centers/scales are estimated from training rows only."""
    train_r=train.groupby('signal_date')[technical+RELATIVE_FEATURES].rank(pct=True)-.5
    test_r=test.groupby('signal_date')[technical+RELATIVE_FEATURES].rank(pct=True)-.5
    left=[train_r[technical].to_numpy()];right=[test_r[technical].to_numpy()];names=list(technical);audit=[]
    if identity:
        for sector in PROXIES:
            names.append('sector_identity::'+sector);left.append(train.sector.eq(sector).to_numpy(dtype=float)[:,None]);right.append(test.sector.eq(sector).to_numpy(dtype=float)[:,None])
    if sectors:
        history=train[['signal_date','sector']+CONTEXT_FEATURES].drop_duplicates(['signal_date','sector'])
        center=history[CONTEXT_FEATURES].mean();scale=history[CONTEXT_FEATURES].std(ddof=0).replace(0,1.)
        left.extend([((train[CONTEXT_FEATURES]-center)/scale).clip(-5,5).to_numpy(),train_r[RELATIVE_FEATURES].to_numpy()])
        right.extend([((test[CONTEXT_FEATURES]-center)/scale).clip(-5,5).to_numpy(),test_r[RELATIVE_FEATURES].to_numpy()]);names+=SECTOR_FEATURES
        audit.extend(dict(block='sector',feature=f,training_mean=center[f],training_std=scale[f],unique_training_sector_months=len(history)) for f in CONTEXT_FEATURES)
    if macros:
        history=train.groupby('signal_date')[list(macros)].first();center=history.mean();scale=history.std(ddof=0).replace(0,1.)
        tz=((train[list(macros)]-center)/scale).clip(-5,5);vz=((test[list(macros)]-center)/scale).clip(-5,5)
        left.append(tz.to_numpy());right.append(vz.to_numpy());names+=list(macros)
        for stock in STOCK_INTERACTIONS:
            for macro in interaction_macros:
                names.append(stock+' x '+macro);left.append((train_r[stock]*tz[macro]).to_numpy()[:,None]);right.append((test_r[stock]*vz[macro]).to_numpy()[:,None])
        audit.extend(dict(block='macro',feature=f,training_mean=center[f],training_std=scale[f],unique_training_sector_months=len(history)) for f in macros)
    a,b=np.hstack(left),np.hstack(right)
    assert a.shape[1]==b.shape[1]==len(names) and np.isfinite(a).all() and np.isfinite(b).all()
    return a,b,names,audit

def performance(returns):
    wealth=(1+returns).cumprod();dd=wealth/wealth.cummax().clip(lower=1)-1
    return pd.DataFrame(dict(months=len(returns),cumulative_return=wealth.iloc[-1]-1,annualized_return=wealth.iloc[-1]**(12/len(returns))-1,annualized_volatility=returns.std()*np.sqrt(12),max_drawdown=dd.min()))

def paired_test(values,block=3):
    x=np.asarray(values);n=len(x);rng=np.random.default_rng(42)
    starts=rng.integers(n,size=(5000,int(np.ceil(n/block))))
    ix=((starts[...,None]+np.arange(block))%n).reshape(5000,-1)[:,:n]
    draws=x[ix].mean(axis=1);mean=x.mean();lo,hi=np.quantile(draws,[.025,.975]);errors=draws-mean
    p=min(1.,2*min((1+np.count_nonzero(errors>=mean))/5001,(1+np.count_nonzero(errors<=mean))/5001))
    return dict(months=n,mean_monthly_difference=mean,ci_low=2*mean-hi,ci_high=2*mean-lo,p=p)

def bh(p):
    p=np.asarray(p);order=np.argsort(p);out=np.empty(len(p));out[order]=np.minimum(1,np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]);return out
