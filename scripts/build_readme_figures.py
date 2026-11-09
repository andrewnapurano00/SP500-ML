"""Rebuild portfolio visuals from public aggregate evidence; no fitting or private reads."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/figures/readme'
OUT.mkdir(parents=True, exist_ok=True)
INK, MUTED, GRID = '#142C43', '#61758A', '#E7EDF3'
TEAL, BLUE, GOLD = '#008C82', '#4278C7', '#D59A32'
SERIES = {'Technicals + balance_sheet': ('Technicals + financial strength', TEAL),
          'Sector technical baseline': ('Technical + sector baseline', BLUE),
          'VOO': ('VOO benchmark', GOLD)}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'text.color': INK, 'axes.labelcolor': MUTED,
                     'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.spines.left': False, 'axes.edgecolor': GRID,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white',
                     'savefig.facecolor': 'white'})

def heading(fig, title, subtitle):
    fig.text(.065, .955, title, fontsize=23, weight='bold', va='top')
    fig.text(.065, .904, subtitle, fontsize=11, color=MUTED, va='top')

def save(fig, name, note):
    fig.text(.065, .027, note, fontsize=9, color=MUTED, va='bottom')
    fig.savefig(OUT / name, dpi=180)
    plt.close(fig)

monthly = pd.read_csv(ROOT/'results/models/matched_monthly_returns.csv', index_col=0, parse_dates=True)
annual = pd.read_csv(ROOT/'results/models/calendar_year_returns.csv', index_col=0)
summary = pd.read_csv(ROOT/'results/models/performance_summary.csv').set_index('model')
provenance = json.loads((ROOT/'results/provenance.json').read_text())
groups = json.loads((ROOT/'src/sp500_ml/feature_groups.json').read_text())
feature_count = sum(len(groups[k]) for k in ['technical_features','macro_features','sector_features','fundamental_raw_features','fundamental_score_features','fundamental_composites'])
# Verify yearly aggregation, dates and headline metrics before drawing.
calendar = monthly.copy()
calendar.index = (calendar.index.to_period('M')+1).to_timestamp('M')
recomputed = (1+calendar).groupby(calendar.index.year).prod()-1
np.testing.assert_allclose(recomputed[monthly.columns], annual[monthly.columns], atol=1e-12)
np.testing.assert_allclose((1+monthly).prod()-1, summary.loc[monthly.columns,'cumulative_return'], atol=1e-12)
assert len(monthly)==provenance['return_months']==68

fig = plt.figure(figsize=(14, 3.6), facecolor=INK)
fig.text(.055,.86,'S&P 500  /  FROM DATA TO DECISIONS',color='#7BE0D0',fontsize=12,weight='bold')
fig.text(.055,.65,'An end-to-end financial ML research project',color='white',fontsize=25,weight='bold')
for x, value, label in zip([.055,.295,.535,.775], [f"{provenance['research_rows']:,}",str(feature_count),str(provenance['models']),'Monthly'], ['stock-level observations','declared predictors','matched model variants','purged walk-forward evaluation']):
    fig.text(x,.34,value,color='white',fontsize=32,weight='bold')
    fig.text(x,.22,label,color='#C1D2E3',fontsize=10)
fig.text(.055,.13,'Each observation is one stock on one signal date; observations are not calendar months.',color='#C1D2E3',fontsize=9)
fig.text(.055,.035,'Databricks Gold  →  Exploratory analysis  →  Purged rolling validation  →  Private batch candidate',color='#C1D2E3',fontsize=10)
fig.savefig(OUT/'research_overview.png',dpi=180,facecolor=INK)
plt.close(fig)

fig, axes = plt.subplots(2,1,figsize=(14,8.3),sharex=True,gridspec_kw={'height_ratios':[2.3,1]})
fig.subplots_adjust(left=.075,right=.88,top=.81,bottom=.13,hspace=.14)
heading(fig,'How the research portfolios evolved','68 matched return months · March 2021–October 2026 · Equal-weight top 30 · Zero trading costs')
start=calendar.index.min()-pd.offsets.MonthEnd(1)
for key,(label,color) in SERIES.items():
    wealth=pd.concat([pd.Series([100.],index=[start]),100*(1+calendar[key]).cumprod()])
    axes[0].plot(wealth.index,wealth,color=color,lw=2.5,label=label)
    axes[0].annotate(f'${wealth.iloc[-1]:.0f}',(wealth.index[-1],wealth.iloc[-1]),xytext=(9,0),textcoords='offset points',color=color,weight='bold',va='center')
    drawdown=wealth/wealth.cummax()-1
    axes[1].plot(drawdown.index,drawdown,color=color,lw=1.7)
axes[0].set_ylabel('Growth of $100');axes[0].yaxis.set_major_formatter(FuncFormatter(lambda x,_:f'${x:,.0f}'))
axes[0].legend(loc='upper left',frameon=False,fontsize=10)
axes[1].set_ylabel('Drawdown');axes[1].yaxis.set_major_formatter(PercentFormatter(1));axes[1].set_xlabel('Realized return month')
for ax in axes:ax.grid(axis='y',color=GRID);ax.set_axisbelow(True)
save(fig,'performance_path.png','Historical development evidence, not a forecast. Higher stock-selection returns came with higher risk; paired advantages are not statistically established.')

fig,ax=plt.subplots(figsize=(14,7))
fig.subplots_adjust(left=.075,right=.96,top=.77,bottom=.17)
heading(fig,'Performance across market years','Compounded calendar-year returns · Same matched universe, dates and zero-cost top-30 policy')
x=np.arange(len(annual));width=.23
for offset,(key,(label,color)) in zip([-1,0,1],SERIES.items()):
    bars=ax.bar(x+offset*width,annual[key]*100,width,color=color,label=label,zorder=3)
    for i,(bar,value) in enumerate(zip(bars,annual[key]*100)):
        if annual.iloc[i]['months']<12:bar.set_hatch('//');bar.set_edgecolor('white');bar.set_linewidth(.6)
        label_pad=5.0 if offset==0 else 1.6
        ax.text(bar.get_x()+bar.get_width()/2,value+(label_pad if value>=0 else -label_pad),f'{value:.1f}%',ha='center',va='bottom' if value>=0 else 'top',fontsize=8.5,color=color,weight='bold')
labels=[f'{year}\n'+('Mar–Dec · 10 months' if year==2021 else 'Jan–Oct · 10 months' if year==2026 else '12 months') for year in annual.index]
ax.set_xticks(x,labels,fontsize=10);ax.set_ylim(-25,76);ax.set_ylabel('Compounded return');ax.yaxis.set_major_formatter(PercentFormatter(100))
ax.axhline(0,color=INK,lw=.8);ax.grid(axis='y',color=GRID);ax.set_axisbelow(True)
fig.legend(loc='upper left',bbox_to_anchor=(.065,.845),ncol=3,frameon=False,fontsize=10)
save(fig,'annual_returns.png','Hatching marks partial years: 2021 begins in March; 2026 ends in October. Annual returns are not annualized. Historical results are developmental.')

# Every fitted variant remains visible; retain original experiment ordering.
models=[name for name in summary.index if name not in ['VOO','Eligible universe equal weight']]
matrix=annual[models].T*100
fig,ax=plt.subplots(figsize=(14,8.8))
fig.subplots_adjust(left=.34,right=.92,top=.81,bottom=.14)
heading(fig,'The full experiment, year by year','All 12 fitted variants · Annual portfolio return (%) · No model selected for production from this table')
labels={'Sector technical baseline':'Technical + sector baseline','Fundamentals only':'Fundamentals only',
        'Technicals + fundamental scores':'Technicals + fundamental scores','Technicals + fundamental raw':'Technicals + raw fundamentals',
        'Technicals + raw and scores':'Technicals + raw + scores','Boosted technicals + fundamentals':'Boosted technicals + fundamentals',
        'Technicals + growth':'Technicals + growth','Technicals + profitability':'Technicals + profitability',
        'Technicals + valuation':'Technicals + valuation','Technicals + balance_sheet':'Technicals + financial strength',
        'Beat VOO technical baseline':'Beat-VOO target: technical baseline','Beat VOO technicals + fundamentals':'Beat-VOO target: technicals + fundamentals'}
im=ax.imshow(matrix,cmap='BrBG',norm=TwoSlopeNorm(vmin=-80,vcenter=0,vmax=80),aspect='auto')
ax.set_yticks(range(len(models)),[labels.get(m,m) for m in models],fontsize=10)
ax.set_xticks(range(len(annual)),[f'{y}*' if annual.loc[y,'months']<12 else str(y) for y in annual.index],fontsize=11)
ax.tick_params(axis='both',length=0,pad=9)
for row in range(len(models)):
    for col in range(len(annual)):
        value=matrix.iloc[row,col]
        ax.text(col,row,f'{value:+.1f}%',ha='center',va='center',fontsize=10,color='white' if abs(value)>45 else INK,weight='bold' if models[row]=='Technicals + balance_sheet' else 'normal')
ax.set_xticks(np.arange(-.5,len(annual),1),minor=True);ax.set_yticks(np.arange(-.5,len(models),1),minor=True);ax.grid(which='minor',color='white',lw=2);ax.tick_params(which='minor',length=0)
save(fig,'all_models_by_year.png','* Partial years: March–December 2021 and January–October 2026. Beat-VOO rows use a separate classification target; all rows use the same top-30 policy.')
print('Built four README visuals; yearly compounding and cumulative returns verified against published aggregates.')
