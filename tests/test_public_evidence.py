import hashlib,json
from pathlib import Path
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]

def test_aggregate_provenance_and_headline():
    manifest=json.loads((ROOT/'results/provenance.json').read_text())
    for name,sha in manifest['aggregate_files_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha
    d=pd.read_csv(ROOT/'results/models/performance_summary.csv').set_index('model')
    assert d.loc['Technicals + balance_sheet','cumulative_return']==pytest.approx(2.022759,abs=1e-6)
    assert d.loc['VOO','months']==68
    t=pd.read_csv(ROOT/'results/models/paired_return_tests.csv')
    t=t.loc[t.model.eq('Technicals + balance_sheet')]
    assert (t.ci_low<0).all() and (t.ci_high>0).all() and (t.BH_q>.05).all()

def test_notebooks_have_no_cached_outputs():
    for p in (ROOT/'notebooks').glob('*.ipynb'):
        n=json.loads(p.read_text())
        assert all(not c.get('outputs') and c.get('execution_count') is None for c in n['cells'] if c['cell_type']=='code')
