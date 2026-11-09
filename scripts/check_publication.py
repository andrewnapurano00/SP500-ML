"""Reject private records/artifacts; allow only checksummed aggregate CSVs."""
import hashlib,json,pathlib,re,subprocess
root=pathlib.Path(__file__).resolve().parents[1]
allowed=json.loads((root/'results/provenance.json').read_text())['aggregate_files_sha256']
paths=subprocess.check_output(['git','ls-files','-z'],cwd=root).decode().split('\0')
failures=[]
for name in filter(None,paths):
    p=root/name
    if p.suffix.lower() in {'.parquet','.jsonl','.pkl','.pickle','.joblib','.zip','.whl','.pem','.key'} or name=='.env' or name.startswith(('data/','artifacts/','outputs/')):failures.append(name+': private artifact')
    if p.suffix=='.csv':
        if name not in allowed or hashlib.sha256(p.read_bytes()).hexdigest()!=allowed[name]:failures.append(name+': unapproved aggregate')
    if p.suffix=='.png':continue
    value=p.read_text(encoding='utf8')
    if re.search(r'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',value):failures.append(name+': credential pattern')
    if p.suffix=='.ipynb':
        if any(c.get('outputs') for c in json.loads(value)['cells']):failures.append(name+': cached notebook output')
if failures:raise SystemExit('Publication failed: '+', '.join(failures))
print('Publication check passed: source, figures and approved aggregate evidence only.')
