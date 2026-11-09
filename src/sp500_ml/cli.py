import argparse
import json
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description='SP500-ML: work-in-progress research and private batch candidates')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ['eda','model']:
        p=sub.add_parser(name);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('prepare');p.add_argument('--parquet',required=True);p.add_argument('--source-json',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('export-gold');p.add_argument('--binding',required=True);p.add_argument('--profile',required=True);p.add_argument('--warehouse',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('train-candidate');p.add_argument('--input',required=True);p.add_argument('--signal-month',required=True);p.add_argument('--bundle',required=True)
    p=sub.add_parser('score-candidate');p.add_argument('--bundle',required=True);p.add_argument('--manifest-sha256',required=True);p.add_argument('--cohort',required=True);p.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.command=='eda':
        from .eda import build_report
        print(build_report(input_dir=args.input,output_dir=args.output)['validation'])
    elif args.command=='model':
        from .modeling import run
        run(research_dir=args.input,output_dir=args.output)
    elif args.command=='prepare':
        import pandas as pd
        from .dataset import prepare
        print(prepare(pd.read_parquet(args.parquet),args.output,json.loads(Path(args.source_json).read_text(encoding='utf8'))))
    elif args.command=='export-gold':
        from .dataset import export_gold
        print(export_gold(args.binding,args.profile,args.warehouse,args.output))
    elif args.command=='train-candidate':
        from .candidate import train
        from .dataset import digest
        train(args.input,args.signal_month,args.bundle)
        print('Private candidate manifest SHA256:',digest(Path(args.bundle)/'manifest.json'))
    else:
        import pandas as pd
        from .candidate import score
        output=Path(args.output)
        if output.exists():raise ValueError('Refusing to replace existing scores')
        result=score(args.bundle,pd.read_parquet(args.cohort),args.manifest_sha256)
        output.parent.mkdir(parents=True,exist_ok=True);result.to_parquet(output,index=False)

if __name__=='__main__':main()
