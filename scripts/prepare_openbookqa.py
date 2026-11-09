#!/usr/bin/env python3
"""Download a named OpenBookQA main split to a portable JSONL file.
Run on a login node with internet and `pip install datasets`. No GPU required.
"""
import argparse,json,sys
from pathlib import Path

def normalize(row):
    labels=list(row['choices']['label']); choices=list(row['choices']['text'])
    if set(labels)!={'A','B','C','D'} or len(choices)!=4:
        raise ValueError(f"Unexpected OpenBookQA choices for {row.get('id')}: {labels}")
    correct=row['answerKey']
    if correct not in labels: raise ValueError(f"Invalid answerKey: {correct}")
    opts={k:str(v) for k,v in zip(labels,choices)}
    return {'id':row['id'],'question':row['question_stem'],'choices':opts,'answer':correct}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--split',choices=['train','validation','test'],default='validation')
    ap.add_argument('--out',required=True)
    ap.add_argument('--limit',type=int,default=0,help='0 downloads all examples')
    a=ap.parse_args()
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit('Install the datasets package on a network-enabled node: pip install datasets')
    data=load_dataset('allenai/openbookqa','main',split=a.split)
    n=len(data) if a.limit<=0 else min(a.limit,len(data))
    path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f:
        for row in data.select(range(n)):
            f.write(json.dumps(normalize(row),ensure_ascii=False)+'\n')
    print(json.dumps({'dataset':'allenai/openbookqa','config':'main','split':a.split,'rows':n,'out':str(path)}))
if __name__=='__main__':main()
