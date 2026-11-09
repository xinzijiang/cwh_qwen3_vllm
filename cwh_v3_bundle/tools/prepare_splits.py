#!/usr/bin/env python3
"""Conversation-disjoint train/validation/test ShareGPT sets, no label leakage."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import os
from durable import atomic_json
from prepare_workloads_v1 import extract_prompts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--n',type=int,default=800)
    p.add_argument('--cal-n',type=int,default=400)
    a=p.parse_args()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    records=json.loads(a.source.read_text())
    if isinstance(records,dict):
        records=records['data']
    random.Random(20261103).shuffle(records)
    targets=dict(train=a.cal_n, validation=a.cal_n, diagnostic=a.n, test=a.n)
    pools={s:{w:[] for w in ('short','long')} for s in targets}
    seen_prompt=set(); seen_conversation=set()
    for rec in records:
        turns=rec.get('conversations') or rec.get('messages') or []
        conversation_hash=hashlib.sha256(json.dumps(turns,sort_keys=True).encode()).hexdigest()
        if conversation_hash in seen_conversation:
            continue
        # Never fill train and test from different turns of one conversation.
        for prompt in extract_prompts(rec):
            ph=hashlib.sha256(prompt.encode()).hexdigest()
            if ph in seen_prompt:
                continue
            n=len(tok.encode(prompt,add_special_tokens=False))
            bucket='short' if 96<=n<=384 else 'long' if 768<=n<=1536 else None
            if bucket is None:
                continue
            split=next((s for s in pools if len(pools[s][bucket])<targets[s]),None)
            if split:
                pools[split][bucket].append(dict(prompt=prompt,input_tokens=n,source_hash=ph,
                    conversation_hash=conversation_hash,request_id=f'{split}_{bucket}_{len(pools[split][bucket])}'))
                seen_prompt.add(ph);seen_conversation.add(conversation_hash)
                break
        if all(len(v)==targets[s] for s,w in pools.items() for v in w.values()):
            break
    if any(len(v)!=targets[s] for s,w in pools.items() for v in w.values()):
        raise RuntimeError('Insufficient unique conversations: do not pad or truncate prompts')
    a.out.mkdir(parents=True,exist_ok=True)
    manifest={}
    for s,ws in pools.items():
        for w,rows in ws.items():
            path=a.out/f'{s}_{w}.jsonl'
            path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
            manifest[path.name]=dict(count=len(rows),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if s in ('train','validation'):
            mixed=ws['short']+ws['long']
            random.Random(20261103).shuffle(mixed)
            path=a.out/f'{s}_mixed.jsonl'
            path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in mixed))
            manifest[path.name]=dict(count=len(mixed),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    for name in manifest:
        fd=os.open(a.out/name,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    atomic_json(a.out/'manifest.json',manifest)


if __name__=='__main__':main()
