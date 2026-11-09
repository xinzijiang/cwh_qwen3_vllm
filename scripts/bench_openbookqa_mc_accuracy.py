#!/usr/bin/env python3
"""Real Qwen3 vLLM A/B OpenBookQA main validation multiple-choice accuracy.

It constrains the FIRST generated token to A/B/C/D using vLLM's
allowed_token_ids, then compares with official answerKey. Do not confuse this
with free-form generation accuracy or a PPL metric. Keep same scoring for A/B.
"""
import argparse,asyncio,csv,json,statistics,time
from pathlib import Path
import aiohttp


def choice_token_ids(tok):
    for prefix in (' ',''):
        enc={ch:tok.encode(prefix+ch,add_special_tokens=False) for ch in 'ABCD'}
        if all(len(v)==1 for v in enc.values()) and len({v[0] for v in enc.values()})==4:
            return {ch:v[0] for ch,v in enc.items()},prefix
    raise RuntimeError('The current Qwen tokenizer does not encode A/B/C/D as one token. '
                       'Use explicit scoring of multi-token labels instead of this script.')


def question_prompt(tok,row):
    s=('Choose the correct answer. Reply with ONLY a single capital letter: A, B, C, or D.\n'
       f'Question: {row["question"]}\n'
       + '\n'.join(f'{letter}. {row["choices"][letter]}' for letter in 'ABCD')
       +'\nYour answer:')
    return tok.apply_chat_template([{'role':'user','content':s}],tokenize=False,
                                   add_generation_prompt=True,enable_thinking=False)


async def main_async(args):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
    ids,prefix=choice_token_ids(tok)
    rows=[json.loads(s) for s in Path(args.dataset_jsonl).read_text().splitlines() if s.strip()]
    if args.limit: rows=rows[:args.limit]
    if not rows:raise ValueError('Empty OpenBookQA dataset')
    sem=asyncio.Semaphore(args.concurrency)
    endpoint=args.base_url.rstrip('/')+'/v1/completions'
    timeout=aiohttp.ClientTimeout(total=600)
    results=[]
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def one(idx,row):
            prompt=question_prompt(tok,row)
            payload={'model':args.served_model,'prompt':prompt,'temperature':0,
                     'max_tokens':1,'allowed_token_ids':list(ids.values()),'ignore_eos':True}
            async with sem:
                async with session.post(endpoint,json=payload) as resp:
                    if resp.status>=400:raise RuntimeError(f'HTTP {resp.status}: {(await resp.text())[:600]}')
                    body=await resp.json()
            txt=body['choices'][0]['text'].strip().upper()
            if txt not in 'ABCD' or len(txt)!=1:
                raise RuntimeError(f'Unexpected choice {txt!r} for question {idx}; check tokenizer/allowed_token_ids')
            return {'idx':idx,'id':row.get('id',''), 'gold':row['answer'],
                    'pred':txt,'correct':int(txt==row['answer'])}
        t0=time.perf_counter()
        results=await asyncio.gather(*(one(i,q) for i,q in enumerate(rows)))
    duration=time.perf_counter()-t0
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    with (out/'accuracy_per_question.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    summary={'mode':args.mode,'dataset':'OpenBookQA main','split':args.split,
             'questions':len(results),'correct':sum(r['correct'] for r in results),
             'accuracy_pct':100*statistics.mean(r['correct'] for r in results),
             'evaluation_definition':'choose best of four single token A/B/C/D using allowed_token_ids',
             'letter_token_ids':ids,'letter_prefix':prefix,
             'total_client_wall_s':duration,'not_a_throughput_benchmark':True}
    (out/'accuracy_summary.json').write_text(json.dumps(summary,indent=2))
    print('OPENBOOKQA',json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--tokenizer',required=True);p.add_argument('--dataset-jsonl',required=True)
    p.add_argument('--served-model',default='qwen3-moe')
    p.add_argument('--base-url',default='http://127.0.0.1:8000')
    p.add_argument('--out',required=True)
    p.add_argument('--mode',required=True)
    p.add_argument('--split',default='validation')
    p.add_argument('--concurrency',type=int,default=16)
    p.add_argument('--limit',type=int,default=500)
    a=p.parse_args();asyncio.run(main_async(a))
