#!/usr/bin/env python3
"""Measure TTFT, TPOT, aggregate throughput via vLLM's streaming /v1/completions API.

Run against either the unchanged vLLM EP server or the fully implemented
C->W+Evict+Replicate server, with IDENTICAL arguments and prompts. This script
never enables or implements the optimization.
"""
import argparse,asyncio,csv,json,statistics,time
from pathlib import Path
import aiohttp

def percentile(values,p):
    if not values:return None
    a=sorted(values); pos=(len(a)-1)*p/100
    i=int(pos); frac=pos-i
    return a[i]*(1-frac)+(a[min(i+1,len(a)-1)])*frac

def read_questions(path):
    rows=[json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    if not rows:raise ValueError('No dataset questions')
    return rows

def build_prompt(tok,q):
    text=("Answer this science question briefly and explain your reasoning in one sentence.\n"
          f"Question: {q['question']}\n"
          + '\n'.join(f'{letter}. {q["choices"][letter]}' for letter in 'ABCD')+'\nAnswer:')
    return tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,
                                   add_generation_prompt=True,enable_thinking=False)

async def stream_one(session, url, model_name, prompt, request_id, n_tokens, tokenizer, ignore_eos=False):
    payload={'model':model_name,'prompt':prompt,'max_tokens':n_tokens,
             'temperature':0,'stream':True,'stream_options':{'include_usage':True}}
    if ignore_eos:payload['ignore_eos']=True
    start=time.perf_counter();first=None;last=None;received_text=[];usage_out=None
    async with session.post(url, json=payload) as res:
        body_error=None
        if res.status>=400:
            body_error=(await res.text())[:800]
            raise RuntimeError(f'HTTP {res.status}: {body_error}')
        buffer=''
        async for chunk in res.content.iter_any():
            buffer+=chunk.decode('utf-8',errors='replace')
            while '\n\n' in buffer:
                event,buffer=buffer.split('\n\n',1)
                for line in event.splitlines():
                    if not line.startswith('data:'):continue
                    content=line[5:].strip()
                    if content=='[DONE]':continue
                    try:packet=json.loads(content)
                    except json.JSONDecodeError:continue
                    usage=packet.get('usage')
                    if isinstance(usage,dict) and usage.get('completion_tokens') is not None:
                        usage_out=usage['completion_tokens']
                    for choice in packet.get('choices',[]):
                        delta=choice.get('text','')
                        if delta:
                            received_text.append(delta)
                            when=time.perf_counter()
                            if first is None:first=when
                            last=when
    end=time.perf_counter()
    text=''.join(received_text)
    # prefer server usage when available; otherwise tokenize returned completion
    nout=int(usage_out) if usage_out is not None else len(tokenizer.encode(text,add_special_tokens=False))
    if first is None or nout<=0:
        raise RuntimeError(f'Request {request_id} produced no streamed text')
    return {'request_id':request_id,'start_monotonic':start,'end_monotonic':end,
            'output_tokens':nout,'first_token_s':first-start,'request_wall_s':end-start,
            'tpot_s_per_token':(end-first)/(nout-1) if nout>1 else None,
            'output_token_source':'server_usage' if usage_out is not None else 'local_tokenizer',
            'n_chunks':len(received_text)}

async def main_async(args):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
    examples=read_questions(args.dataset_jsonl)
    base_url=args.base_url.rstrip('/')+'/v1/completions'
    timeout=aiohttp.ClientTimeout(total=600,connect=120)
    connector=aiohttp.TCPConnector(limit=max(args.concurrencies)*2,force_close=False)
    results=[];summaries=[]
    async with aiohttp.ClientSession(timeout=timeout,connector=connector) as sess:
        prompts=[build_prompt(tok,q) for q in examples]
        for i in range(args.warmup):
            await stream_one(sess,base_url,args.served_model,prompts[i%len(prompts)],f'warmup-{i}',
                             min(16,args.output_tokens),tok,args.ignore_eos)
        for conc in args.concurrencies:
            if conc>len(prompts):raise ValueError(f'Need at least {conc} questions; dataset has {len(prompts)}')
            for rep in range(args.repeats):
                jobs=[stream_one(sess,base_url,args.served_model,prompts[(i+rep*conc)%len(prompts)],
                                 f'c{conc}-r{rep}-q{i}',args.output_tokens,tok,args.ignore_eos)
                      for i in range(conc)]
                output=await asyncio.gather(*jobs)
                total=sum(x['output_tokens'] for x in output)
                duration=max(x['end_monotonic'] for x in output)-min(x['start_monotonic'] for x in output)
                ttf=[x['first_token_s'] for x in output]
                tpot=[x['tpot_s_per_token'] for x in output if x['tpot_s_per_token'] is not None]
                row={'mode':args.mode,'concurrency':conc,'repeat':rep,
                     'request_count':conc,'total_output_tokens':total,'duration_s':duration,
                     'aggregate_output_tokens_per_s':total/duration,
                     'median_ttft_s':statistics.median(ttf), 'p95_ttft_s':percentile(ttf,95),
                     'median_tpot_s_per_token':statistics.median(tpot) if tpot else None,
                     'p95_tpot_s_per_token':percentile(tpot,95)}
                summaries.append(row)
                for r in output:results.append({**r,'mode':args.mode,'concurrency':conc,'repeat':rep})
                print('RUN',json.dumps(row),flush=True)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    with (out/'per_request.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    metadata={'mode':args.mode,'model_name':args.served_model,'base_url':args.base_url,
              'note':'metrics measured through streaming API; includes client/server overhead and network if remote',
              'output_tokens_requested':args.output_tokens,'ignore_eos':args.ignore_eos,
              'concurrency':args.concurrencies,'repeats':args.repeats,'warmup':args.warmup,
              'eplb_status':'set manually based on server logs; not verified by this script',
              'optimization_status':'set manually based on server logs; not verified by this script'}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print('COMPLETED',out,flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base-url',default='http://127.0.0.1:8000')
    p.add_argument('--served-model',default='qwen3-moe')
    p.add_argument('--tokenizer',required=True,help='Local Hugging Face model path')
    p.add_argument('--dataset-jsonl',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--mode',choices=['native_ep','eplb','custom_full_method'],default='native_ep')
    p.add_argument('--concurrencies',default='1,4,16')
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--warmup',type=int,default=2)
    p.add_argument('--output-tokens',type=int,default=64)
    p.add_argument('--ignore-eos',action='store_true')
    a=p.parse_args();a.concurrencies=[int(x) for x in a.concurrencies.split(',')]
    if min(a.concurrencies)<1 or a.repeats<1:raise ValueError('Positive concurrency and repeats required')
    asyncio.run(main_async(a))
if __name__=='__main__':main()
