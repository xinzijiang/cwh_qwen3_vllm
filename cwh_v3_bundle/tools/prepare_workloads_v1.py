#!/usr/bin/env python3
"""Build reproducible, unique real ShareGPT request sets, using Qwen3 tokenizer."""
import argparse
import hashlib
import json
import pathlib
import random


def extract_prompts(conversation):
    turns = conversation.get("conversations") or conversation.get("messages") or []
    history = []
    for turn in turns:
        role = str(turn.get("from", turn.get("role", ""))).lower()
        val = turn.get("value", turn.get("content", ""))
        if not isinstance(val, str):
            continue
        val = val.strip()
        if not val or len(val) > 32000 or "<image" in val.lower():
            continue
        label = "User" if role in ("human", "user") else "Assistant"
        if role in ("human", "user"):
            context = "\n\n".join(history[-5:])
            prompt = ((context + "\n\n") if context else "") + "User: " + val + "\nAssistant:"
            yield prompt
        history.append(label + ": " + val)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="ShareGPT JSON local path")
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--requests", type=int, default=800)
    p.add_argument("--seed", type=int, default=20261102)
    p.add_argument("--output-tokens", type=int, default=256)
    a = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True, trust_remote_code=False)
    source = pathlib.Path(a.source)
    if not source.is_file() or source.stat().st_size < 1000000:
        raise RuntimeError(f"Missing/too-small ShareGPT JSON: {source}. Download complete file first.")
    print("Reading", source, "bytes", source.stat().st_size, flush=True)
    records = json.loads(source.read_text())
    if isinstance(records, dict):
        records = records.get("data", records.get("conversations", []))
    if not isinstance(records, list):
        raise RuntimeError("ShareGPT root must be a JSON list of conversations")
    rng = random.Random(a.seed)
    rng.shuffle(records)
    pools = {"short": [], "long": []}
    # Distinct conversations primarily, to avoid prefix caching artifacts.
    seen = set()
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        for prompt in extract_prompts(record):
            if len(prompt) < 30:
                continue
            digest = hashlib.sha256(prompt.encode("utf8")).hexdigest()
            if digest in seen:
                continue
            ids = tok.encode(prompt, add_special_tokens=False)
            n = len(ids)
            bucket = "short" if 96 <= n <= 384 else "long" if 768 <= n <= 1536 else None
            if bucket and len(pools[bucket]) < a.requests:
                seen.add(digest)
                pools[bucket].append({"prompt": prompt, "input_tokens": n,
                                      "output_tokens": a.output_tokens, "source_hash": digest})
                # Only one prompt per conversation by preference.
                break
        if all(len(pools[k]) >= a.requests for k in pools):
            break
        if idx and idx % 10000 == 0:
            print("SCANNED", idx, "SELECTED", {k:len(v) for k,v in pools.items()}, flush=True)
    if any(len(v) < a.requests for v in pools.values()):
        raise RuntimeError(f"Not enough eligible distinct ShareGPT prompts, got: { {k:len(v) for k,v in pools.items()} }. "
                           "Try reducing --requests or extending token ranges in prepare_workloads.py; do not silently pad/duplicate.")
    out = pathlib.Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": "ShareGPT", "source": str(source), "source_bytes": source.stat().st_size,
                "seed":a.seed, "requests_per_bucket":a.requests,
                "output_tokens":a.output_tokens, "model": a.model, "workloads":{}}
    for name, rows in pools.items():
        path = out / (name + ".jsonl")
        with path.open("w") as f:
            for i, row in enumerate(rows):
                row["request_id"] = f"{name}_{i:04d}"
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["workloads"][name] = {
            "path": str(path), "count": len(rows),
            "min_input_tokens": min(r["input_tokens"] for r in rows),
            "max_input_tokens": max(r["input_tokens"] for r in rows),
            "mean_input_tokens": sum(r["input_tokens"] for r in rows)/len(rows)}
        print("READY", name, manifest["workloads"][name], flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
