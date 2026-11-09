"""Application-level checkpoints. No CUDA context / KV-cache checkpoint claim."""
import hashlib
import json
import os
from pathlib import Path
import tempfile


def atomic_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(dir=path.parent,prefix=path.name+'.tmp_')
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(data,f,indent=2,default=str);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(name,path)
        dfd=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(dfd)
        finally:os.close(dfd)
    finally:
        if os.path.exists(name):os.unlink(name)


def append_jsonl(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(data,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())


def read_journal(path,repair=False):
    path=Path(path)
    if not path.exists():return []
    data=path.read_bytes();rows=[];used=0
    lines=data.splitlines(keepends=True)
    for i,line in enumerate(lines):
        try:
            if not line.endswith(b'\n'):raise ValueError('Incomplete last record')
            rows.append(json.loads(line))
        except (ValueError,UnicodeDecodeError):
            if i!=len(lines)-1:raise RuntimeError(f'Journal corrupt before final record: {path}')
            if repair:
                with path.open('r+b') as f:f.truncate(used);f.flush();os.fsync(f.fileno())
            break
        used+=len(line)
    return rows


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_copy(source,target):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(dir=target.parent,prefix=target.name+'.tmp_')
    try:
        with os.fdopen(fd,'wb') as f:
            f.write(Path(source).read_bytes());f.flush();os.fsync(f.fileno())
        os.replace(name,target)
        dfd=os.open(target.parent,os.O_RDONLY)
        try:os.fsync(dfd)
        finally:os.close(dfd)
    finally:
        if os.path.exists(name):os.unlink(name)
