#!/usr/bin/env python3
"""Host Python 3.6-compatible release integrity check."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'MANIFEST.json').read_text())
for name,digest in manifest['files'].items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
        raise SystemExit('MISMATCH: '+name)
print('BUNDLE_VERIFIED',manifest['version'],len(manifest['files']),'files')
