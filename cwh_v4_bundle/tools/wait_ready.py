#!/usr/bin/env python3
import argparse, os, time, urllib.request
p=argparse.ArgumentParser()
p.add_argument('--url',required=True);p.add_argument('--pid',type=int,required=True)
p.add_argument('--seconds',type=int,default=1500)
a=p.parse_args()
start=time.monotonic()
while time.monotonic()-start<a.seconds:
    try:
        os.kill(a.pid,0)
    except ProcessLookupError:
        raise SystemExit('SERVER PROCESS DIED before readiness')
    try:
        with urllib.request.urlopen(a.url+'/health',timeout=5) as r:
            if r.status==200:
                print('HTTP HEALTH READY',round(time.monotonic()-start,2),'s',flush=True)
                raise SystemExit(0)
    except Exception:
        pass
    time.sleep(5)
raise SystemExit('SERVER READY TIMEOUT')
