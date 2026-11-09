"""Controller lifecycle. Fresh port for each server, bounded reuse checks."""
import errno,json,os,signal,socket,subprocess,time,urllib.request
from pathlib import Path
from durable import atomic_json
from run_suite import Server
STOP=False
class Paused(Exception):pass

def interrupted(*_):
    global STOP
    STOP=True

def enough(deadline,seconds=120):
    if STOP or time.time()+seconds>=deadline:raise Paused('Safe boundary / insufficient allocation time')

def child(command,deadline,log,monitor=None):
    enough(deadline,5);print('RUN',*map(str,command),flush=True)
    with Path(log).open('a') as f:
        p=subprocess.Popen(list(map(str,command)),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while p.poll() is None:
                if monitor is not None and monitor.poll() is not None:
                    raise RuntimeError('Server exited before readiness; inspect server.log next to '+str(log))
                enough(deadline,5);time.sleep(.5)
            if p.returncode==75:raise Paused('Checkpoint saved at layer/bucket boundary')
            if p.returncode:raise RuntimeError('Child exit %s; inspect %s'%(p.returncode,log))
        finally:
            if p.poll() is None:
                try:os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:os.killpg(p.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    p.wait()

def fresh_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1',0));return s.getsockname()[1]

class ManagedServer(Server):
    def __enter__(self):
        enough(self.a.deadline,120)
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            probe.bind(('127.0.0.1',self.a.port))
        self.log=(self.target/'server.log').open('w')
        atomic_json(self.target/'server_config.json',dict(command=self.cmd,environment={k:v for k,v in self.env.items() if k.startswith(('CWH_','VLLM_'))}))
        self.p=subprocess.Popen(self.cmd,env=self.env,stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            import sys
            tools=Path(__file__).parent
            child([sys.executable,tools/'wait_ready.py','--url',f'http://127.0.0.1:{self.a.port}','--pid',self.p.pid,'--seconds','1800'],self.a.deadline,self.target/'readiness.log',monitor=self.p)
            if '--hf-overrides' in self.cmd and self.env['CWH_MODE']!='v1':
                child([sys.executable,tools/'check_artifacts.py','weights',self.target/'weight_audits'],self.a.deadline,self.target/'weight_check.log')
            return self
        except BaseException:
            self.__exit__(None,None,None);raise
    def __exit__(self,*exc):
        # Separate process groups: stop the owned server and its workers only.
        super().__exit__(*exc)
