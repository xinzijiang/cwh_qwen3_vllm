import os
import runpy
import sys
import socket
import time
import inspect

OriginalSocket = socket.socket

class ReusableSocket(OriginalSocket):
    def bind(self, address):
        frame = inspect.currentframe().f_back

        is_probe = (
            frame is not None
            and frame.f_code.co_name == "__enter__"
            and frame.f_globals.get("__name__") == "__main__"
            and isinstance(frame.f_locals.get("self"), object)
            and type(frame.f_locals.get("self")).__name__ == "DurableServer"
            and isinstance(address, tuple)
            and address[0] == "127.0.0.1"
        )

        if is_probe:
            self.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            for attempt in range(60):
                try:
                    return super().bind(address)
                except OSError as e:
                    if e.errno != 98:
                        raise
                    print(f"[PORT_FIX] Retry {attempt+1}/60", flush=True)
                    time.sleep(2)

            raise OSError(98, "Port still unavailable")

        return super().bind(address)

socket.socket = ReusableSocket

print("[PORT_FIX] Controller patch installed", flush=True)

from pathlib import Path

PACK = Path("/work/cwh_qwen3_vllm/cwh_v3_bundle")
sys.path.insert(0, str(PACK / "tools"))

sys.argv = ["tools/session.py"] + sys.argv[1:]
runpy.run_path("tools/session.py", run_name="__main__")
