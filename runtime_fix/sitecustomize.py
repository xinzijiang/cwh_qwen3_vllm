import os
import socket
import time
import inspect

if os.environ.get("CWH_PORT_FIX") == "1":
    _original_socket = socket.socket

    class ReusableSocket(_original_socket):
        def bind(self, address):
            frame = inspect.currentframe().f_back

            is_probe = (
                frame is not None
                and frame.f_code.co_name == "__enter__"
                and frame.f_globals.get("__name__") == "__main__"
                and frame.f_locals.get("self") is not None
                and type(frame.f_locals["self"]).__name__ == "DurableServer"
                and isinstance(address, tuple)
                and address[0] == "127.0.0.1"
            )

            if not is_probe:
                return super().bind(address)

            self.setsockopt(
                socket.SOL_SOCKET,
                socket.SO_REUSEADDR,
                1
            )

            for attempt in range(60):
                try:
                    return super().bind(address)
                except OSError as e:
                    if e.errno != 98:
                        raise
                    print(
                        f"[PORT_FIX] Port {address[1]} busy; "
                        f"retry {attempt + 1}/60",
                        flush=True
                    )
                    time.sleep(2)

            raise OSError(
                98,
                f"Port {address[1]} unavailable after 120 seconds"
            )

    socket.socket = ReusableSocket
