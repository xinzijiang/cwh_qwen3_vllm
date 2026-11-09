"""Fixed-slot cold-to-warm / hot-replication research prototype.

Do not import vLLM or torch at module import time: plugin discovery may run
before vLLM workers fork / initialize CUDA.
"""
