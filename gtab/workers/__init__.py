"""GPU worker entry points, each run as `<gpu venv>\\Scripts\\python.exe -m gtab.workers.<name> request.json result.json`.

Only modules in this package may import torch, and only lazily (inside the handler), so that
`gtab.workers.base` itself stays importable from the core env.
"""
