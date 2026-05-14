"""Inspect AI plugin: exposes Tinker checkpoints to Inspect as a model provider.

Registered via ``[project.entry-points.inspect_ai]`` in pyproject.toml, so
Inspect auto-discovers the ``latteries/...`` provider on startup. Users do
not need to import this package manually — just run:

    uv run inspect eval my_task.py \\
        --model latteries/{model_short}/{claim}/{condition}/{step}
"""
